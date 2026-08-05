from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from research.config import ResearchConfig, load_config
from research.data.binance import (
    BinanceDataError,
    BinanceFuturesClient,
    collect_symbol,
    validate_symbols,
)
from research.data.storage import ensure_directories, load_frame, save_frame, save_json
from research.data.synthetic import SyntheticConfig, generate_synthetic_market
from research.features.build import build_features, merge_derivatives, model_feature_columns
from research.labels.events import add_labels, extract_events, extract_warning_samples
from research.models.baselines import (
    chronological_group_split,
    chronological_split,
    train_models,
    train_models_external_holdout,
    usable_feature_columns,
)
from research.models.evaluation import (
    evaluate_event_predictions,
    evaluate_prediction_frame,
)
from research.models.evaluation import (
    feature_availability as calculate_feature_availability,
)
from research.models.prospective import run_purged_walk_forward_loso
from research.monitor.preflight import run_monitor_preflight
from research.monitor.runtime import run_paper_replay
from research.reports.html import render_loso_report, render_prospective_report, render_report

app = typer.Typer(no_args_is_help=True, help="Pump/Dump Research v0.3.0.2 Controlled Intelligence Development")
console = Console()


def _raw_dir(symbol: str) -> Path:
    return Path("data/raw") / symbol.upper()


def _processed_path(symbol: str) -> Path:
    return Path("data/processed") / f"{symbol.upper()}.parquet"


def _events_path(symbol: str) -> Path:
    return Path("artifacts") / f"{symbol.upper()}_events.parquet"


def _samples_path(symbol: str) -> Path:
    return Path("artifacts") / f"{symbol.upper()}_warning_samples.parquet"


def _requested_base(value: str) -> str:
    cleaned = "".join(ch for ch in value.upper().strip() if ch.isalnum())
    return cleaned[:-4] if cleaned.endswith("USDT") else cleaned


def _resolve_local_symbol(value: str, *, processed: bool) -> str:
    """Resolve aliases such as PEPE to locally saved 1000PEPEUSDT data."""
    base = _requested_base(value)
    direct = f"{base}USDT"
    if processed:
        exact_exists = _processed_path(direct).exists()
        candidates = [path.stem.upper() for path in Path("data/processed").glob("*.parquet")]
    else:
        exact_exists = (_raw_dir(direct) / "klines.parquet").exists()
        candidates = [
            path.name.upper()
            for path in Path("data/raw").glob("*USDT")
            if (path / "klines.parquet").exists()
        ]
    if exact_exists:
        return direct
    scaled = sorted(
        symbol
        for symbol in candidates
        if symbol.endswith(f"{base}USDT")
        and symbol[: -len(f"{base}USDT")].isdigit()
    )
    if len(scaled) == 1:
        return scaled[0]
    if len(scaled) > 1:
        raise typer.BadParameter(
            f"Ambiguous local alias {value!r}: {', '.join(scaled)}. Use exact contract."
        )
    return direct


def _load_processed_symbols(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    events: list[pd.DataFrame] = []
    for value in symbols:
        symbol = _resolve_local_symbol(value, processed=True)
        frame = load_frame(_processed_path(symbol)).copy()
        frame["symbol"] = symbol
        frames.append(frame)
        if _events_path(symbol).exists():
            current_events = load_frame(_events_path(symbol))
            if not current_events.empty:
                events.append(current_events)
    dataset = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    combined_events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    return dataset, combined_events




def _load_market_coverage(symbols: list[str]) -> pd.DataFrame:
    """Load only per-symbol timestamp coverage without materializing full feature tables.

    Prospective evaluation needs market start/end timestamps solely to calculate
    monitored symbol-days. Reading every processed feature column for every minute
    can exceed Docker Desktop memory on year-long multi-symbol datasets.
    """
    rows: list[dict[str, object]] = []
    for value in symbols:
        symbol = _resolve_local_symbol(value, processed=True)
        source = _processed_path(symbol)
        try:
            timestamps = pd.read_parquet(source, columns=["timestamp"])["timestamp"]
        except (ImportError, ModuleNotFoundError, ValueError, KeyError):
            timestamps = load_frame(source)["timestamp"]
        parsed = pd.to_datetime(timestamps, utc=True, errors="coerce").dropna()
        if parsed.empty:
            rows.append(
                {
                    "symbol": symbol,
                    "market_start": pd.NaT,
                    "market_end": pd.NaT,
                }
            )
        else:
            rows.append(
                {
                    "symbol": symbol,
                    "market_start": parsed.min(),
                    "market_end": parsed.max(),
                }
            )
    return pd.DataFrame(rows)

def _load_warning_samples(symbols: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for value in symbols:
        symbol = _resolve_local_symbol(value, processed=True)
        path = _samples_path(symbol)
        if path.exists():
            frame = load_frame(path).copy()
            if frame.empty:
                continue
            frame["symbol"] = symbol
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp")


def _is_dump_target(target: str) -> bool:
    prefixes = ("correction_", "strong_", "dump_", "extreme_", "pump_drawdown_class_")
    return target.startswith(prefixes)


def _normalize_regime(value: str) -> str:
    regime = value.strip().upper()
    allowed = {"ALL", "FAST", "MEDIUM", "SLOW"}
    if regime not in allowed:
        raise typer.BadParameter(
            f"Unknown regime {value!r}. Use one of: {', '.join(sorted(allowed))}."
        )
    return regime


def _filter_regime(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    normalized = _normalize_regime(regime)
    if frame.empty or normalized == "ALL":
        return frame
    if "pump_regime" not in frame:
        raise typer.BadParameter(
            "pump_regime is missing. Rebuild the selected symbols with v0.2.2."
        )
    return frame[frame["pump_regime"].astype(str).str.upper() == normalized].copy()


def _resolved_symbols(symbols: list[str]) -> list[str]:
    return [_resolve_local_symbol(value, processed=True) for value in symbols]


def _prepare_training_dataset(
    symbols: list[str],
    target: str,
    dataset_mode: str,
    regime: str = "ALL",
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    full, events = _load_processed_symbols(symbols)
    events = _filter_regime(events, regime) if not events.empty else events
    if dataset_mode == "warnings" and _is_dump_target(target):
        warning = _filter_regime(_load_warning_samples(symbols), regime)
        if not warning.empty and target in warning:
            accepted = set(events["event_id"].astype(str)) if not events.empty else set()
            if accepted:
                warning = warning[warning["event_id"].astype(str).isin(accepted)].copy()
            return warning, events, "event_id"
    if target not in full:
        raise typer.BadParameter(f"Unknown target: {target}")
    if _is_dump_target(target):
        full = full[(full["pump_context"] == 1) & full["event_id"].notna()].copy()
        return full, events, "event_id"
    return full, events, None


def _split_frames(
    dataset: pd.DataFrame,
    config: ResearchConfig,
    group_column: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if group_column:
        return chronological_group_split(
            dataset,
            config.training.train_fraction,
            config.training.validation_fraction,
            group_column,
        )
    return chronological_split(
        dataset,
        config.training.train_fraction,
        config.training.validation_fraction,
    )


def _split_class_summary(
    dataset: pd.DataFrame,
    target: str,
    config: ResearchConfig,
    group_column: str | None,
) -> list[dict[str, object]]:
    usable = dataset.dropna(subset=["timestamp", target]).copy()
    train, validation, test = _split_frames(usable, config, group_column)
    rows: list[dict[str, object]] = []
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        positives = int(pd.to_numeric(frame[target], errors="coerce").fillna(0).sum())
        total = len(frame)
        event_count = int(frame[group_column].nunique()) if group_column and total else 0
        positive_events = (
            int(frame.loc[frame[target] == 1, group_column].nunique())
            if group_column and total
            else 0
        )
        rows.append(
            {
                "split": name,
                "rows": total,
                "positives": positives,
                "negatives": total - positives,
                "positive_rate": positives / total if total else 0.0,
                "classes": int(frame[target].nunique()) if total else 0,
                "events": event_count,
                "positive_events": positive_events,
                "start": frame["timestamp"].min() if total else None,
                "end": frame["timestamp"].max() if total else None,
            }
        )
    return rows


def _print_class_summary(rows: list[dict[str, object]], target: str) -> None:
    table = Table(
        "Split", "Rows", "Positive", "Negative", "Rate", "Classes", "Events", "+Events"
    )
    for row in rows:
        table.add_row(
            str(row["split"]),
            f"{int(row['rows']):,}",
            f"{int(row['positives']):,}",
            f"{int(row['negatives']):,}",
            f"{float(row['positive_rate']):.4%}",
            str(row["classes"]),
            str(row["events"]),
            str(row["positive_events"]),
        )
    console.print(f"[bold]Target:[/bold] {target}")
    console.print(table)


def _build_one(symbol_value: str, config: ResearchConfig) -> tuple[str, int, int, int]:
    symbol = _resolve_local_symbol(symbol_value, processed=False)
    source = _raw_dir(symbol)
    klines = load_frame(source / "klines.parquet")
    funding = load_frame(source / "funding.parquet") if (source / "funding.parquet").exists() else None
    open_interest = (
        load_frame(source / "open_interest.parquet")
        if (source / "open_interest.parquet").exists()
        else None
    )
    merged = merge_derivatives(klines, funding, open_interest)
    featured = build_features(merged, config.features)
    labelled = add_labels(featured, config.labels)
    save_frame(labelled, _processed_path(symbol))
    events = extract_events(labelled, config.labels)
    save_frame(events, _events_path(symbol))
    samples = extract_warning_samples(
        labelled, events, config.labels, model_feature_columns(labelled)
    )
    save_frame(samples, _samples_path(symbol))
    return symbol, len(labelled), len(events), len(samples)


@app.command()
def version() -> None:
    """Print program version."""
    console.print("Pump/Dump Research [bold]v0.3.0.2[/bold]")


@app.command()
def symbols(
    limit: Annotated[int, typer.Option(min=1, max=200, help="Number of symbols")] = 30,
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """List active Binance USDT perpetuals sorted by quote volume."""
    config = load_config(config_path)

    async def run() -> pd.DataFrame:
        async with BinanceFuturesClient(config.binance) as client:
            return await client.top_symbols(limit)

    frame = asyncio.run(run())
    table = Table("Symbol", "Last", "24h %", "Quote volume", "Trades")
    for row in frame.itertuples(index=False):
        table.add_row(
            row.symbol,
            f"{row.last_price:g}",
            f"{row.price_change_pct:.2f}",
            f"{row.quote_volume:,.0f}",
            f"{row.trades:,}",
        )
    console.print(table)


@app.command()
def collect(
    symbol: Annotated[str, typer.Argument(help="Example: KOMA, PEPE or exact contract")],
    days: Annotated[int, typer.Option(min=1, max=365)] = 90,
    interval: Annotated[str, typer.Option()] = "1m",
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Download public Futures candles, funding and best-effort OI history."""
    ensure_directories()
    config = load_config(config_path)

    async def run() -> tuple[str, dict[str, pd.DataFrame], list[str]]:
        async with BinanceFuturesClient(config.binance) as client:
            active = await client.active_usdt_perpetuals()
        normalized = validate_symbols([symbol], active)[0]
        datasets, notices = await collect_symbol(config.binance, normalized, days, interval)
        return normalized, datasets, notices

    try:
        normalized, datasets, notices = asyncio.run(run())
    except BinanceDataError as exc:
        console.print(f"[bold red]Collection stopped:[/bold red] {exc}")
        raise typer.Exit(code=2) from None
    requested = symbol.upper().strip()
    if normalized != requested and normalized != f"{requested}USDT":
        console.print(f"[cyan]Resolved Binance contract:[/cyan] {symbol} -> {normalized}")
    for notice in notices:
        console.print(f"[yellow]Notice:[/yellow] {notice}")
    destination = _raw_dir(normalized)
    for name, frame in datasets.items():
        path = save_frame(frame, destination / f"{name}.parquet")
        console.print(f"[green]Saved[/green] {name}: {len(frame):,} rows -> {path}")


@app.command(name="collect-many")
def collect_many(
    symbols: Annotated[list[str], typer.Argument(help="Several assets, e.g. KOMA PEPE WIF")],
    days: Annotated[int, typer.Option(min=1, max=365)] = 180,
    interval: Annotated[str, typer.Option()] = "1m",
    build_after: Annotated[bool, typer.Option("--build/--no-build")] = True,
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Collect several contracts sequentially and optionally build v0.2 datasets."""
    ensure_directories()
    config = load_config(config_path)

    async def run() -> list[tuple[str, str, dict[str, pd.DataFrame] | None, list[str]]]:
        async with BinanceFuturesClient(config.binance) as client:
            active = await client.active_usdt_perpetuals()
        results: list[tuple[str, str, dict[str, pd.DataFrame] | None, list[str]]] = []
        for requested in symbols:
            try:
                normalized = validate_symbols([requested], active)[0]
                datasets, notices = await collect_symbol(
                    config.binance, normalized, days, interval
                )
                results.append((requested, normalized, datasets, notices))
            except BinanceDataError as exc:
                results.append((requested, "", None, [str(exc)]))
        return results

    results = asyncio.run(run())
    completed: list[str] = []
    for requested, normalized, datasets, notices in results:
        if datasets is None:
            console.print(f"[bold red]Skipped {requested}:[/bold red] {notices[0]}")
            continue
        if normalized != requested.upper() and normalized != f"{requested.upper()}USDT":
            console.print(f"[cyan]Resolved:[/cyan] {requested} -> {normalized}")
        for notice in notices:
            console.print(f"[yellow]{normalized}:[/yellow] {notice}")
        destination = _raw_dir(normalized)
        for name, frame in datasets.items():
            save_frame(frame, destination / f"{name}.parquet")
        completed.append(normalized)
        console.print(f"[green]Collected[/green] {normalized}: {len(datasets['klines']):,} candles")

    if build_after:
        for symbol in completed:
            resolved, rows, event_count, samples = _build_one(symbol, config)
            console.print(
                f"[green]Built[/green] {resolved}: {rows:,} rows; "
                f"events={event_count}; warning samples={samples}"
            )


@app.command()
def migrate(
    source: Annotated[str, typer.Argument(help="Old project folder, e.g. ../pump-dump-research-v0.1.3")],
) -> None:
    """Copy raw market data from an older project; processed labels are rebuilt."""
    ensure_directories()
    source_path = Path(source).expanduser().resolve()
    raw_source = source_path / "data" / "raw"
    if not raw_source.exists():
        console.print(f"[bold red]Raw data folder not found:[/bold red] {raw_source}")
        raise typer.Exit(code=2)
    copied = 0
    for directory in raw_source.iterdir():
        if directory.is_dir() and (directory / "klines.parquet").exists():
            shutil.copytree(directory, _raw_dir(directory.name), dirs_exist_ok=True)
            copied += 1
    console.print(f"[green]Migrated[/green] raw datasets: {copied}")
    console.print("Run: [cyan]docker compose run --rm research rebuild[/cyan]")


@app.command(name="build")
def build_dataset(
    symbols: Annotated[list[str], typer.Argument(help="One or more locally collected symbols")],
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Build v0.2.4 features, cleaned events and leakage-safe warning snapshots."""
    ensure_directories()
    config = load_config(config_path)
    for value in symbols:
        try:
            symbol, rows, events, samples = _build_one(value, config)
        except FileNotFoundError as exc:
            console.print(f"[bold red]Build stopped for {value}:[/bold red] {exc}")
            continue
        console.print(
            f"[green]Built[/green] {symbol}: {rows:,} rows; "
            f"independent events: {events}; warning samples: {samples}"
        )


@app.command()
def rebuild(
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Rebuild every locally stored raw symbol with v0.2.4 cleaned events."""
    ensure_directories()
    config = load_config(config_path)
    symbols = sorted(
        path.name for path in Path("data/raw").glob("*USDT") if (path / "klines.parquet").exists()
    )
    if not symbols:
        console.print("[yellow]No raw symbols found.[/yellow]")
        raise typer.Exit(code=2)
    for symbol in symbols:
        resolved, rows, events, samples = _build_one(symbol, config)
        console.print(
            f"[green]Built[/green] {resolved}: {rows:,} rows; events={events}; samples={samples}"
        )


@app.command()
def events(
    symbols: Annotated[list[str], typer.Argument(help="One or more processed symbols")],
    limit: Annotated[int, typer.Option(min=1, max=200)] = 30,
) -> None:
    """Show independent pump events and peak-relative future drawdowns."""
    _, combined = _load_processed_symbols(symbols)
    if combined.empty:
        console.print("[yellow]No independent pump events found.[/yellow]")
        return
    columns = [
        column
        for column in [
            "symbol",
            "pump_start",
            "trigger_modes",
            "pump_regime",
            "merged_event_count",
            "pump_return_to_peak",
            "event_pump_threshold",
            "minutes_to_peak",
            "peak_drawdown_5m",
            "peak_drawdown_15m",
            "peak_drawdown_30m",
            "peak_drawdown_60m",
            "max_event_drawdown_class",
        ]
        if column in combined
    ]
    table = Table(*columns)
    for row in combined.sort_values("pump_start").tail(limit)[columns].itertuples(index=False):
        values = []
        for column, value in zip(columns, row, strict=True):
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        table.add_row(*values)
    console.print(table)


@app.command()
def scan(
    symbols: Annotated[list[str], typer.Argument(help="One or more processed symbols")],
    regime: Annotated[str, typer.Option("--regime")] = "ALL",
) -> None:
    """Compare predeclared targets on cleaned, independent pump events."""
    _, events_frame = _load_processed_symbols(symbols)
    events_frame = _filter_regime(events_frame, regime) if not events_frame.empty else events_frame
    samples = _filter_regime(_load_warning_samples(symbols), regime)
    targets = [
        "correction_3_5m",
        "strong_5_5m",
        "dump_8_5m",
        "correction_3_15m",
        "strong_5_15m",
        "dump_8_15m",
        "correction_3_30m",
        "strong_5_30m",
        "dump_8_30m",
        "dump_8_60m",
    ]
    table = Table("Target", "Positive snapshots", "Positive events", "Rate")
    for target in targets:
        if target not in samples:
            continue
        positive = samples[samples[target] == 1]
        table.add_row(
            target,
            f"{len(positive):,}",
            str(positive["event_id"].nunique()),
            f"{len(positive) / max(len(samples), 1):.4%}",
        )
    console.print(table)
    console.print(
        f"Cleaned independent events: [bold]{len(events_frame)}[/bold]; "
        f"warning snapshots: [bold]{len(samples):,}[/bold]; "
        f"regime: [bold]{_normalize_regime(regime)}[/bold]"
    )


@app.command()
def diagnose(
    symbols: Annotated[list[str], typer.Argument(help="One or more processed symbols")],
    target: Annotated[str, typer.Option()] = "correction_3_15m",
    dataset_mode: Annotated[str, typer.Option("--dataset")] = "warnings",
    regime: Annotated[str, typer.Option("--regime")] = "ALL",
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Show row and independent-event counts in leakage-safe time splits."""
    config = load_config(config_path)
    dataset, _, group_column = _prepare_training_dataset(symbols, target, dataset_mode, regime)
    if dataset.empty:
        console.print("[bold red]No eligible samples.[/bold red] Rebuild symbols with v0.2.")
        raise typer.Exit(code=2)
    rows = _split_class_summary(dataset, target, config, group_column)
    _print_class_summary(rows, target)
    console.print(f"Regime: [bold]{_normalize_regime(regime)}[/bold]")
    insufficient = [row["split"] for row in rows if int(row["classes"]) < 2]
    if insufficient:
        console.print(
            "[yellow]Not enough class diversity in: "
            + ", ".join(map(str, insufficient))
            + ". Add history/symbols or test a less severe predeclared target.[/yellow]"
        )
    if group_column:
        low_events = [
            row["split"]
            for row in rows
            if int(row["events"]) < config.training.minimum_events_per_split
        ]
        if low_events:
            console.print(
                "[yellow]Too few independent events in: "
                + ", ".join(map(str, low_events))
                + f". Recommended minimum: {config.training.minimum_events_per_split}.[/yellow]"
            )



@app.command(name="availability")
def availability_command(
    symbols: Annotated[list[str], typer.Argument(help="One or more processed symbols")],
    target: Annotated[str, typer.Option()] = "correction_3_15m",
    dataset_mode: Annotated[str, typer.Option("--dataset")] = "warnings",
    regime: Annotated[str, typer.Option("--regime")] = "ALL",
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Show feature availability by train/validation/test split."""
    ensure_directories()
    config = load_config(config_path)
    dataset, _, group_column = _prepare_training_dataset(
        symbols, target, dataset_mode, regime
    )
    if dataset.empty:
        console.print("[bold red]No eligible samples.[/bold red]")
        raise typer.Exit(code=2)
    train_frame, validation_frame, test_frame = _split_frames(
        dataset.dropna(subset=["timestamp", target]), config, group_column
    )
    requested = model_feature_columns(dataset)
    availability = calculate_feature_availability(
        {"train": train_frame, "validation": validation_frame, "test": test_frame},
        requested,
    )
    usable, dropped = usable_feature_columns(train_frame, requested)
    table = Table("Feature", "Train", "Validation", "Test", "Status")
    for row in availability.itertuples(index=False):
        feature = str(row.feature)
        status = "USED" if feature in usable else "DROPPED"
        table.add_row(
            feature,
            f"{float(row.train_available):.1%}",
            f"{float(row.validation_available):.1%}",
            f"{float(row.test_available):.1%}",
            status,
        )
    console.print(table)
    availability.to_csv("artifacts/feature_availability.csv", index=False)
    console.print(
        f"Effective features: [bold]{len(usable)}[/bold]; dropped all-null in train: "
        + (", ".join(dropped) if dropped else "none")
    )


@app.command()
def train(
    symbols: Annotated[list[str], typer.Argument(help="One or more processed symbols")],
    target: Annotated[str, typer.Option()] = "correction_3_15m",
    dataset_mode: Annotated[str, typer.Option("--dataset")] = "warnings",
    regime: Annotated[str, typer.Option("--regime")] = "ALL",
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Run chronological baselines with event metrics and bootstrap intervals."""
    ensure_directories()
    config = load_config(config_path)
    dataset, combined_events, group_column = _prepare_training_dataset(
        symbols, target, dataset_mode, regime
    )
    if dataset.empty:
        console.print("[bold red]Training stopped:[/bold red] no eligible samples.")
        raise typer.Exit(code=2)
    split_rows = _split_class_summary(dataset, target, config, group_column)
    _print_class_summary(split_rows, target)
    insufficient = [str(row["split"]) for row in split_rows if int(row["classes"]) < 2]
    low_events = [
        str(row["split"])
        for row in split_rows
        if group_column and int(row["events"]) < config.training.minimum_events_per_split
    ]
    if insufficient or low_events:
        console.print("[bold red]Training stopped safely.[/bold red]")
        if insufficient:
            console.print("One-class splits: " + ", ".join(insufficient))
        if low_events:
            console.print("Too few independent events: " + ", ".join(low_events))
        raise typer.Exit(code=2)

    clean = dataset.dropna(subset=["timestamp", target]).copy()
    train_frame, validation_frame, test_frame = _split_frames(clean, config, group_column)
    requested_features = model_feature_columns(dataset)
    effective_features, dropped_features = usable_feature_columns(
        train_frame, requested_features
    )
    availability = calculate_feature_availability(
        {"train": train_frame, "validation": validation_frame, "test": test_frame},
        requested_features,
    )
    availability.to_csv("artifacts/feature_availability.csv", index=False)
    save_json(
        {"effective": effective_features, "dropped_all_null_train": dropped_features},
        "artifacts/feature_manifest.json",
    )
    if dropped_features:
        console.print(
            "[yellow]Dropped all-null train features:[/yellow] "
            + ", ".join(dropped_features)
        )

    outputs = train_models(
        dataset,
        effective_features,
        target,
        config.training,
        group_column=group_column,
    )
    metrics: dict[str, dict[str, object]] = {}
    prediction_frames: list[pd.DataFrame] = []
    event_frames: list[pd.DataFrame] = []
    for output in outputs:
        values = dict(output.metrics)
        values["threshold"] = output.threshold
        metrics[output.name] = values
        prediction = output.predictions.copy()
        prediction["model"] = output.name
        prediction_frames.append(prediction)
        event_prediction = output.event_predictions.copy()
        event_prediction["model"] = output.name
        event_frames.append(event_prediction)
        console.print(
            f"{output.name}: PR-AUC={values['average_precision']:.3f}, "
            f"snapshot P/R={values['precision']:.3f}/{values['recall']:.3f}, "
            f"event P/R={values['event_precision']:.3f}/{values['event_recall']:.3f}, "
            f"false events/day={values['false_events_per_day']:.3f}"
        )

    save_json(metrics, "artifacts/metrics.json")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        "artifacts/predictions.csv", index=False
    )
    pd.concat(event_frames, ignore_index=True).to_csv(
        "artifacts/event_predictions.csv", index=False
    )
    provenance = {
        "source": "REAL_BINANCE_FUTURES",
        "symbols": _resolved_symbols(symbols),
        "start": dataset["timestamp"].min(),
        "end": dataset["timestamp"].max(),
        "rows": len(dataset),
        "events": int(dataset[group_column].nunique()) if group_column else len(combined_events),
        "regime": _normalize_regime(regime),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(provenance, "artifacts/run_manifest.json")
    report = render_report(
        metrics,
        combined_events,
        target,
        effective_features,
        "artifacts/report.html",
        version="0.3.0.2",
        provenance=provenance,
        split_summary=split_rows,
        feature_availability=availability,
        dropped_features=dropped_features,
    )
    console.print(f"[bold green]Report:[/bold green] {report}")


@app.command()
def loso(
    symbols: Annotated[list[str], typer.Argument(help="Symbols for leave-one-symbol-out")],
    target: Annotated[str, typer.Option()] = "dump_8_15m",
    regime: Annotated[str, typer.Option("--regime")] = "ALL",
    models: Annotated[str, typer.Option("--models", help="random_forest or comma-separated names/all")] = "random_forest",
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Exploratory LOSO with fold-specific thresholds; use prospective for strict evaluation."""
    ensure_directories()
    config = load_config(config_path)
    dataset, _, group_column = _prepare_training_dataset(
        symbols, target, "warnings", regime
    )
    if dataset.empty or group_column is None:
        console.print("[bold red]LOSO stopped:[/bold red] no event snapshots.")
        raise typer.Exit(code=2)
    requested_models = None if models.strip().lower() == "all" else [
        value.strip() for value in models.split(",") if value.strip()
    ]
    feature_columns = model_feature_columns(dataset)
    holdouts = sorted(dataset["symbol"].astype(str).unique())
    fold_rows: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []

    for holdout_symbol in holdouts:
        train_pool = dataset[dataset["symbol"].astype(str) != holdout_symbol].copy()
        holdout = dataset[dataset["symbol"].astype(str) == holdout_symbol].copy()
        if holdout.empty:
            continue
        try:
            outputs, used, dropped = train_models_external_holdout(
                train_pool,
                holdout,
                feature_columns,
                target,
                config.training,
                group_column=group_column,
                model_names=requested_models,
            )
        except ValueError as exc:
            fold_rows.append(
                {
                    "holdout_symbol": holdout_symbol,
                    "model": "вЂ”",
                    "status": f"SKIPPED: {exc}",
                    "test_rows": len(holdout),
                    "test_events": int(holdout[group_column].nunique()),
                }
            )
            console.print(f"[yellow]Skipped {holdout_symbol}:[/yellow] {exc}")
            continue
        for output in outputs:
            prediction = output.predictions.copy()
            prediction["model"] = output.name
            prediction["holdout_symbol"] = holdout_symbol
            prediction["fold_threshold"] = output.threshold
            all_predictions.append(prediction)
            values = output.metrics
            fold_rows.append(
                {
                    "holdout_symbol": holdout_symbol,
                    "model": output.name,
                    "status": "OK",
                    "test_rows": len(holdout),
                    "test_events": int(holdout[group_column].nunique()),
                    "positive_snapshots": int(holdout[target].sum()),
                    "positive_events": int(
                        holdout.loc[holdout[target] == 1, group_column].nunique()
                    ),
                    "features_used": len(used),
                    "features_dropped": ",".join(dropped),
                    "threshold": output.threshold,
                    "average_precision": values.get("average_precision"),
                    "precision": values.get("precision"),
                    "recall": values.get("recall"),
                    "event_precision": values.get("event_precision"),
                    "event_recall": values.get("event_recall"),
                    "false_events_per_day": values.get("false_events_per_day"),
                }
            )
        console.print(
            f"[green]LOSO[/green] {holdout_symbol}: rows={len(holdout)}, "
            f"events={holdout[group_column].nunique()}"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv("artifacts/loso_metrics.csv", index=False)
    if not all_predictions:
        console.print("[bold red]LOSO produced no valid folds.[/bold red]")
        raise typer.Exit(code=2)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv("artifacts/loso_predictions.csv", index=False)
    pooled: dict[str, dict[str, object]] = {}
    pooled_events: list[pd.DataFrame] = []
    for model_name, frame in predictions.groupby("model", sort=False):
        snapshot = evaluate_prediction_frame(frame).as_dict()
        event_metrics, event_table = evaluate_event_predictions(
            frame,
            bootstrap_repeats=config.training.bootstrap_repeats,
            random_state=config.training.random_state,
        )
        snapshot.update(event_metrics)
        snapshot["threshold"] = None
        pooled[str(model_name)] = snapshot
        event_table["model"] = model_name
        pooled_events.append(event_table)
    save_json(pooled, "artifacts/loso_summary.json")
    pd.concat(pooled_events, ignore_index=True).to_csv(
        "artifacts/loso_event_predictions.csv", index=False
    )
    report = render_loso_report(
        fold_metrics,
        pooled,
        "artifacts/loso_report.html",
        target,
        version="0.3.0.2",
    )
    console.print(f"[bold green]LOSO report:[/bold green] {report}")


@app.command(name="prospective")
def prospective_command(
    symbols: Annotated[list[str], typer.Argument(help="Symbols for purged walk-forward LOSO")],
    target: Annotated[str, typer.Option()] = "dump_8_15m",
    regime: Annotated[str, typer.Option("--regime")] = "ALL",
    model: Annotated[str, typer.Option("--model")] = "random_forest",
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Run past-only LOSO with one global validation threshold."""
    ensure_directories()
    config = load_config(config_path)
    warning, _, group_column = _prepare_training_dataset(
        symbols, target, "warnings", regime
    )
    if warning.empty or group_column is None:
        console.print("[bold red]Prospective evaluation stopped:[/bold red] no event snapshots.")
        raise typer.Exit(code=2)
    console.print("[cyan]Loading compact market coverage...[/cyan]")
    market_coverage = _load_market_coverage(symbols)
    features = model_feature_columns(warning)
    console.print(
        f"[cyan]Running purged walk-forward evaluation on {warning['event_id'].nunique()} events...[/cyan]"
    )
    try:
        result = run_purged_walk_forward_loso(
            warning,
            market_coverage,
            features,
            target,
            config.training,
            model_name=model,
            group_column=group_column,
        )
    except ValueError as exc:
        console.print(f"[bold red]Prospective evaluation stopped:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    result.calibration_predictions.to_csv(
        "artifacts/prospective_calibration_predictions.csv", index=False
    )
    result.evaluation_predictions.to_csv(
        "artifacts/prospective_predictions.csv", index=False
    )
    result.event_predictions.to_csv(
        "artifacts/prospective_event_predictions.csv", index=False
    )
    result.fold_log.to_csv("artifacts/prospective_folds.csv", index=False)
    result.symbol_metrics.to_csv("artifacts/prospective_symbol_metrics.csv", index=False)
    result.exposure.to_csv("artifacts/prospective_exposure.csv", index=False)
    save_json(result.summary, "artifacts/prospective_summary.json")
    report = render_prospective_report(
        result.summary,
        result.symbol_metrics,
        result.fold_log,
        result.exposure,
        "artifacts/prospective_report.html",
        target,
        result.model_name,
        version="0.3.0.2",
        calibration_start=result.calibration_start,
        evaluation_start=result.evaluation_start,
    )
    console.print(
        f"[green]Global threshold[/green]: {result.global_threshold:.2f}; "
        f"event P/R={float(result.summary.get('event_precision', 0)):.3f}/"
        f"{float(result.summary.get('event_recall', 0)):.3f}; "
        f"macro positive F1={float((result.summary.get('macro_positive_symbols') or {}).get('event_f1') or 0):.3f}"
    )
    console.print(f"[bold green]Prospective report:[/bold green] {report}")


@app.command(name="paper-replay")
def paper_replay_command(
    alerts_path: Annotated[
        str,
        typer.Option("--alerts"),
    ] = "artifacts/paper-alert-events.jsonl",
    prices_path: Annotated[
        str,
        typer.Option("--prices"),
    ] = "artifacts/paper-price-events.jsonl",
    messages_path: Annotated[
        str,
        typer.Option("--messages"),
    ] = "artifacts/paper-messages.jsonl",
    config_path: Annotated[
        str,
        typer.Option("--config"),
    ] = "configs/monitor.yaml",
    project_root: Annotated[
        str,
        typer.Option("--project-root"),
    ] = ".",
) -> None:
    """Replay strict local paper-monitor events without network access."""
    report = run_paper_replay(
        alerts_path,
        prices_path,
        messages_path,
        config_path=config_path,
        project_root=project_root,
    )
    typer.echo(report.to_json())

    if report.status != "passed":
        raise typer.Exit(code=2)


@app.command(name="paper-preflight")
def paper_preflight_command(
    config_path: Annotated[
        str,
        typer.Option("--config"),
    ] = "configs/monitor.yaml",
    project_root: Annotated[
        str,
        typer.Option("--project-root"),
    ] = ".",
) -> None:
    """Validate paper-monitor safety without network side effects."""
    report = run_monitor_preflight(
        config_path,
        project_root=project_root,
    )
    typer.echo(report.to_json())

    if report.status != "passed":
        raise typer.Exit(code=2)


@app.command(name="pytest")
def pytest_command() -> None:
    """Run the bundled automated test suite inside the container."""
    try:
        import pytest
    except ImportError as exc:
        console.print(
            "[bold red]pytest is not installed.[/bold red] Rebuild with Dockerfile v0.2.4."
        )
        raise typer.Exit(code=2) from exc
    raise typer.Exit(code=int(pytest.main(["-q", "tests"])))

@app.command()
def demo(
    days: Annotated[int, typer.Option(min=10, max=120)] = 45,
    seed: Annotated[int, typer.Option()] = 42,
    config_path: Annotated[str, typer.Option("--config")] = "configs/research.yaml",
) -> None:
    """Run the full v0.2.4 pipeline on deterministic synthetic regimes."""
    ensure_directories()
    config = load_config(config_path)
    symbol = "SYNTHUSDT"
    raw = generate_synthetic_market(SyntheticConfig(days=days, seed=seed, symbol=symbol))
    save_frame(raw, _raw_dir(symbol) / "klines.parquet")
    featured = build_features(raw, config.features)
    labelled = add_labels(featured, config.labels)
    save_frame(labelled, _processed_path(symbol))
    events_frame = extract_events(labelled, config.labels)
    save_frame(events_frame, _events_path(symbol))
    samples = extract_warning_samples(
        labelled, events_frame, config.labels, model_feature_columns(labelled)
    )
    save_frame(samples, _samples_path(symbol))
    target = "correction_3_15m"
    split_rows = _split_class_summary(samples, target, config, "event_id")
    train_frame, validation_frame, test_frame = _split_frames(
        samples, config, "event_id"
    )
    requested = model_feature_columns(samples)
    effective, dropped = usable_feature_columns(train_frame, requested)
    availability = calculate_feature_availability(
        {"train": train_frame, "validation": validation_frame, "test": test_frame},
        requested,
    )
    outputs = train_models(
        samples, effective, target, config.training, group_column="event_id"
    )

    metrics: dict[str, dict[str, object]] = {}
    predictions: list[pd.DataFrame] = []
    event_predictions: list[pd.DataFrame] = []
    for output in outputs:
        values = dict(output.metrics)
        values["threshold"] = output.threshold
        metrics[output.name] = values
        current = output.predictions.copy()
        current["model"] = output.name
        predictions.append(current)
        current_events = output.event_predictions.copy()
        current_events["model"] = output.name
        event_predictions.append(current_events)
    save_json(metrics, "artifacts/metrics.json")
    pd.concat(predictions, ignore_index=True).to_csv(
        "artifacts/predictions.csv", index=False
    )
    pd.concat(event_predictions, ignore_index=True).to_csv(
        "artifacts/event_predictions.csv", index=False
    )
    availability.to_csv("artifacts/feature_availability.csv", index=False)
    provenance = {
        "source": "SYNTHETIC",
        "symbols": [symbol],
        "start": samples["timestamp"].min(),
        "end": samples["timestamp"].max(),
        "rows": len(samples),
        "events": len(events_frame),
        "regime": "ALL",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_json(provenance, "artifacts/run_manifest.json")
    report = render_report(
        metrics,
        events_frame,
        target,
        effective,
        "artifacts/report.html",
        version="0.3.0.2",
        provenance=provenance,
        split_summary=split_rows,
        feature_availability=availability,
        dropped_features=dropped,
    )
    console.print(
        f"Synthetic rows: {len(labelled):,}; independent events: {len(events_frame)}; "
        f"warning samples: {len(samples)}"
    )
    console.print(json.dumps(metrics, ensure_ascii=False, indent=2))
    console.print(f"[bold green]Demo completed. Open:[/bold green] {report}")


if __name__ == "__main__":
    app()
