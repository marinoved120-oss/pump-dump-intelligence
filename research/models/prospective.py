from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score

from research.config import TrainingConfig
from research.models.baselines import build_estimators, usable_feature_columns
from research.models.evaluation import (
    evaluate_event_predictions,
    evaluate_prediction_frame,
    event_prediction_table,
)


@dataclass(frozen=True)
class ProspectiveResult:
    model_name: str
    global_threshold: float
    calibration_predictions: pd.DataFrame
    evaluation_predictions: pd.DataFrame
    event_predictions: pd.DataFrame
    fold_log: pd.DataFrame
    symbol_metrics: pd.DataFrame
    exposure: pd.DataFrame
    summary: dict[str, Any]
    calibration_start: pd.Timestamp | None
    evaluation_start: pd.Timestamp | None


def _as_utc(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    return result.dropna(subset=["timestamp"])


def event_summary(
    frame: pd.DataFrame,
    target: str,
    group_column: str = "event_id",
) -> pd.DataFrame:
    """Return one chronological row per independent event."""
    if frame.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "symbol",
                "event_start",
                "event_end",
                "actual_event",
                "snapshots",
            ]
        )
    required = {group_column, "symbol", "timestamp", target}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prospective frame is missing columns: {sorted(missing)}")
    clean = _as_utc(frame).dropna(subset=[group_column]).copy()
    clean[target] = pd.to_numeric(clean[target], errors="coerce").fillna(0).astype(int)
    if "minutes_before_peak" in clean:
        minutes = pd.to_numeric(clean["minutes_before_peak"], errors="coerce").fillna(0)
        clean["__event_end"] = clean["timestamp"] + pd.to_timedelta(minutes, unit="m")
    else:
        clean["__event_end"] = clean["timestamp"]
    summary = (
        clean.groupby(group_column, as_index=False)
        .agg(
            symbol=("symbol", "first"),
            event_start=("timestamp", "min"),
            event_end=("__event_end", "max"),
            actual_event=(target, "max"),
            snapshots=(target, "size"),
        )
        .sort_values(["event_start", group_column])
        .reset_index(drop=True)
    )
    return summary


def prospective_event_partitions(
    events: pd.DataFrame,
    test_fraction: float,
    calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically divide events into base development, calibration and evaluation."""
    ordered = events.sort_values("event_start").reset_index(drop=True)
    count = len(ordered)
    if count < 5:
        raise ValueError("At least five independent events are required")
    evaluation_count = max(1, int(round(count * test_fraction)))
    evaluation_count = min(evaluation_count, count - 3)
    development = ordered.iloc[: count - evaluation_count].copy()
    evaluation = ordered.iloc[count - evaluation_count :].copy()
    calibration_count = max(1, int(round(len(development) * calibration_fraction)))
    calibration_count = min(calibration_count, len(development) - 2)
    base = development.iloc[: len(development) - calibration_count].copy()
    calibration = development.iloc[len(development) - calibration_count :].copy()
    return base, calibration, evaluation


def _probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    decision = np.asarray(model.decision_function(features))
    return 1 / (1 + np.exp(-decision))


def _event_prediction_frame(
    event_rows: pd.DataFrame,
    target: str,
    probabilities: np.ndarray,
    model_name: str,
    fold_type: str,
    threshold: float | None,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "timestamp": event_rows["timestamp"].to_numpy(),
            "actual": event_rows[target].to_numpy(dtype=int),
            "probability": probabilities,
            "model": model_name,
            "fold_type": fold_type,
        }
    )
    if threshold is not None:
        result["prediction"] = (probabilities >= threshold).astype(int)
        result["global_threshold"] = threshold
    for column in (
        "sample_id",
        "event_id",
        "symbol",
        "pump_regime",
        "minutes_before_peak",
        "minutes_from_pump_start",
    ):
        if column in event_rows:
            result[column] = event_rows[column].to_numpy()
    return result


def _eligible_training_events(
    events: pd.DataFrame,
    holdout_symbol: str,
    event_start: pd.Timestamp,
    purge_minutes: int,
    group_column: str,
) -> pd.DataFrame:
    cutoff = event_start - timedelta(minutes=purge_minutes)
    return events[
        (events["symbol"].astype(str) != str(holdout_symbol))
        & (events["event_end"] < cutoff)
    ].copy()


def walk_forward_probabilities(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    evaluation_events: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: TrainingConfig,
    model_name: str,
    fold_type: str,
    threshold: float | None = None,
    group_column: str = "event_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a fresh other-symbol model before every event and return honest probabilities."""
    clean = _as_utc(frame).copy()
    estimators = build_estimators(feature_columns, config)
    if model_name not in estimators:
        raise ValueError(
            f"Prospective evaluation supports: {', '.join(sorted(estimators))}"
        )
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    event_ids = set(evaluation_events[group_column].astype(str))
    ordered_events = evaluation_events.sort_values("event_start")

    for event in ordered_events.itertuples(index=False):
        event_id = str(getattr(event, group_column))
        if event_id not in event_ids:
            continue
        holdout_symbol = str(event.symbol)
        event_start = pd.Timestamp(event.event_start)
        eligible = _eligible_training_events(
            events,
            holdout_symbol,
            event_start,
            config.prospective_purge_minutes,
            group_column,
        )
        eligible_ids = set(eligible[group_column].astype(str))
        train = clean[clean[group_column].astype(str).isin(eligible_ids)].copy()
        test = clean[clean[group_column].astype(str) == event_id].copy()
        reason = "OK"
        used: list[str] = []
        dropped: list[str] = []
        if len(eligible_ids) < config.prospective_min_train_events:
            reason = "SKIPPED_MIN_TRAIN_EVENTS"
        elif train[target].nunique() < 2:
            reason = "SKIPPED_ONE_CLASS_TRAIN"
        elif test.empty:
            reason = "SKIPPED_EMPTY_EVENT"
        else:
            used, dropped = usable_feature_columns(train, feature_columns)
            if not used:
                reason = "SKIPPED_NO_FEATURES"

        fold_rows.append(
            {
                "fold_type": fold_type,
                "event_id": event_id,
                "holdout_symbol": holdout_symbol,
                "event_start": event_start,
                "purge_minutes": config.prospective_purge_minutes,
                "train_rows": len(train),
                "train_events": len(eligible_ids),
                "train_positive_events": int(eligible["actual_event"].sum()) if not eligible.empty else 0,
                "train_symbols": ",".join(sorted(train["symbol"].astype(str).unique())) if not train.empty else "",
                "train_max_event_end": eligible["event_end"].max() if not eligible.empty else pd.NaT,
                "features_used": len(used),
                "features_dropped": ",".join(dropped),
                "status": reason,
            }
        )
        if reason != "OK":
            continue

        estimator = build_estimators(used, config)[model_name]
        estimator.fit(train[used], train[target].astype(int))
        probability = _probability(estimator, test[used])
        predictions.append(
            _event_prediction_frame(
                test,
                target,
                probability,
                model_name,
                fold_type,
                threshold,
            )
        )

    prediction_frame = (
        pd.concat(predictions, ignore_index=True).sort_values("timestamp")
        if predictions
        else pd.DataFrame()
    )
    return prediction_frame, pd.DataFrame(fold_rows)


def choose_global_event_threshold(
    calibration_predictions: pd.DataFrame,
    grid_size: int,
) -> tuple[float, dict[str, float]]:
    """Choose one threshold on pooled out-of-symbol, walk-forward calibration events."""
    if calibration_predictions.empty:
        raise ValueError("No calibration predictions were produced")
    if calibration_predictions["actual"].nunique() < 2:
        raise ValueError("Calibration predictions contain only one snapshot target class")
    event_classes = (
        calibration_predictions.groupby("event_id")["actual"].max().astype(int).nunique()
    )
    if event_classes < 2:
        raise ValueError("Calibration predictions contain only one event target class")
    best_threshold = 0.5
    best_f1 = -1.0
    best_precision = -1.0
    for threshold in np.linspace(0.0, 1.0, grid_size):
        candidate = calibration_predictions.copy()
        candidate["prediction"] = (
            candidate["probability"].to_numpy() >= threshold
        ).astype(int)
        events = event_prediction_table(candidate)
        actual = events["actual_event"].to_numpy(dtype=int)
        predicted = events["predicted_event"].to_numpy(dtype=int)
        score = float(f1_score(actual, predicted, zero_division=0))
        precision = float(precision_score(actual, predicted, zero_division=0))
        if score > best_f1 or (np.isclose(score, best_f1) and precision > best_precision):
            best_threshold = float(threshold)
            best_f1 = score
            best_precision = precision
    return best_threshold, {
        "calibration_event_f1": best_f1,
        "calibration_event_precision": best_precision,
    }


def symbol_exposure_days(
    full_market: pd.DataFrame,
    evaluation_start: pd.Timestamp,
    evaluation_end: pd.Timestamp,
    symbols: Iterable[str],
) -> pd.DataFrame:
    """Calculate monitored symbol-days from minute data or compact coverage rows.

    A compact coverage frame contains ``symbol``, ``market_start`` and
    ``market_end``. Supporting it avoids loading millions of full feature rows
    merely to determine exposure boundaries.
    """
    rows: list[dict[str, Any]] = []
    compact = {"symbol", "market_start", "market_end"}.issubset(full_market.columns)
    if compact:
        coverage = full_market.copy()
        coverage["market_start"] = pd.to_datetime(
            coverage["market_start"], utc=True, errors="coerce"
        )
        coverage["market_end"] = pd.to_datetime(
            coverage["market_end"], utc=True, errors="coerce"
        )
        for symbol in sorted(set(map(str, symbols))):
            current = coverage[coverage["symbol"].astype(str) == symbol]
            if current.empty:
                start = pd.NaT
                end = pd.NaT
                days = 0.0
            else:
                market_start = current["market_start"].min()
                market_end = current["market_end"].max()
                start = max(market_start, evaluation_start) if pd.notna(market_start) else pd.NaT
                end = min(market_end, evaluation_end) if pd.notna(market_end) else pd.NaT
                if pd.isna(start) or pd.isna(end) or end < start:
                    days = 0.0
                else:
                    days = max((end - start).total_seconds() / 86400, 1 / 1440)
            rows.append(
                {
                    "symbol": symbol,
                    "exposure_start": start,
                    "exposure_end": end,
                    "symbol_days": days,
                }
            )
        return pd.DataFrame(rows)

    market = _as_utc(full_market)
    for symbol in sorted(set(map(str, symbols))):
        current = market[market["symbol"].astype(str) == symbol]
        current = current[
            (current["timestamp"] >= evaluation_start)
            & (current["timestamp"] <= evaluation_end)
        ]
        if current.empty:
            days = 0.0
            start = pd.NaT
            end = pd.NaT
        else:
            start = current["timestamp"].min()
            end = current["timestamp"].max()
            days = max((end - start).total_seconds() / 86400, 1 / 1440)
        rows.append(
            {
                "symbol": symbol,
                "exposure_start": start,
                "exposure_end": end,
                "symbol_days": days,
            }
        )
    return pd.DataFrame(rows)


def per_symbol_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame()
    for symbol, frame in predictions.groupby("symbol", sort=True):
        snapshot = evaluate_prediction_frame(frame).as_dict()
        event_metrics, event_table = evaluate_event_predictions(
            frame, bootstrap_repeats=0
        )
        true_positive = event_table[
            (event_table["actual_event"] == 1)
            & (event_table["predicted_event"] == 1)
        ]
        leads = pd.to_numeric(
            true_positive["lead_to_first_positive_minutes"], errors="coerce"
        ).dropna()
        rows.append(
            {
                "symbol": str(symbol),
                "rows": len(frame),
                "positive_snapshots": int(frame["actual"].sum()),
                "events": len(event_table),
                "positive_events": int(event_metrics["actual_positive_events"]),
                "detected_events": int(event_metrics["true_positive_events"]),
                "false_positive_events": int(event_metrics["false_positive_events"]),
                "average_precision": snapshot["average_precision"],
                "snapshot_precision": snapshot["precision"],
                "snapshot_recall": snapshot["recall"],
                "event_precision": event_metrics["event_precision"],
                "event_recall": event_metrics["event_recall"],
                "event_f1": event_metrics["event_f1"],
                "median_lead_minutes": float(leads.median()) if not leads.empty else None,
            }
        )
    return pd.DataFrame(rows)


def _mean_numeric(frame: pd.DataFrame, columns: list[str]) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        values[column] = float(series.mean()) if not series.empty else None
    return values


def prospective_summary(
    predictions: pd.DataFrame,
    symbol_metrics: pd.DataFrame,
    exposure: pd.DataFrame,
    threshold: float,
    calibration_metrics: dict[str, float],
    config: TrainingConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    snapshot = evaluate_prediction_frame(predictions).as_dict()
    event_metrics, event_predictions = evaluate_event_predictions(
        predictions,
        bootstrap_repeats=config.bootstrap_repeats,
        random_state=config.random_state,
    )
    positive_symbols = symbol_metrics[symbol_metrics["positive_events"] > 0]
    all_macro = _mean_numeric(
        symbol_metrics,
        ["average_precision", "event_precision", "event_recall", "event_f1"],
    )
    positive_macro = _mean_numeric(
        positive_symbols,
        ["average_precision", "event_precision", "event_recall", "event_f1"],
    )
    detected_symbols = positive_symbols[positive_symbols["detected_events"] > 0]
    true_positive = event_predictions[
        (event_predictions["actual_event"] == 1)
        & (event_predictions["predicted_event"] == 1)
    ]
    leads = pd.to_numeric(
        true_positive["lead_to_first_positive_minutes"], errors="coerce"
    ).dropna()
    total_symbol_days = float(pd.to_numeric(exposure["symbol_days"], errors="coerce").sum())
    false_events = int(event_metrics["false_positive_events"])
    false_alerts = int(
        ((predictions["prediction"] == 1) & (predictions["actual"] == 0)).sum()
    )
    summary: dict[str, Any] = dict(snapshot)
    summary.update(event_metrics)
    summary.update(calibration_metrics)
    summary.update(
        {
            "global_threshold": threshold,
            "purge_minutes": config.prospective_purge_minutes,
            "symbols_evaluated": int(len(symbol_metrics)),
            "symbols_with_positive_events": int(len(positive_symbols)),
            "symbols_with_detected_positive_event": int(len(detected_symbols)),
            "symbol_detection_coverage": (
                float(len(detected_symbols) / len(positive_symbols))
                if len(positive_symbols)
                else 0.0
            ),
            "macro_all_symbols": all_macro,
            "macro_positive_symbols": positive_macro,
            "total_symbol_days": total_symbol_days,
            "false_events_per_100_symbol_days": (
                false_events / total_symbol_days * 100 if total_symbol_days else None
            ),
            "false_alerts_per_100_symbol_days": (
                false_alerts / total_symbol_days * 100 if total_symbol_days else None
            ),
            "lead_minutes_p25": float(leads.quantile(0.25)) if not leads.empty else None,
            "lead_minutes_median": float(leads.median()) if not leads.empty else None,
            "lead_minutes_p75": float(leads.quantile(0.75)) if not leads.empty else None,
            "lead_share_ge_5m": float((leads >= 5).mean()) if not leads.empty else None,
            "lead_share_ge_10m": float((leads >= 10).mean()) if not leads.empty else None,
        }
    )
    return summary, event_predictions


def run_purged_walk_forward_loso(
    frame: pd.DataFrame,
    full_market: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: TrainingConfig,
    model_name: str = "random_forest",
    group_column: str = "event_id",
) -> ProspectiveResult:
    """Nested temporal LOSO with one calibration threshold and strict past-only training."""
    clean = _as_utc(frame).dropna(subset=[group_column, target]).copy()
    clean[target] = pd.to_numeric(clean[target], errors="coerce").fillna(0).astype(int)
    events = event_summary(clean, target, group_column)
    _, calibration_events, evaluation_events = prospective_event_partitions(
        events,
        config.prospective_test_fraction,
        config.prospective_calibration_fraction,
    )
    calibration_predictions, calibration_folds = walk_forward_probabilities(
        clean,
        events,
        calibration_events,
        feature_columns,
        target,
        config,
        model_name,
        "CALIBRATION",
        threshold=None,
        group_column=group_column,
    )
    threshold, calibration_metrics = choose_global_event_threshold(
        calibration_predictions,
        config.decision_threshold_grid,
    )
    evaluation_predictions, evaluation_folds = walk_forward_probabilities(
        clean,
        events,
        evaluation_events,
        feature_columns,
        target,
        config,
        model_name,
        "EVALUATION",
        threshold=threshold,
        group_column=group_column,
    )
    if evaluation_predictions.empty:
        raise ValueError("No prospective evaluation predictions were produced")
    symbol_metrics = per_symbol_metrics(evaluation_predictions)
    evaluation_start = pd.Timestamp(evaluation_events["event_start"].min())
    if {"market_start", "market_end"}.issubset(full_market.columns):
        market_end = pd.to_datetime(full_market["market_end"], utc=True, errors="coerce")
        evaluation_end = pd.Timestamp(market_end.max())
    else:
        market_timestamps = pd.to_datetime(
            full_market["timestamp"], utc=True, errors="coerce"
        )
        evaluation_end = pd.Timestamp(market_timestamps.max())
    exposure = symbol_exposure_days(
        full_market,
        evaluation_start,
        evaluation_end,
        full_market["symbol"].astype(str).unique(),
    )
    summary, event_predictions = prospective_summary(
        evaluation_predictions,
        symbol_metrics,
        exposure,
        threshold,
        calibration_metrics,
        config,
    )
    fold_log = pd.concat(
        [calibration_folds, evaluation_folds], ignore_index=True
    ).sort_values(["event_start", "fold_type"])
    return ProspectiveResult(
        model_name=model_name,
        global_threshold=threshold,
        calibration_predictions=calibration_predictions,
        evaluation_predictions=evaluation_predictions,
        event_predictions=event_predictions,
        fold_log=fold_log,
        symbol_metrics=symbol_metrics,
        exposure=exposure,
        summary=summary,
        calibration_start=(
            pd.Timestamp(calibration_events["event_start"].min())
            if not calibration_events.empty
            else None
        ),
        evaluation_start=evaluation_start,
    )
