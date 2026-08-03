from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from research.config import LabelConfig


def forward_min(series: pd.Series, periods: int) -> pd.Series:
    return series.shift(-1)[::-1].rolling(periods, min_periods=1).min()[::-1]


def forward_max(series: pd.Series, periods: int) -> pd.Series:
    return series.shift(-1)[::-1].rolling(periods, min_periods=1).max()[::-1]


def drawdown_class(
    drawdown: pd.Series | np.ndarray | float,
    thresholds: dict[int, float],
) -> pd.Series | np.ndarray | int:
    """Convert a negative return to an ordinal severity class from 0 to 4."""
    bins = sorted((float(threshold), int(level)) for level, threshold in thresholds.items())

    def classify(value: float) -> int:
        if not np.isfinite(value):
            return 0
        magnitude = max(0.0, -float(value))
        result = 0
        for threshold, level in bins:
            if magnitude >= threshold:
                result = level
        return result

    if isinstance(drawdown, pd.Series):
        return drawdown.map(classify).astype("int8")
    if isinstance(drawdown, np.ndarray):
        return np.asarray([classify(float(value)) for value in drawdown], dtype=np.int8)
    return classify(float(drawdown))


def _realized_volatility_column(data: pd.DataFrame, horizon: int) -> pd.Series:
    exact = f"realized_vol_{horizon}m"
    if exact in data:
        return pd.to_numeric(data[exact], errors="coerce")
    available = [
        int(column.removeprefix("realized_vol_").removesuffix("m"))
        for column in data.columns
        if column.startswith("realized_vol_") and column.endswith("m")
    ]
    if not available:
        return pd.Series(0.0, index=data.index)
    nearest = min(available, key=lambda value: abs(value - horizon))
    base = pd.to_numeric(data[f"realized_vol_{nearest}m"], errors="coerce")
    return base * np.sqrt(horizon / nearest)


def _trigger_modes(data: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    result = pd.DataFrame(index=data.index)
    quality = pd.to_numeric(data.get("data_quality", 0.0), errors="coerce").fillna(0.0)
    volume_z = pd.to_numeric(data.get("volume_robust_z", np.nan), errors="coerce")
    trade_z = pd.to_numeric(data.get("trade_count_robust_z", np.nan), errors="coerce")
    activity = (volume_z >= config.pump_volume_z) & (trade_z >= config.pump_trade_z)

    for horizon, fixed_threshold in sorted(config.pump_horizons.items()):
        return_column = f"return_{horizon}m"
        if return_column not in data:
            result[f"pump_trigger_{horizon}m"] = 0
            continue
        volatility = _realized_volatility_column(data, horizon).fillna(0.0)
        dynamic_threshold = np.maximum(
            fixed_threshold,
            config.pump_volatility_multipliers[horizon] * volatility,
        )
        result[f"pump_threshold_{horizon}m"] = dynamic_threshold
        result[f"pump_trigger_{horizon}m"] = (
            (pd.to_numeric(data[return_column], errors="coerce") >= dynamic_threshold)
            & activity
            & (quality >= config.min_data_quality)
        ).astype("int8")
    return result


def _assign_event_context(
    data: pd.DataFrame,
    trigger: pd.Series,
    config: LabelConfig,
) -> tuple[pd.Series, pd.Series]:
    event_ids = pd.Series(pd.NA, index=data.index, dtype="string")
    context = pd.Series(0, index=data.index, dtype="int8")
    timestamps = pd.to_datetime(data["timestamp"], utc=True)
    symbol = str(data.get("symbol", pd.Series(["UNKNOWN"])).iloc[0])

    current_event: str | None = None
    active_until = -1
    last_trigger = -10**9
    event_number = 0

    for index in range(len(data)):
        is_trigger = bool(trigger.iloc[index])
        if is_trigger:
            starts_new = (
                current_event is None
                or index > active_until
                or index - last_trigger > config.pump_cluster_gap_minutes
            )
            if starts_new:
                event_number += 1
                stamp = timestamps.iloc[index].strftime("%Y%m%dT%H%M%S")
                current_event = f"{symbol}-{stamp}-RAW{event_number:04d}"
            last_trigger = index
            active_until = max(active_until, index + config.pump_context_minutes)

        if current_event is not None and index <= active_until:
            context.iloc[index] = 1
            event_ids.iloc[index] = current_event
        elif index > active_until:
            current_event = None

    return event_ids, context


def add_labels(frame: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Add multi-horizon pump triggers and continuous future drawdown targets.

    All trigger and feature inputs use only current or past information. Future
    lows/highs are used exclusively for research labels.
    """
    data = frame.sort_values("timestamp").reset_index(drop=True).copy()
    modes = _trigger_modes(data, config)
    for column in modes:
        data[column] = modes[column]

    trigger_columns = [
        column for column in data.columns if column.startswith("pump_trigger_")
    ]
    data["pump_trigger"] = data[trigger_columns].max(axis=1).astype("int8")
    data["pump_trigger_mode"] = [
        ",".join(
            column.removeprefix("pump_trigger_")
            for column in trigger_columns
            if int(data.at[index, column]) == 1
        )
        for index in data.index
    ]
    event_ids, context = _assign_event_context(data, data["pump_trigger"], config)
    data["event_id"] = event_ids
    data["pump_context"] = context

    for horizon in config.forward_horizons:
        future_low = forward_min(pd.to_numeric(data["low"], errors="coerce"), horizon)
        future_high = forward_max(pd.to_numeric(data["high"], errors="coerce"), horizon)
        close = pd.to_numeric(data["close"], errors="coerce")
        future_drawdown = future_low / close - 1
        future_runup = future_high / close - 1
        severity = drawdown_class(future_drawdown, config.severity_thresholds)

        data[f"forward_drawdown_{horizon}m"] = future_drawdown
        data[f"forward_runup_{horizon}m"] = future_runup
        data[f"drawdown_class_{horizon}m"] = severity
        data[f"pump_drawdown_class_{horizon}m"] = (
            pd.Series(severity, index=data.index) * data["pump_context"]
        ).astype("int8")

        for level, threshold in sorted(config.severity_thresholds.items()):
            prefix = {1: "correction_3", 2: "strong_5", 3: "dump_8", 4: "extreme_12"}.get(
                level, f"drawdown_{int(threshold * 100)}"
            )
            data[f"{prefix}_{horizon}m"] = (
                (data["pump_context"] == 1) & (future_drawdown <= -threshold)
            ).astype("int8")

    # Compatibility targets used by v0.1 commands.
    if "strong_5_5m" in data:
        data["dump_5m"] = data["strong_5_5m"]
    if "dump_8_15m" in data:
        data["dump_15m"] = data["dump_8_15m"]
    if "dump_8_30m" in data:
        data["dump_30m"] = data["dump_8_30m"]
    return data


def _minutes_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 60.0


def _trough_details(
    data: pd.DataFrame,
    peak_index: int,
    horizon: int,
) -> tuple[pd.Timestamp | None, float | None, float | None, float | None]:
    segment = data.iloc[peak_index + 1 : peak_index + 1 + horizon]
    if segment.empty:
        return None, None, None, None
    trough_index = int(segment["low"].idxmin())
    trough_time = pd.Timestamp(data.loc[trough_index, "timestamp"])
    trough_price = float(data.loc[trough_index, "low"])
    peak_price = float(data.loc[peak_index, "high"])
    drawdown = trough_price / peak_price - 1
    minutes = _minutes_between(pd.Timestamp(data.loc[peak_index, "timestamp"]), trough_time)
    return trough_time, trough_price, drawdown, minutes


def _raw_event_candidates(data: pd.DataFrame, config: LabelConfig) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    if "event_id" not in data:
        return candidates

    for raw_event_id, context in data.dropna(subset=["event_id"]).groupby("event_id", sort=False):
        context = context.sort_values("timestamp")
        trigger_rows = context[context["pump_trigger"] == 1]
        if trigger_rows.empty:
            continue
        start_index = int(trigger_rows.index.min())
        last_trigger_index = int(trigger_rows.index.max())
        peak_end = min(len(data), start_index + config.peak_lookahead_minutes + 1)
        peak_segment = data.iloc[start_index:peak_end]
        if peak_segment.empty:
            continue
        peak_index = int(peak_segment["high"].idxmax())
        modes = {
            mode
            for value in trigger_rows["pump_trigger_mode"].astype(str)
            for mode in value.split(",")
            if mode
        }
        candidates.append(
            {
                "start_index": start_index,
                "last_trigger_index": last_trigger_index,
                "peak_index": peak_index,
                "trigger_indices": set(map(int, trigger_rows.index)),
                "trigger_modes": modes,
                "source_event_ids": {str(raw_event_id)},
            }
        )
    return sorted(candidates, key=lambda item: int(item["start_index"]))


def _recompute_cluster_peak(data: pd.DataFrame, cluster: dict[str, object]) -> None:
    start_index = int(cluster["start_index"])
    candidate_peak_end = int(cluster["candidate_peak_end"])
    segment = data.iloc[start_index : candidate_peak_end + 1]
    cluster["peak_index"] = int(segment["high"].idxmax())


def _merge_overlapping_candidates(
    data: pd.DataFrame,
    candidates: list[dict[str, object]],
    config: LabelConfig,
) -> list[dict[str, object]]:
    """Merge candidate events whose pump-to-peak intervals overlap.

    This prevents one multi-hour move from being counted multiple times merely
    because trigger minutes were separated by the raw clustering gap.
    """
    merged: list[dict[str, object]] = []
    for candidate in candidates:
        current = {
            **candidate,
            "candidate_peak_end": int(candidate["peak_index"]),
        }
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        overlaps = int(current["start_index"]) <= (
            int(previous["peak_index"]) + config.event_merge_gap_minutes
        )
        same_peak = int(current["peak_index"]) == int(previous["peak_index"])
        if not (overlaps or same_peak):
            merged.append(current)
            continue

        previous["start_index"] = min(
            int(previous["start_index"]), int(current["start_index"])
        )
        previous["last_trigger_index"] = max(
            int(previous["last_trigger_index"]), int(current["last_trigger_index"])
        )
        previous["candidate_peak_end"] = max(
            int(previous["candidate_peak_end"]), int(current["candidate_peak_end"])
        )
        previous["trigger_indices"] = set(previous["trigger_indices"]) | set(
            current["trigger_indices"]
        )
        previous["trigger_modes"] = set(previous["trigger_modes"]) | set(
            current["trigger_modes"]
        )
        previous["source_event_ids"] = set(previous["source_event_ids"]) | set(
            current["source_event_ids"]
        )
        _recompute_cluster_peak(data, previous)
    return merged


def _pump_regime(minutes_to_peak: float, config: LabelConfig) -> str:
    if minutes_to_peak <= config.fast_pump_max_minutes:
        return "FAST"
    if minutes_to_peak <= config.medium_pump_max_minutes:
        return "MEDIUM"
    return "SLOW"


def _event_columns(config: LabelConfig) -> list[str]:
    columns = [
        "event_id",
        "symbol",
        "pump_start",
        "last_trigger_time",
        "peak_time",
        "trigger_modes",
        "trigger_minutes",
        "source_event_ids",
        "merged_event_count",
        "pump_start_close",
        "peak_price",
        "pump_return_to_peak",
        "event_volatility_at_start",
        "event_pump_threshold",
        "minutes_to_peak",
        "pump_regime",
    ]
    for horizon in config.forward_horizons:
        columns.extend(
            [
                f"trough_time_{horizon}m",
                f"trough_price_{horizon}m",
                f"peak_drawdown_{horizon}m",
                f"minutes_to_trough_{horizon}m",
                f"event_drawdown_class_{horizon}m",
            ]
        )
    columns.append("max_event_drawdown_class")
    return columns


def extract_events(frame: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Create one row per cleaned, independent pump event.

    Raw trigger clusters are merged when their pump-to-peak intervals overlap.
    Events that never achieve a meaningful event-level rise are excluded.
    """
    data = frame.sort_values("timestamp").reset_index(drop=True).copy()
    candidates = _raw_event_candidates(data, config)
    clusters = _merge_overlapping_candidates(data, candidates, config)

    events: list[dict[str, object]] = []
    symbol_default = str(data.get("symbol", pd.Series(["UNKNOWN"])).iloc[0])
    volatility = _realized_volatility_column(data, 60).fillna(0.0)

    for cluster in clusters:
        start_index = int(cluster["start_index"])
        last_trigger_index = int(cluster["last_trigger_index"])
        peak_index = int(cluster["peak_index"])
        start_time = pd.Timestamp(data.loc[start_index, "timestamp"])
        peak_time = pd.Timestamp(data.loc[peak_index, "timestamp"])
        start_close = float(data.loc[start_index, "close"])
        peak_price = float(data.loc[peak_index, "high"])
        pump_return = peak_price / start_close - 1
        event_volatility = float(volatility.iloc[start_index]) if np.isfinite(volatility.iloc[start_index]) else 0.0
        event_threshold = max(
            config.event_min_pump_return,
            config.event_min_pump_volatility_multiplier * event_volatility,
        )
        if not np.isfinite(pump_return) or pump_return < event_threshold:
            continue

        minutes_to_peak = _minutes_between(start_time, peak_time)
        event_number = len(events) + 1
        event_id = f"{symbol_default}-{start_time.strftime('%Y%m%dT%H%M%S')}-{event_number:04d}"
        row: dict[str, object] = {
            "event_id": event_id,
            "symbol": data.loc[start_index].get("symbol", symbol_default),
            "pump_start": start_time,
            "last_trigger_time": pd.Timestamp(data.loc[last_trigger_index, "timestamp"]),
            "peak_time": peak_time,
            "trigger_modes": ",".join(sorted(set(cluster["trigger_modes"]))),
            "trigger_minutes": int(len(set(cluster["trigger_indices"]))),
            "source_event_ids": ",".join(sorted(set(cluster["source_event_ids"]))),
            "merged_event_count": int(len(set(cluster["source_event_ids"]))),
            "pump_start_close": start_close,
            "peak_price": peak_price,
            "pump_return_to_peak": pump_return,
            "event_volatility_at_start": event_volatility,
            "event_pump_threshold": event_threshold,
            "minutes_to_peak": minutes_to_peak,
            "pump_regime": _pump_regime(minutes_to_peak, config),
        }

        max_class = 0
        for horizon in config.forward_horizons:
            trough_time, trough_price, drawdown, minutes = _trough_details(
                data, peak_index, horizon
            )
            row[f"trough_time_{horizon}m"] = trough_time
            row[f"trough_price_{horizon}m"] = trough_price
            row[f"peak_drawdown_{horizon}m"] = drawdown
            row[f"minutes_to_trough_{horizon}m"] = minutes
            severity = drawdown_class(
                drawdown if drawdown is not None else np.nan,
                config.severity_thresholds,
            )
            row[f"event_drawdown_class_{horizon}m"] = severity
            max_class = max(max_class, int(severity))
        row["max_event_drawdown_class"] = max_class
        events.append(row)

    if not events:
        return pd.DataFrame(columns=_event_columns(config))
    return pd.DataFrame(events, columns=_event_columns(config)).sort_values("pump_start").reset_index(drop=True)


def extract_warning_samples(
    frame: pd.DataFrame,
    events: pd.DataFrame,
    config: LabelConfig,
    feature_columns: Iterable[str],
) -> pd.DataFrame:
    """Create leakage-safe feature snapshots after pump confirmation.

    Each target is measured from the close at the snapshot timestamp, never
    from the event's future peak. Event-level peak fields are metadata only and
    are not part of ``model_feature_columns``.
    """
    metadata_columns = [
        "sample_id",
        "event_id",
        "symbol",
        "timestamp",
        "snapshot_close",
        "label_reference",
        "minutes_from_pump_start",
        "minutes_before_peak",
        "pump_regime",
        "merged_event_count",
    ]
    if events.empty:
        columns = [*metadata_columns, *list(feature_columns)]
        for horizon in config.forward_horizons:
            columns.extend(
                [
                    f"forward_drawdown_{horizon}m",
                    f"drawdown_class_{horizon}m",
                    f"correction_3_{horizon}m",
                    f"strong_5_{horizon}m",
                    f"dump_8_{horizon}m",
                    f"extreme_12_{horizon}m",
                ]
            )
        return pd.DataFrame(columns=list(dict.fromkeys(columns)))

    data = frame.sort_values("timestamp").reset_index(drop=True).copy()
    timestamp_to_index = {
        pd.Timestamp(timestamp): index for index, timestamp in enumerate(data["timestamp"])
    }
    rows: list[dict[str, object]] = []

    for event in events.itertuples(index=False):
        start_time = pd.Timestamp(event.pump_start)
        peak_time = pd.Timestamp(event.peak_time)
        for offset in config.warning_offsets_minutes:
            snapshot_time = start_time + pd.Timedelta(minutes=offset)
            if snapshot_time > peak_time or snapshot_time not in timestamp_to_index:
                continue
            index = timestamp_to_index[snapshot_time]
            source = data.iloc[index]
            row: dict[str, object] = {
                "sample_id": f"{event.event_id}-t{offset:03d}",
                "event_id": event.event_id,
                "symbol": event.symbol,
                "timestamp": snapshot_time,
                "snapshot_close": source.get("close", np.nan),
                "label_reference": "snapshot_close",
                "minutes_from_pump_start": offset,
                "minutes_before_peak": _minutes_between(snapshot_time, peak_time),
                "pump_regime": event.pump_regime,
                "merged_event_count": event.merged_event_count,
            }
            for feature in feature_columns:
                if feature in source:
                    row[feature] = source[feature]
            for horizon in config.forward_horizons:
                drawdown = source.get(f"forward_drawdown_{horizon}m", np.nan)
                severity = drawdown_class(drawdown, config.severity_thresholds)
                row[f"forward_drawdown_{horizon}m"] = drawdown
                row[f"drawdown_class_{horizon}m"] = severity
                row[f"correction_3_{horizon}m"] = int(severity >= 1)
                row[f"strong_5_{horizon}m"] = int(severity >= 2)
                row[f"dump_8_{horizon}m"] = int(severity >= 3)
                row[f"extreme_12_{horizon}m"] = int(severity >= 4)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=metadata_columns)
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
