import pandas as pd

from research.config import load_config
from research.labels.events import extract_events, extract_warning_samples


def _event_frame(peak_gain: float = 0.10) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=400, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 1.0,
            "high": 1.0,
            "low": 0.999,
            "close": 1.0,
            "symbol": "TESTUSDT",
            "event_id": pd.Series(pd.NA, index=range(400), dtype="string"),
            "pump_trigger": 0,
            "pump_trigger_mode": "",
            "realized_vol_60m": 0.0,
            "return_1m": 0.0,
        }
    )
    frame.loc[10, ["event_id", "pump_trigger", "pump_trigger_mode"]] = ["RAW-A", 1, "15m"]
    frame.loc[80, ["event_id", "pump_trigger", "pump_trigger_mode"]] = ["RAW-B", 1, "60m"]
    frame.loc[100, "high"] = 1.0 + peak_gain
    frame.loc[101:115, "low"] = 0.92
    return frame


def test_overlapping_raw_events_with_same_peak_are_merged() -> None:
    config = load_config()
    events = extract_events(_event_frame(), config.labels)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["merged_event_count"] == 2
    assert event["source_event_ids"] == "RAW-A,RAW-B"
    assert event["pump_regime"] == "MEDIUM"
    assert event["pump_return_to_peak"] >= 0.09


def test_insignificant_event_level_rise_is_filtered() -> None:
    config = load_config()
    events = extract_events(_event_frame(peak_gain=0.01), config.labels)
    assert events.empty


def test_warning_target_is_measured_from_snapshot_close() -> None:
    config = load_config()
    frame = _event_frame()
    for horizon in config.labels.forward_horizons:
        future_low = frame["low"].shift(-1)[::-1].rolling(horizon, min_periods=1).min()[::-1]
        frame[f"forward_drawdown_{horizon}m"] = future_low / frame["close"] - 1
    events = extract_events(frame, config.labels)
    samples = extract_warning_samples(frame, events, config.labels, ["return_1m"])
    first = samples.iloc[0]
    index = int(frame.index[frame["timestamp"] == first["timestamp"]][0])
    expected = frame.iloc[index + 1 : index + 16]["low"].min() / frame.iloc[index]["close"] - 1
    assert first["label_reference"] == "snapshot_close"
    assert abs(first["forward_drawdown_15m"] - expected) < 1e-12
