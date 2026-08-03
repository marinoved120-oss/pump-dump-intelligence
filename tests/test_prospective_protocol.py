from __future__ import annotations

import numpy as np
import pandas as pd

from research.config import TrainingConfig
from research.models.prospective import (
    choose_global_event_threshold,
    event_summary,
    prospective_event_partitions,
    run_purged_walk_forward_loso,
)


def _config() -> TrainingConfig:
    return TrainingConfig(
        train_fraction=0.60,
        validation_fraction=0.20,
        random_state=7,
        decision_threshold_grid=21,
        minimum_events_per_split=2,
        bootstrap_repeats=100,
        loso_validation_fraction=0.20,
        prospective_test_fraction=0.25,
        prospective_calibration_fraction=0.25,
        prospective_purge_minutes=60,
        prospective_min_train_events=4,
    )


def _samples(event_count: int = 36) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(3)
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    start = pd.Timestamp("2025-01-01", tz="UTC")
    for index in range(event_count):
        symbol = symbols[index % len(symbols)]
        event_start = start + pd.Timedelta(days=index)
        positive = int(index % 4 == 0 or index % 11 == 0)
        event_id = f"{symbol}-{index:03d}"
        for offset in (0, 5):
            timestamp = event_start + pd.Timedelta(minutes=offset)
            rows.append(
                {
                    "event_id": event_id,
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "minutes_before_peak": 20 - offset,
                    "pump_regime": "FAST" if index % 2 else "MEDIUM",
                    "dump_8_15m": positive if offset == 5 else 0,
                    "feature_signal": positive * 2.5 + rng.normal(0, 0.3),
                    "feature_noise": rng.normal(),
                }
            )
        for minute in range(0, 60, 10):
            market_rows.append(
                {
                    "symbol": symbol,
                    "timestamp": event_start + pd.Timedelta(minutes=minute),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(market_rows)


def test_event_summary_uses_peak_as_event_end() -> None:
    samples, _ = _samples(8)
    summary = event_summary(samples, "dump_8_15m")
    first = summary.iloc[0]
    assert first["event_end"] == first["event_start"] + pd.Timedelta(minutes=20)


def test_prospective_partitions_are_strictly_chronological() -> None:
    samples, _ = _samples(20)
    events = event_summary(samples, "dump_8_15m")
    base, calibration, evaluation = prospective_event_partitions(events, 0.25, 0.25)
    assert base["event_end"].max() < calibration["event_start"].min()
    assert calibration["event_end"].max() < evaluation["event_start"].min()


def test_global_event_threshold_is_single_value() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["a", "a", "b", "b", "c", "c"],
            "symbol": ["A", "A", "B", "B", "C", "C"],
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="min", tz="UTC"),
            "actual": [0, 1, 0, 0, 0, 1],
            "probability": [0.2, 0.8, 0.1, 0.3, 0.4, 0.7],
            "minutes_before_peak": [10, 9, 10, 9, 10, 9],
        }
    )
    threshold, metrics = choose_global_event_threshold(frame, 11)
    assert 0 <= threshold <= 1
    assert metrics["calibration_event_f1"] > 0


def test_purged_walk_forward_loso_uses_past_other_symbols() -> None:
    samples, full_market = _samples()
    result = run_purged_walk_forward_loso(
        samples,
        full_market,
        ["feature_signal", "feature_noise"],
        "dump_8_15m",
        _config(),
        model_name="logistic_regression",
    )
    assert not result.evaluation_predictions.empty
    assert result.evaluation_predictions["global_threshold"].nunique() == 1
    successful = result.fold_log[result.fold_log["status"] == "OK"]
    assert not successful.empty
    for row in successful.itertuples(index=False):
        assert row.holdout_symbol not in str(row.train_symbols).split(",")
        assert pd.Timestamp(row.train_max_event_end) < (
            pd.Timestamp(row.event_start) - pd.Timedelta(minutes=row.purge_minutes)
        )
    assert "macro_positive_symbols" in result.summary
    assert "false_events_per_100_symbol_days" in result.summary
    assert result.summary["symbols_evaluated"] > 0


def test_purged_walk_forward_loso_accepts_compact_market_coverage() -> None:
    samples, full_market = _samples()
    coverage = (
        full_market.groupby("symbol", as_index=False)
        .agg(market_start=("timestamp", "min"), market_end=("timestamp", "max"))
    )
    result = run_purged_walk_forward_loso(
        samples,
        coverage,
        ["feature_signal", "feature_noise"],
        "dump_8_15m",
        _config(),
        model_name="logistic_regression",
    )
    assert not result.evaluation_predictions.empty
    assert not result.exposure.empty
    assert set(result.exposure["symbol"]) == set(coverage["symbol"])
    assert (result.exposure["symbol_days"] >= 0).all()
