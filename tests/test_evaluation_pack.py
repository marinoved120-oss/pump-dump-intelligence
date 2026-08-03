from __future__ import annotations

import pandas as pd

from research.config import load_config
from research.data.synthetic import SyntheticConfig, generate_synthetic_market
from research.features.build import build_features, model_feature_columns
from research.labels.events import add_labels, extract_events, extract_warning_samples
from research.models.baselines import (
    train_models_external_holdout,
    usable_feature_columns,
)
from research.models.evaluation import (
    evaluate_event_predictions,
    feature_availability,
)


def test_event_metrics_collapse_repeated_alerts() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["A", "A", "B", "B", "C"],
            "symbol": ["AAA", "AAA", "BBB", "BBB", "CCC"],
            "pump_regime": ["FAST"] * 5,
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC"),
            "actual": [0, 1, 0, 0, 1],
            "prediction": [1, 1, 1, 0, 0],
            "probability": [0.8, 0.9, 0.7, 0.1, 0.2],
            "minutes_before_peak": [20, 15, 10, 5, 2],
        }
    )
    metrics, events = evaluate_event_predictions(frame, bootstrap_repeats=100, random_state=7)
    assert len(events) == 3
    assert metrics["true_positive_events"] == 1
    assert metrics["false_positive_events"] == 1
    assert metrics["event_precision"] == 0.5
    assert metrics["event_recall"] == 0.5
    assert metrics["alerts_per_predicted_event"] == 1.5
    assert metrics["event_precision_ci_low"] <= metrics["event_precision"] <= metrics["event_precision_ci_high"]


def test_all_null_feature_is_dropped_and_reported() -> None:
    train = pd.DataFrame({"ok": [1.0, 2.0], "empty": [None, None]})
    validation = pd.DataFrame({"ok": [3.0], "empty": [1.0]})
    test = pd.DataFrame({"ok": [4.0], "empty": [2.0]})
    used, dropped = usable_feature_columns(train, ["ok", "empty"])
    availability = feature_availability(
        {"train": train, "validation": validation, "test": test},
        ["ok", "empty"],
    )
    assert used == ["ok"]
    assert dropped == ["empty"]
    empty = availability.loc[availability["feature"] == "empty"].iloc[0]
    assert empty["train_available"] == 0
    assert empty["test_available"] == 1


def test_external_symbol_holdout_runs_without_symbol_leakage() -> None:
    config = load_config()
    raw_a = generate_synthetic_market(
        SyntheticConfig(days=45, seed=1, symbol="AAAUSDT", event_spacing_hours=18)
    )
    raw_b = generate_synthetic_market(
        SyntheticConfig(days=45, seed=2, symbol="BBBUSDT", event_spacing_hours=18)
    )

    def samples(raw: pd.DataFrame) -> pd.DataFrame:
        featured = build_features(raw, config.features)
        labelled = add_labels(featured, config.labels)
        events = extract_events(labelled, config.labels)
        return extract_warning_samples(
            labelled, events, config.labels, model_feature_columns(labelled)
        )

    training = samples(raw_a)
    holdout = samples(raw_b)
    outputs, used, _ = train_models_external_holdout(
        training,
        holdout,
        model_feature_columns(pd.concat([training, holdout], ignore_index=True)),
        "correction_3_15m",
        config.training,
        model_names=["logistic_regression"],
    )
    assert used
    assert len(outputs) == 1
    assert set(outputs[0].predictions["symbol"]) == {"BBBUSDT"}
