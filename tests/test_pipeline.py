from research.config import load_config
from research.data.synthetic import SyntheticConfig, generate_synthetic_market
from research.features.build import build_features, model_feature_columns
from research.labels.events import add_labels, extract_events, extract_warning_samples
from research.models.baselines import chronological_group_split, train_models


def test_group_split_does_not_leak_events() -> None:
    config = load_config()
    raw = generate_synthetic_market(SyntheticConfig(days=45, event_spacing_hours=20))
    featured = build_features(raw, config.features)
    labelled = add_labels(featured, config.labels)
    events = extract_events(labelled, config.labels)
    samples = extract_warning_samples(
        labelled, events, config.labels, model_feature_columns(labelled)
    )
    train, validation, test = chronological_group_split(
        samples,
        config.training.train_fraction,
        config.training.validation_fraction,
        "event_id",
    )
    assert set(train["event_id"]).isdisjoint(set(validation["event_id"]))
    assert set(train["event_id"]).isdisjoint(set(test["event_id"]))
    assert set(validation["event_id"]).isdisjoint(set(test["event_id"]))


def test_end_to_end_event_snapshot_training() -> None:
    config = load_config()
    raw = generate_synthetic_market(SyntheticConfig(days=60, event_spacing_hours=18))
    featured = build_features(raw, config.features)
    labelled = add_labels(featured, config.labels)
    events = extract_events(labelled, config.labels)
    samples = extract_warning_samples(
        labelled, events, config.labels, model_feature_columns(labelled)
    )
    outputs = train_models(
        samples,
        model_feature_columns(samples),
        "correction_3_15m",
        config.training,
        group_column="event_id",
    )
    assert len(outputs) >= 3
    assert all(0 <= output.metrics["average_precision"] <= 1 for output in outputs)


def test_small_group_split_still_does_not_leak() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="min", tz="UTC"),
            "event_id": ["A", "A", "A", "B", "B", "B"],
            "target": [0, 1, 0, 0, 1, 0],
        }
    )
    train, validation, test = chronological_group_split(frame, 0.6, 0.2, "event_id")
    assert set(train["event_id"]).isdisjoint(set(validation["event_id"]))
    assert test.empty
