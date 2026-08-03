from research.config import load_config
from research.data.synthetic import SyntheticConfig, generate_synthetic_market
from research.features.build import build_features, model_feature_columns
from research.labels.events import add_labels, extract_events, extract_warning_samples


def test_synthetic_events_are_clustered_and_continuous_targets_exist() -> None:
    config = load_config()
    raw = generate_synthetic_market(SyntheticConfig(days=12, event_spacing_hours=18))
    featured = build_features(raw, config.features)
    labelled = add_labels(featured, config.labels)
    events = extract_events(labelled, config.labels)
    samples = extract_warning_samples(
        labelled, events, config.labels, model_feature_columns(labelled)
    )
    assert labelled["pump_context"].sum() > 0
    assert len(events) <= labelled["event_id"].nunique()
    assert "forward_drawdown_15m" in labelled
    assert "drawdown_class_15m" in labelled
    assert len(events) > 3
    assert not samples.empty
    assert samples["event_id"].nunique() == len(events)



def test_extract_events_handles_no_pumps_without_crashing() -> None:
    import pandas as pd

    config = load_config()
    timestamps = pd.date_range("2026-01-01", periods=500, freq="min", tz="UTC")
    raw = pd.DataFrame({
        "timestamp": timestamps,
        "open": 1.0,
        "high": 1.001,
        "low": 0.999,
        "close": 1.0,
        "volume": 10.0,
        "volume_quote": 10.0,
        "trade_count": 10,
        "taker_buy_base": 5.0,
        "taker_buy_quote": 5.0,
        "symbol": "QUIETUSDT",
    })
    featured = build_features(raw, config.features)
    labelled = add_labels(featured, config.labels)
    events = extract_events(labelled, config.labels)
    samples = extract_warning_samples(
        labelled, events, config.labels, model_feature_columns(labelled)
    )
    assert events.empty
    assert "pump_start" in events.columns
    assert samples.empty
    assert "timestamp" in samples.columns
