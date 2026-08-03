from research.config import load_config
from research.data.synthetic import SyntheticConfig, generate_synthetic_market
from research.features.build import build_features, model_feature_columns


def test_features_are_built() -> None:
    config = load_config()
    raw = generate_synthetic_market(SyntheticConfig(days=5, event_spacing_hours=18))
    featured = build_features(raw, config.features)
    columns = model_feature_columns(featured)
    assert "volume_robust_z" in columns
    assert "taker_imbalance" in columns
    assert featured["data_quality"].dropna().between(0, 1).all()
