from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BinanceConfig:
    base_url: str
    timeout_seconds: float
    max_retries: int
    backoff_seconds: float


@dataclass(frozen=True)
class FeatureConfig:
    ewma_days: int
    robust_window_minutes: int
    high_window_minutes: int
    min_history_minutes: int


@dataclass(frozen=True)
class LabelConfig:
    pump_horizons: dict[int, float]
    pump_volatility_multipliers: dict[int, float]
    pump_volume_z: float
    pump_trade_z: float
    min_data_quality: float
    pump_cluster_gap_minutes: int
    pump_context_minutes: int
    peak_lookahead_minutes: int
    event_merge_gap_minutes: int
    event_min_pump_return: float
    event_min_pump_volatility_multiplier: float
    fast_pump_max_minutes: int
    medium_pump_max_minutes: int
    forward_horizons: tuple[int, ...]
    severity_thresholds: dict[int, float]
    warning_offsets_minutes: tuple[int, ...]


@dataclass(frozen=True)
class TrainingConfig:
    train_fraction: float
    validation_fraction: float
    random_state: int
    decision_threshold_grid: int
    minimum_events_per_split: int
    bootstrap_repeats: int
    loso_validation_fraction: float
    prospective_test_fraction: float
    prospective_calibration_fraction: float
    prospective_purge_minutes: int
    prospective_min_train_events: int


@dataclass(frozen=True)
class ResearchConfig:
    binance: BinanceConfig
    features: FeatureConfig
    labels: LabelConfig
    training: TrainingConfig


def _int_key_dict(value: dict[Any, Any]) -> dict[int, float]:
    return {int(key): float(item) for key, item in value.items()}


def _int_tuple(value: list[Any] | tuple[Any, ...]) -> tuple[int, ...]:
    return tuple(int(item) for item in value)


_DEFAULT_CONFIG_PATH = Path("configs/research.yaml")
_DEFAULT_CONFIG_RESOURCE = "default_research.yaml"


def _read_config_text(path: str | Path) -> str:
    config_path = Path(path)
    if config_path.exists():
        return config_path.read_text(encoding="utf-8")

    if config_path == _DEFAULT_CONFIG_PATH:
        return (
            resources.files("research")
            .joinpath(_DEFAULT_CONFIG_RESOURCE)
            .read_text(encoding="utf-8")
        )

    raise FileNotFoundError(f"Config file not found: {config_path}")


def load_config(path: str | Path = _DEFAULT_CONFIG_PATH) -> ResearchConfig:
    raw = yaml.safe_load(_read_config_text(path))
    labels = dict(raw["labels"])
    labels["pump_horizons"] = _int_key_dict(labels["pump_horizons"])
    labels["pump_volatility_multipliers"] = _int_key_dict(
        labels["pump_volatility_multipliers"]
    )
    labels["severity_thresholds"] = _int_key_dict(labels["severity_thresholds"])
    labels["forward_horizons"] = _int_tuple(labels["forward_horizons"])
    labels["warning_offsets_minutes"] = _int_tuple(labels["warning_offsets_minutes"])

    return ResearchConfig(
        binance=BinanceConfig(**raw["binance"]),
        features=FeatureConfig(**raw["features"]),
        labels=LabelConfig(**labels),
        training=TrainingConfig(**raw["training"]),
    )
