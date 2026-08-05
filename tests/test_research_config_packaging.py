from importlib import resources
from pathlib import Path

import pytest

from research.config import load_config


def test_packaged_default_matches_repository_config() -> None:
    packaged = (
        resources.files("research")
        .joinpath("default_research.yaml")
        .read_text(encoding="utf-8")
    )
    repository = Path(__file__).resolve().parents[1] / "configs" / "research.yaml"

    assert packaged == repository.read_text(encoding="utf-8")


def test_load_config_falls_back_to_packaged_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = load_config()

    assert config.binance.base_url
    assert config.labels.forward_horizons


def test_load_config_keeps_missing_explicit_paths_strict(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError, match="missing.yaml"):
        load_config(missing)