from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directories(root: str | Path = ".") -> None:
    base = Path(root)
    for relative in ("data/raw", "data/processed", "artifacts"):
        (base / relative).mkdir(parents=True, exist_ok=True)


def save_frame(frame: pd.DataFrame, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(destination, index=False)
    except (ImportError, ModuleNotFoundError):
        # Lightweight fallback for environments without a parquet engine.
        # Docker installs pyarrow, so production research runs use real parquet.
        frame.to_pickle(destination)
    return destination


def load_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    try:
        return pd.read_parquet(source)
    except (ImportError, ModuleNotFoundError, ValueError):
        return pd.read_pickle(source)


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return destination
