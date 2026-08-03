from pathlib import Path

import pytest

from orchestrator.constitution import Constitution, ConstitutionError


def test_constitution_blocks_protected_file(tmp_path: Path) -> None:
    path = tmp_path / "PROJECT_CONSTITUTION.yaml"
    path.write_text(
        "governance:\n  protected_paths:\n    - PROJECT_CONSTITUTION.yaml\n",
        encoding="utf-8",
    )
    constitution = Constitution.load(path)
    with pytest.raises(ConstitutionError):
        constitution.validate_changed_paths(["PROJECT_CONSTITUTION.yaml"])


def test_constitution_allows_critical_protected_change(tmp_path: Path) -> None:
    path = tmp_path / "PROJECT_CONSTITUTION.yaml"
    path.write_text(
        "governance:\n  protected_paths:\n    - PROJECT_CONSTITUTION.yaml\n",
        encoding="utf-8",
    )
    constitution = Constitution.load(path)
    constitution.validate_changed_paths(
        ["PROJECT_CONSTITUTION.yaml"], critical_approved=True
    )


def test_constitution_hash_detects_change(tmp_path: Path) -> None:
    path = tmp_path / "PROJECT_CONSTITUTION.yaml"
    path.write_text("version: 1\n", encoding="utf-8")
    first = Constitution.load(path)
    path.write_text("version: 2\n", encoding="utf-8")
    second = Constitution.load(path)
    with pytest.raises(ConstitutionError):
        second.verify_hash(first.sha256)
