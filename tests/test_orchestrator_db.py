from pathlib import Path

from orchestrator.db import OrchestratorDB
from orchestrator.models import ChangeStatus, RiskLevel, TaskSpec


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="T-1",
        title="Test change",
        description="A controlled change",
        acceptance_criteria=("tests pass",),
        allowed_paths=("research/live",),
        risk_level=RiskLevel.HIGH,
    )


def test_change_lifecycle_and_decision_audit(tmp_path: Path) -> None:
    db = OrchestratorDB(tmp_path / "state.sqlite3")
    db.create_change("CHANGE-1", _task())
    db.update_change(
        "CHANGE-1",
        status=ChangeStatus.PENDING_APPROVAL,
        changed_paths_json=db.encode_paths(["research/live/schema.py"]),
    )
    change = db.get_change("CHANGE-1")
    assert change is not None
    assert change["status"] == "PENDING_APPROVAL"
    db.record_decision("CHANGE-1", "APPROVE", 123)
    assert len(db.list_changes(ChangeStatus.PENDING_APPROVAL)) == 1


def test_settings_round_trip(tmp_path: Path) -> None:
    db = OrchestratorDB(tmp_path / "state.sqlite3")
    db.set_setting("constitution_sha256", "abc")
    assert db.get_setting("constitution_sha256") == "abc"
