from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.developer import _decode_change
from orchestrator.gitops import GitRepo
from orchestrator.models import ChangeStatus, RiskLevel, TaskSpec
from orchestrator.patches import PatchError, normalize_unified_diff, validate_patch_structure
from orchestrator.worker import RoadmapWorker


VALID_PATCH = """diff --git a/research/live/schema.py b/research/live/schema.py
new file mode 100644
--- /dev/null
+++ b/research/live/schema.py
@@ -0,0 +1 @@
+VALUE = 1
"""


def _task() -> TaskSpec:
    return TaskSpec(
        task_id="V030-001",
        title="Schemas",
        description="Add schemas",
        acceptance_criteria=("Works",),
        allowed_paths=("research/live",),
        risk_level=RiskLevel.HIGH,
        requires_approval=True,
    )


def test_normalize_removes_markdown_fence_and_prose() -> None:
    wrapped = f"Explanation that must be removed\n```diff\n{VALID_PATCH}```\n"
    normalized = normalize_unified_diff(wrapped)
    assert normalized.startswith("diff --git")
    assert normalized.endswith("\n")
    assert "```" not in normalized


def test_patch_with_headers_but_no_hunk_is_rejected() -> None:
    with pytest.raises(PatchError, match="no valid @@ hunk"):
        validate_patch_structure("--- a/file.py\n+++ b/file.py\n")


def test_decode_change_accepts_fenced_model_patch() -> None:
    change = _decode_change(
        {
            "summary": "ok",
            "rationale": "test",
            "risks": [],
            "unified_diff": f"```diff\n{VALID_PATCH}```",
        },
        _task(),
    )
    assert change.diff == VALID_PATCH


class _FakeDB:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def list_changes(self, limit: int = 20):
        return self.rows[:limit]


def test_failed_task_is_retryable() -> None:
    worker = RoadmapWorker(
        pipeline=SimpleNamespace(),
        db=_FakeDB([{"task_id": "V030-001", "status": ChangeStatus.FAILED.value}]),
        telegram=None,
        roadmap_path=Path("ROADMAP.yaml"),
    )
    assert "V030-001" not in worker._completed_task_ids()



def test_rejected_task_is_retryable() -> None:
    worker = RoadmapWorker(
        pipeline=SimpleNamespace(),
        db=_FakeDB(
            [{"task_id": "V030-001", "status": ChangeStatus.REJECTED.value}]
        ),
        telegram=None,
        roadmap_path=Path("ROADMAP.yaml"),
    )
    assert "V030-001" not in worker._completed_task_ids()


def test_pending_task_is_not_duplicated() -> None:
    worker = RoadmapWorker(
        pipeline=SimpleNamespace(),
        db=_FakeDB(
            [{"task_id": "V030-001", "status": ChangeStatus.PENDING_APPROVAL.value}]
        ),
        telegram=None,
        roadmap_path=Path("ROADMAP.yaml"),
    )
    assert "V030-001" in worker._completed_task_ids()


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)


def test_cleanup_generated_paths_preserves_unrelated_untracked_file(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-m", "base")

    tracked.write_text("changed\n", encoding="utf-8")
    generated = tmp_path / "research" / "live" / "generated.py"
    generated.parent.mkdir(parents=True)
    generated.write_text("VALUE = 1\n", encoding="utf-8")
    unrelated = tmp_path / "local-note.txt"
    unrelated.write_text("keep me\n", encoding="utf-8")

    repo = GitRepo(tmp_path)
    repo.cleanup_generated_paths(["tracked.txt", "research/live/generated.py"])

    assert tracked.read_text(encoding="utf-8") == "base\n"
    assert not generated.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep me\n"


def test_pipeline_repairs_git_apply_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from orchestrator.config import OrchestratorConfig
    from orchestrator.db import OrchestratorDB
    from orchestrator.developer import GeneratedChange
    from orchestrator.pipeline import DevelopmentPipeline

    root = tmp_path / "repo"
    state = tmp_path / "state"
    (root / "src").mkdir(parents=True)
    (root / "src" / "value.txt").write_text("old\n", encoding="utf-8")
    (root / "PROJECT_CONSTITUTION.yaml").write_text(
        "governance:\n  protected_paths: []\n",
        encoding="utf-8",
    )

    invalid = """diff --git a/src/value.txt b/src/value.txt
--- a/src/value.txt
+++ b/src/value.txt
@@ -1 +1 @@
-not-present
+new
"""
    repaired = """diff --git a/src/value.txt b/src/value.txt
--- a/src/value.txt
+++ b/src/value.txt
@@ -1 +1 @@
-old
+new
"""

    class FakeDeveloper:
        repair_calls = 0

        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return GeneratedChange("initial", "", (), invalid)

        def repair(self, *args, **kwargs):
            FakeDeveloper.repair_calls += 1
            return GeneratedChange("repaired", "", (), repaired)

    monkeypatch.setattr("orchestrator.pipeline.OpenAIDeveloper", FakeDeveloper)
    config = OrchestratorConfig(
        project_root=root,
        state_dir=state,
        telegram_bot_token=None,
        telegram_allowed_user_id=None,
        telegram_chat_id=None,
        openai_api_key="test-key",
        openai_model="test-model",
        polling_timeout_seconds=1,
        worker_enabled=False,
        auto_merge_low_risk=False,
        git_user_name="Test",
        git_user_email="test@example.invalid",
        test_command=("python", "-c", "import sys; sys.exit(0)"),
    )
    db = OrchestratorDB(state / "orchestrator.sqlite3")
    pipeline = DevelopmentPipeline(config, db)
    pipeline.prepare_repository()
    task = TaskSpec(
        task_id="T-1",
        title="Repair patch",
        description="Test repair",
        acceptance_criteria=("file changes",),
        allowed_paths=("src",),
        risk_level=RiskLevel.HIGH,
        requires_approval=True,
    )

    change_id = pipeline.create_proposal(task)
    change = db.get_change(change_id)

    assert FakeDeveloper.repair_calls == 1
    assert change is not None
    assert change["status"] == ChangeStatus.PENDING_APPROVAL.value
    assert pipeline.repo.current_branch() == db.get_setting("base_branch")
    assert pipeline.repo.run("show", f"{change['branch_name']}:src/value.txt").stdout == "new\n"
