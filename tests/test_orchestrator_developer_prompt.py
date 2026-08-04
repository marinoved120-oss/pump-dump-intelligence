from types import SimpleNamespace

from orchestrator.developer import OpenAIDeveloper
from orchestrator.models import ChangeStatus
from orchestrator.pipeline import DevelopmentPipeline


def test_base_prompt_requires_complete_runtime_implementation(tmp_path) -> None:
    task = SimpleNamespace(
        task_id="V030-002",
        title="Binance WebSocket recorder",
        description=(
            "Implement WebSocket collectors with REST snapshot bootstrap, "
            "reconnect backoff and automatic resynchronization."
        ),
        acceptance_criteria=(
            "Raw events are retained before aggregation.",
            "Any missing update forces snapshot resynchronization.",
        ),
        allowed_paths=(),
    )

    prompt = OpenAIDeveloper._base_prompt(
        tmp_path,
        task,
        "Test constitution",
    )

    assert "IMPLEMENTATION COMPLETENESS CONTRACT" in prompt
    assert "Do not substitute helpers" in prompt
    assert "working lifecycle code" in prompt
    assert "connect, bootstrap, normal processing, failure, reconnect, and recovery" in prompt
    assert "audit every acceptance criterion" in prompt
    assert task.acceptance_criteria[0] in prompt
    assert task.acceptance_criteria[1] in prompt

def test_base_prompt_includes_previous_reviewer_feedback(tmp_path) -> None:
    task = SimpleNamespace(
        task_id="V030-002",
        title="Binance WebSocket recorder",
        description="Implement real WebSocket collectors.",
        acceptance_criteria=("Reconnect and recover after a gap.",),
        allowed_paths=(),
    )

    prompt = OpenAIDeveloper._base_prompt(
        tmp_path,
        task,
        "Test constitution",
        reviewer_feedback=(
            "The implementation contains only an offline parser; "
            "WebSocket and REST lifecycle code is missing."
        ),
    )

    assert "PREVIOUS REVIEWER FEEDBACK FOR THIS SAME TASK" in prompt
    assert "WebSocket and REST lifecycle code is missing" in prompt
    assert "do not repeat the rejected architecture" in prompt


def test_latest_rejection_feedback_is_scoped_to_same_task() -> None:
    rows = [
        {
            "task_id": "V030-003",
            "status": ChangeStatus.REJECTED.value,
            "rejection_reason": "Feedback for another task",
        },
        {
            "task_id": "V030-002",
            "status": ChangeStatus.FAILED.value,
            "rejection_reason": "Failure is not reviewer feedback",
        },
        {
            "task_id": "V030-002",
            "status": ChangeStatus.REJECTED.value,
            "rejection_reason": "  Latest relevant rejection  ",
        },
        {
            "task_id": "V030-002",
            "status": ChangeStatus.REJECTED.value,
            "rejection_reason": "Older rejection",
        },
    ]

    pipeline = DevelopmentPipeline.__new__(DevelopmentPipeline)
    pipeline.db = SimpleNamespace(list_changes=lambda limit=20: rows)

    assert (
        pipeline._latest_rejection_feedback("V030-002")
        == "Latest relevant rejection"
    )


def test_latest_rejection_feedback_returns_none_without_reason() -> None:
    rows = [
        {
            "task_id": "V030-002",
            "status": ChangeStatus.REJECTED.value,
            "rejection_reason": "   ",
        }
    ]

    pipeline = DevelopmentPipeline.__new__(DevelopmentPipeline)
    pipeline.db = SimpleNamespace(list_changes=lambda limit=20: rows)

    assert pipeline._latest_rejection_feedback("V030-002") is None

