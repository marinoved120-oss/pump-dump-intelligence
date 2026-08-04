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


def test_request_uses_strict_structured_output(monkeypatch) -> None:
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def json(self):
            return {
                "output_text": (
                    '{"summary":"summary",'
                    '"rationale":"rationale",'
                    '"risks":[],'
                    '"unified_diff":"diff --git a/a.py b/a.py\\n"}'
                )
            }

    class FakeClient:
        def __init__(self, timeout):
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def post(self, url, *, headers, json):
            captured["url"] = url
            captured["headers"] = headers
            captured["request"] = json
            return FakeResponse()

    monkeypatch.setattr(
        "orchestrator.developer.httpx.Client",
        FakeClient,
    )

    developer = OpenAIDeveloper(
        api_key="test-key",
        model="test-model",
    )
    result = developer._request("test prompt")

    request = captured["request"]
    output_format = request["text"]["format"]
    schema = output_format["schema"]

    assert output_format["type"] == "json_schema"
    assert output_format["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "summary",
        "rationale",
        "risks",
        "unified_diff",
    ]
    assert result["summary"] == "summary"


def test_base_prompt_requires_concrete_network_adapter_and_lifecycle_tests(
    tmp_path,
) -> None:
    task = SimpleNamespace(
        task_id="V030-002",
        title="Binance WebSocket recorder",
        description=(
            "Implement independent Spot and Futures collectors with "
            "REST bootstrap, reconnect and resynchronization."
        ),
        acceptance_criteria=(
            "Spot and futures collectors operate independently.",
            "Missing updates force snapshot resynchronization.",
        ),
        allowed_paths=("research/live", "tests/test_live_collectors.py"),
    )

    prompt = OpenAIDeveloper._base_prompt(
        tmp_path,
        task,
        "Test constitution",
        reviewer_feedback=(
            "The previous implementation supplied only a Transport Protocol "
            "and no concrete WebSocket or HTTP adapter."
        ),
    )

    assert "Dependency injection does not replace a production implementation" in prompt
    assert "concrete runtime adapter" in prompt
    assert "Protocol methods containing only ellipsis" in prompt
    assert "tests must run both collector variants" in prompt
    assert "gap-triggered snapshot resynchronization" in prompt
    assert "disconnect-triggered reconnect" in prompt
    assert "Passing only pre-existing tests is not evidence" in prompt

