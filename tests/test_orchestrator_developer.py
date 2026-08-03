import pytest

from orchestrator.developer import DeveloperError, _extract_output_text, _parse_json_text


def test_extract_output_text_from_responses_payload() -> None:
    payload = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"summary":"ok"}'}],
            }
        ]
    }
    assert _extract_output_text(payload) == '{"summary":"ok"}'


def test_parse_json_fence() -> None:
    assert _parse_json_text('```json\n{"summary":"ok"}\n```')["summary"] == "ok"


def test_parse_invalid_json_fails() -> None:
    with pytest.raises(DeveloperError):
        _parse_json_text("not-json")
