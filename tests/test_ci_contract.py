from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
INTEGRITY_SCRIPT = ROOT / "scripts" / "ci" / "verify_release_integrity.py"


def test_ci_workflow_has_required_triggers_and_permissions() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "\non:\n" in workflow
    assert "\n  push:\n" in workflow
    assert "\n  pull_request:\n" in workflow
    assert "\npermissions:\n  contents: read\n" in workflow
    assert "persist-credentials: false" in workflow
    assert 'sudo ln -s "$GITHUB_WORKSPACE" /workspace' in workflow


def test_ci_workflow_runs_required_validation() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    required_commands = (
        'python -m pip install ".[dev]"',
        "python -m pytest",
        "python -m compileall -q research orchestrator",
        "python scripts/ci/verify_release_integrity.py",
    )

    for command in required_commands:
        assert command in workflow


def test_ci_workflow_does_not_request_exchange_secrets() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8").lower()

    forbidden_values = (
        "secrets.",
        "binance_api_key",
        "binance_api_secret",
        "exchange_api_key",
        "exchange_api_secret",
    )

    for value in forbidden_values:
        assert value not in workflow


def test_release_integrity_script_has_required_contract() -> None:
    script = INTEGRITY_SCRIPT.read_text(encoding="utf-8")

    compile(script, str(INTEGRITY_SCRIPT), "exec")

    required_values = (
        "research/default_research.yaml",
        "pip",
        "wheel",
        "install",
        "create_order",
        "place_order",
        "send_order",
        "cancel_order",
        "withdraw",
        "INSTALLED WHEEL CONFIG FALLBACK: OK",
        "RELEASE INTEGRITY: OK",
    )

    for value in required_values:
        assert value in script
