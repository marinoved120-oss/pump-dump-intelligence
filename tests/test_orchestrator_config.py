from pathlib import Path

from orchestrator.config import OrchestratorConfig


def test_config_defaults_to_worker_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.delenv("ORCHESTRATOR_WORKER_ENABLED", raising=False)
    config = OrchestratorConfig.from_env()
    assert config.project_root == tmp_path.resolve()
    assert config.worker_enabled is False
    assert config.auto_merge_low_risk is False
