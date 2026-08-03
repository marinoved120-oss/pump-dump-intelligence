from pathlib import Path

from orchestrator.models import RiskLevel
from orchestrator.roadmap import load_roadmap


def test_roadmap_loads_task_contract(tmp_path: Path) -> None:
    path = tmp_path / "ROADMAP.yaml"
    path.write_text(
        """
tasks:
  - id: T-1
    title: Recorder
    description: Add recorder
    allowed_paths: [research/live]
    acceptance_criteria: [tests pass]
    risk_level: high
""",
        encoding="utf-8",
    )
    tasks = load_roadmap(path)
    assert tasks[0].risk_level == RiskLevel.HIGH
    assert tasks[0].allowed_paths == ("research/live",)
