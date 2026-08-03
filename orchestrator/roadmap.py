from __future__ import annotations

from pathlib import Path

import yaml

from orchestrator.models import RiskLevel, TaskSpec


class RoadmapError(RuntimeError):
    pass


def load_roadmap(path: Path) -> list[TaskSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = raw.get("tasks", [])
    result: list[TaskSpec] = []
    for item in tasks:
        try:
            result.append(
                TaskSpec(
                    task_id=str(item["id"]),
                    title=str(item["title"]),
                    description=str(item["description"]),
                    acceptance_criteria=tuple(str(x) for x in item.get("acceptance_criteria", [])),
                    allowed_paths=tuple(str(x) for x in item.get("allowed_paths", [])),
                    risk_level=RiskLevel(str(item.get("risk_level", "HIGH")).upper()),
                    requires_approval=bool(item.get("requires_approval", True)),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise RoadmapError(f"Invalid task entry: {item}") from exc
    return result
