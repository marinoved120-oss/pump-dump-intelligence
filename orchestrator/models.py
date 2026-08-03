from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ChangeStatus(StrEnum):
    QUEUED = "QUEUED"
    GENERATING = "GENERATING"
    VALIDATING = "VALIDATING"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    MERGED = "MERGED"
    FAILED = "FAILED"


class RiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    description: str
    acceptance_criteria: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    risk_level: RiskLevel
    requires_approval: bool = True


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    summary: str
    log_path: Path | None = None
