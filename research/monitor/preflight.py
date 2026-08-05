from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import (
    MonitorConfigDocument,
    MonitorConfigError,
    load_monitor_config,
    resolve_outcome_path,
)

PreflightStatus = Literal["passed", "failed"]


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: PreflightStatus
    detail: str


@dataclass(frozen=True)
class MonitorPreflightReport:
    schema_version: int
    status: PreflightStatus
    config_path: str
    outcome_path: str | None
    checks: tuple[PreflightCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "config_path": self.config_path,
            "outcome_path": self.outcome_path,
            "checks": [
                {
                    "name": check.name,
                    "status": check.status,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


def _display_path(
    path: str | Path,
    *,
    project_root: str | Path,
) -> str:
    candidate = Path(path)
    root = Path(project_root).resolve()

    try:
        return (
            candidate.resolve()
            .relative_to(root)
            .as_posix()
        )
    except (OSError, ValueError):
        return candidate.name or "<config>"


def _is_binance_credential_name(
    name: str,
) -> bool:
    normalized = name.upper()

    if not normalized.startswith("BINANCE_"):
        return False

    if normalized in {
        "BINANCE_KEY",
        "BINANCE_SECRET",
    }:
        return True

    return normalized.endswith(
        (
            "API_KEY",
            "API_SECRET",
            "SECRET_KEY",
            "PRIVATE_KEY",
        )
    )


def _credential_check(
    environ: Mapping[str, str],
) -> PreflightCheck:
    present = sorted(
        {
            name.upper()
            for name, value in environ.items()
            if value.strip()
            and _is_binance_credential_name(name)
        }
    )

    if present:
        return PreflightCheck(
            name="credential_environment",
            status="failed",
            detail=(
                "Binance credential variables present: "
                + ", ".join(present)
            ),
        )

    return PreflightCheck(
        name="credential_environment",
        status="passed",
        detail=(
            "no Binance credential variables present"
        ),
    )


def _safety_check(
    config: MonitorConfigDocument,
) -> PreflightCheck:
    settings = config.paper_monitor
    safety = settings.safety

    unsafe = sorted(
        name
        for name in (
            "automatic_trading_enabled",
            "exchange_trading_credentials_allowed",
            "exchange_trading_enabled",
            "order_cancellation_enabled",
            "order_placement_enabled",
            "withdrawals_enabled",
        )
        if getattr(safety, name)
    )

    if (
        unsafe
        or settings.outcomes
        .production_model_updates_enabled
        or not settings.telegram.redact_secrets
    ):
        return PreflightCheck(
            name="safety_contract",
            status="failed",
            detail=(
                "paper-monitor safety invariants failed"
            ),
        )

    return PreflightCheck(
        name="safety_contract",
        status="passed",
        detail=(
            "trading, credentials, outbound asset transfers and "
            "production updates are disabled"
        ),
    )


def _validate_outcome_target(
    path: Path,
) -> None:
    if path.exists():
        if not path.is_file():
            raise MonitorConfigError(
                "outcomes.path must target a file"
            )
        if not os.access(path, os.W_OK):
            raise MonitorConfigError(
                "existing outcome file is not writable"
            )
        return

    ancestor = path.parent

    while (
        not ancestor.exists()
        and ancestor != ancestor.parent
    ):
        ancestor = ancestor.parent

    if not ancestor.is_dir():
        raise MonitorConfigError(
            "outcome path has no writable directory"
        )

    if not os.access(ancestor, os.W_OK):
        raise MonitorConfigError(
            "outcome directory is not writable"
        )


def _outcome_check(
    config: MonitorConfigDocument,
    *,
    project_root: str | Path,
) -> tuple[PreflightCheck, str | None]:
    try:
        resolved = resolve_outcome_path(
            config.paper_monitor.outcomes.path,
            project_root=project_root,
        )
        _validate_outcome_target(resolved)
    except MonitorConfigError as exc:
        return (
            PreflightCheck(
                name="outcome_storage",
                status="failed",
                detail=str(exc),
            ),
            None,
        )

    root = Path(project_root).resolve()
    display = resolved.relative_to(root).as_posix()

    return (
        PreflightCheck(
            name="outcome_storage",
            status="passed",
            detail=(
                "local append-only JSONL target is valid"
            ),
        ),
        display,
    )


def run_monitor_preflight(
    config_path: str | Path = (
        "configs/monitor.yaml"
    ),
    *,
    project_root: str | Path = ".",
    environ: Mapping[str, str] | None = None,
) -> MonitorPreflightReport:
    display_config = _display_path(
        config_path,
        project_root=project_root,
    )

    try:
        config = load_monitor_config(config_path)
    except MonitorConfigError as exc:
        return MonitorPreflightReport(
            schema_version=1,
            status="failed",
            config_path=display_config,
            outcome_path=None,
            checks=(
                PreflightCheck(
                    name="config_document",
                    status="failed",
                    detail=str(exc),
                ),
            ),
        )

    checks: list[PreflightCheck] = [
        PreflightCheck(
            name="config_document",
            status="passed",
            detail=(
                "strict monitor configuration loaded"
            ),
        ),
        PreflightCheck(
            name="paper_only_mode",
            status="passed",
            detail="mode is paper_only",
        ),
        _safety_check(config),
        _credential_check(
            os.environ
            if environ is None
            else environ
        ),
    ]

    outcome_check, outcome_path = (
        _outcome_check(
            config,
            project_root=project_root,
        )
    )
    checks.append(outcome_check)
    checks.append(
        PreflightCheck(
            name="network_side_effects",
            status="passed",
            detail=(
                "no exchange or Telegram request attempted"
            ),
        )
    )

    status: PreflightStatus = (
        "passed"
        if all(
            check.status == "passed"
            for check in checks
        )
        else "failed"
    )

    return MonitorPreflightReport(
        schema_version=1,
        status=status,
        config_path=display_config,
        outcome_path=outcome_path,
        checks=tuple(checks),
    )


__all__ = [
    "MonitorPreflightReport",
    "PreflightCheck",
    "PreflightStatus",
    "run_monitor_preflight",
]
