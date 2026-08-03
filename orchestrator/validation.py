from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from orchestrator.models import ValidationResult


class ValidationError(RuntimeError):
    pass


def run_validation(
    project_root: Path,
    state_dir: Path,
    change_id: str,
    command: tuple[str, ...],
    timeout_seconds: int = 1800,
) -> ValidationResult:
    if not command:
        raise ValidationError("Validation command is empty")
    log_path = state_dir / "logs" / f"{change_id}-tests.log"
    env = os.environ.copy()
    started = datetime.now(timezone.utc).isoformat()
    completed = subprocess.run(
        list(command),
        cwd=project_root,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
        env=env,
    )
    log = (
        f"started={started}\ncommand={' '.join(command)}\nreturncode={completed.returncode}\n\n"
        f"STDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}\n"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log, encoding="utf-8")
    tail = (completed.stdout + "\n" + completed.stderr).strip()[-1200:]
    summary = f"returncode={completed.returncode}\n{tail}"
    return ValidationResult(ok=completed.returncode == 0, summary=summary, log_path=log_path)
