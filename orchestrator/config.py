from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path


class ConfigError(RuntimeError):
    pass


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int | None = None) -> int | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class OrchestratorConfig:
    project_root: Path
    state_dir: Path
    telegram_bot_token: str | None
    telegram_allowed_user_id: int | None
    telegram_chat_id: int | None
    openai_api_key: str | None
    openai_model: str
    polling_timeout_seconds: int
    worker_enabled: bool
    auto_merge_low_risk: bool
    git_user_name: str
    git_user_email: str
    test_command: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        root = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
        state = Path(os.getenv("ORCHESTRATOR_STATE_DIR", root / "orchestrator_state")).resolve()
        test_command = tuple(shlex.split(os.getenv("ORCHESTRATOR_TEST_COMMAND", "python -m pytest")))
        return cls(
            project_root=root,
            state_dir=state,
            telegram_bot_token=os.getenv("TELEGRAM_DEV_BOT_TOKEN") or None,
            telegram_allowed_user_id=_int("TELEGRAM_ALLOWED_USER_ID"),
            telegram_chat_id=_int("TELEGRAM_DEV_CHAT_ID"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5"),
            polling_timeout_seconds=int(os.getenv("TELEGRAM_POLL_TIMEOUT", "25")),
            worker_enabled=_bool("ORCHESTRATOR_WORKER_ENABLED", False),
            auto_merge_low_risk=_bool("ORCHESTRATOR_AUTO_MERGE_LOW_RISK", False),
            git_user_name=os.getenv("ORCHESTRATOR_GIT_USER_NAME", "PumpDump Orchestrator"),
            git_user_email=os.getenv("ORCHESTRATOR_GIT_USER_EMAIL", "orchestrator@localhost"),
            test_command=test_command,
        )

    def ensure_directories(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.state_dir / "proposals").mkdir(parents=True, exist_ok=True)

    def require_telegram(self) -> None:
        missing: list[str] = []
        if not self.telegram_bot_token:
            missing.append("TELEGRAM_DEV_BOT_TOKEN")
        if self.telegram_allowed_user_id is None:
            missing.append("TELEGRAM_ALLOWED_USER_ID")
        if missing:
            raise ConfigError("Missing Telegram settings: " + ", ".join(missing))
