from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Annotated, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from .paper import (
    JsonlOutcomeStore,
    OutcomeTracker,
    PaperMonitor,
    PaperMonitorConfig,
    TelegramSink,
)

_DEFAULT_CONFIG_PATH = Path("configs/monitor.yaml")


class MonitorConfigError(ValueError):
    """Missing, malformed, or unsafe paper-monitor configuration."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class AlertSettings(_StrictModel):
    cooldown_seconds: Annotated[int, Field(ge=0)]
    red_min_independent_groups: Annotated[int, Field(ge=1)]


class TelegramSettings(_StrictModel):
    parse_mode: Literal["HTML"]
    max_message_chars: Annotated[int, Field(ge=2_048)]
    redact_secrets: bool
    disable_web_page_preview: bool

    @model_validator(mode="after")
    def require_redaction(self) -> Self:
        if not self.redact_secrets:
            raise ValueError(
                "redact_secrets must remain enabled"
            )
        return self


class OutcomeSettings(_StrictModel):
    offsets_minutes: tuple[int, ...]
    storage_format: Literal["jsonl"]
    path: Annotated[str, Field(min_length=1)]
    production_model_updates_enabled: bool

    @field_validator(
        "offsets_minutes",
        mode="before",
    )
    @classmethod
    def parse_offsets(
        cls,
        value: object,
    ) -> tuple[int, ...]:
        if not isinstance(value, list):
            raise PydanticCustomError(
                "yaml_sequence",
                "offsets_minutes must be a YAML sequence",
            )
        if any(
            type(item) is not int
            for item in value
        ):
            raise PydanticCustomError(
                "integer_sequence",
                "offsets_minutes must contain integers",
            )
        return tuple(value)

    @model_validator(mode="after")
    def validate_outcomes(self) -> Self:
        offsets = self.offsets_minutes

        if not offsets:
            raise ValueError(
                "offsets_minutes cannot be empty"
            )
        if any(
            offset <= 0
            for offset in offsets
        ):
            raise ValueError(
                "outcome offsets must be positive"
            )
        if (
            tuple(sorted(set(offsets)))
            != offsets
        ):
            raise ValueError(
                "outcome offsets must be sorted and unique"
            )
        if (
            self.production_model_updates_enabled
        ):
            raise ValueError(
                "production model updates must remain disabled"
            )

        return self


class SafetySettings(_StrictModel):
    exchange_trading_enabled: bool
    exchange_trading_credentials_allowed: bool
    order_placement_enabled: bool
    order_cancellation_enabled: bool
    withdrawals_enabled: bool
    automatic_trading_enabled: bool

    @model_validator(mode="after")
    def reject_unsafe_flags(self) -> Self:
        names = (
            "exchange_trading_enabled",
            "exchange_trading_credentials_allowed",
            "order_placement_enabled",
            "order_cancellation_enabled",
            "withdrawals_enabled",
            "automatic_trading_enabled",
        )
        enabled = tuple(
            name
            for name in names
            if getattr(self, name)
        )

        if enabled:
            raise ValueError(
                "unsafe safety flags enabled: "
                + ", ".join(enabled)
            )

        return self


class PaperMonitorSettings(_StrictModel):
    mode: Literal["paper_only"]
    alerts: AlertSettings
    telegram: TelegramSettings
    outcomes: OutcomeSettings
    safety: SafetySettings

    def to_paper_monitor_config(
        self,
    ) -> PaperMonitorConfig:
        return PaperMonitorConfig(
            cooldown_seconds=(
                self.alerts.cooldown_seconds
            ),
            outcome_offsets_minutes=(
                self.outcomes.offsets_minutes
            ),
            red_min_independent_groups=(
                self.alerts
                .red_min_independent_groups
            ),
            max_message_chars=(
                self.telegram.max_message_chars
            ),
            exchange_trading_enabled=(
                self.safety
                .exchange_trading_enabled
            ),
            production_model_updates_enabled=(
                self.outcomes
                .production_model_updates_enabled
            ),
        )


class MonitorConfigDocument(_StrictModel):
    paper_monitor: PaperMonitorSettings


def _format_validation_error(
    exc: ValidationError,
) -> str:
    messages: list[str] = []

    for error in exc.errors(
        include_input=False,
        include_url=False,
    ):
        location = ".".join(
            str(part)
            for part in error["loc"]
        ) or "<root>"

        messages.append(
            f"{location}: {error['msg']}"
        )

    return (
        "invalid monitor config: "
        + "; ".join(sorted(messages))
    )



def resolve_monitor_config_path(
    value: str | Path,
    *,
    project_root: str | Path = ".",
) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (
        Path(project_root).resolve()
        / candidate
    ).resolve()


def load_monitor_config(
    path: str | Path = _DEFAULT_CONFIG_PATH,
) -> MonitorConfigDocument:
    config_path = Path(path)

    if not config_path.is_file():
        raise MonitorConfigError(
            "monitor config file not found"
        )

    try:
        text = config_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise MonitorConfigError(
            "monitor config cannot be read"
        ) from exc

    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(
            exc,
            "problem_mark",
            None,
        )

        if mark is None:
            detail = "invalid YAML syntax"
        else:
            detail = (
                "invalid YAML syntax at "
                f"line {mark.line + 1}, "
                f"column {mark.column + 1}"
            )

        raise MonitorConfigError(
            detail
        ) from exc

    try:
        return (
            MonitorConfigDocument
            .model_validate(raw)
        )
    except ValidationError as exc:
        raise MonitorConfigError(
            _format_validation_error(exc)
        ) from exc


def _relative_outcome_path(
    value: str,
) -> PurePosixPath:
    relative = PurePosixPath(value)

    if (
        value.strip() != value
        or "\\" in value
        or relative.as_posix() != value
    ):
        raise MonitorConfigError(
            "outcomes.path must use a normalized "
            "relative POSIX path"
        )

    if relative.is_absolute():
        raise MonitorConfigError(
            "outcomes.path must be relative"
        )

    if (
        not relative.parts
        or relative.parts[0] != "artifacts"
    ):
        raise MonitorConfigError(
            "outcomes.path must remain under artifacts"
        )

    if ".." in relative.parts:
        raise MonitorConfigError(
            "outcomes.path cannot contain "
            "parent traversal"
        )

    if relative.suffix.lower() != ".jsonl":
        raise MonitorConfigError(
            "outcomes.path must end in .jsonl"
        )

    return relative


def resolve_outcome_path(
    value: str,
    *,
    project_root: str | Path = ".",
) -> Path:
    relative = _relative_outcome_path(
        value
    )
    root = Path(project_root).resolve()
    artifacts_root = (
        root / "artifacts"
    ).resolve()
    resolved = root.joinpath(
        *relative.parts
    ).resolve()

    if (
        resolved == artifacts_root
        or artifacts_root
        not in resolved.parents
    ):
        raise MonitorConfigError(
            "outcomes.path escapes the "
            "artifacts directory"
        )

    return resolved


def build_paper_monitor(
    config: MonitorConfigDocument,
    sink: TelegramSink,
    *,
    project_root: str | Path = ".",
) -> PaperMonitor:
    settings = config.paper_monitor
    outcome_path = resolve_outcome_path(
        settings.outcomes.path,
        project_root=project_root,
    )
    tracker = OutcomeTracker(
        settings.outcomes.offsets_minutes,
        store=JsonlOutcomeStore(
            outcome_path
        ),
    )

    return PaperMonitor(
        sink,
        config=(
            settings
            .to_paper_monitor_config()
        ),
        outcome_tracker=tracker,
    )


__all__ = [
    "AlertSettings",
    "MonitorConfigDocument",
    "MonitorConfigError",
    "OutcomeSettings",
    "PaperMonitorSettings",
    "SafetySettings",
    "TelegramSettings",
    "build_paper_monitor",
    "load_monitor_config",
    "resolve_monitor_config_path",
    "resolve_outcome_path",
]
