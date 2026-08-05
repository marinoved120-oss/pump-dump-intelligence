from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from .config import (
    MonitorConfigError,
    build_paper_monitor,
    load_monitor_config,
    resolve_outcome_path,
)
from .paper import PaperAlert, PaperAnalysisOutput
from .preflight import run_monitor_preflight

ReplayStatus = Literal["passed", "failed"]


class PaperReplayError(ValueError):
    """Invalid or unsafe local paper-monitor replay input."""


class _StrictReplayModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
    )


class PaperAnalysisInput(_StrictReplayModel):
    market_phase: Annotated[str, Field(min_length=1)]
    likely_pump_causes: tuple[str, ...]
    spot_role: Annotated[str, Field(min_length=1)]
    futures_role: Annotated[str, Field(min_length=1)]
    open_interest_state: Annotated[str, Field(min_length=1)]
    funding_and_leverage: Annotated[str, Field(min_length=1)]
    liquidation_state: Annotated[str, Field(min_length=1)]
    spot_order_book_evidence: tuple[str, ...]
    futures_order_book_evidence: tuple[str, ...]
    whale_limit_orders: tuple[str, ...]
    spoofing_evidence: tuple[str, ...]
    iceberg_evidence: tuple[str, ...]
    absorption_evidence: tuple[str, ...]
    liquidity_pull_evidence: tuple[str, ...]
    spot_futures_divergence: Annotated[str, Field(min_length=1)]
    technical_structure: Annotated[str, Field(min_length=1)]
    onchain_and_news_context: Annotated[str, Field(min_length=1)]
    supporting_evidence: tuple[str, ...]
    contradictions: tuple[str, ...]
    missing_data: tuple[str, ...]
    alternative_scenarios: tuple[str, ...]
    invalidation_conditions: tuple[str, ...]
    observed_outcome: Annotated[str, Field(min_length=1)] = "pending"

    @field_validator(
        "likely_pump_causes",
        "spot_order_book_evidence",
        "futures_order_book_evidence",
        "whale_limit_orders",
        "spoofing_evidence",
        "iceberg_evidence",
        "absorption_evidence",
        "liquidity_pull_evidence",
        "supporting_evidence",
        "contradictions",
        "missing_data",
        "alternative_scenarios",
        "invalidation_conditions",
        mode="before",
    )
    @classmethod
    def parse_text_sequence(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError("must be a JSON array")  # noqa: TRY004
        if any(
            not isinstance(item, str)
            or not item.strip()
            for item in value
        ):
            raise ValueError(
                "must contain non-empty strings"
            )
        return tuple(value)

    def to_analysis_output(self) -> PaperAnalysisOutput:
        return PaperAnalysisOutput(
            market_phase=self.market_phase,
            likely_pump_causes=self.likely_pump_causes,
            spot_role=self.spot_role,
            futures_role=self.futures_role,
            open_interest_state=self.open_interest_state,
            funding_and_leverage=self.funding_and_leverage,
            liquidation_state=self.liquidation_state,
            spot_order_book_evidence=(
                self.spot_order_book_evidence
            ),
            futures_order_book_evidence=(
                self.futures_order_book_evidence
            ),
            whale_limit_orders=self.whale_limit_orders,
            spoofing_evidence=self.spoofing_evidence,
            iceberg_evidence=self.iceberg_evidence,
            absorption_evidence=self.absorption_evidence,
            liquidity_pull_evidence=(
                self.liquidity_pull_evidence
            ),
            spot_futures_divergence=(
                self.spot_futures_divergence
            ),
            technical_structure=self.technical_structure,
            onchain_and_news_context=(
                self.onchain_and_news_context
            ),
            supporting_evidence=self.supporting_evidence,
            contradictions=self.contradictions,
            missing_data=self.missing_data,
            alternative_scenarios=(
                self.alternative_scenarios
            ),
            invalidation_conditions=(
                self.invalidation_conditions
            ),
            observed_outcome=self.observed_outcome,
        )


class PaperAlertInput(_StrictReplayModel):
    alert_id: Annotated[str, Field(min_length=1)]
    exchange: Annotated[str, Field(min_length=1)]
    symbol: Annotated[str, Field(min_length=1)]
    created_ts_ms: Annotated[int, Field(ge=0)]
    reference_price: Annotated[float, Field(gt=0)]
    alert_level: Literal[
        "none",
        "yellow",
        "orange",
        "red",
    ]
    score: Annotated[float, Field(ge=0, le=1)]
    confidence: Annotated[float, Field(ge=0, le=1)]
    data_quality_score: Annotated[
        float,
        Field(ge=0, le=1),
    ]
    independent_evidence_groups: tuple[str, ...]
    analysis: PaperAnalysisInput

    @field_validator(
        "independent_evidence_groups",
        mode="before",
    )
    @classmethod
    def parse_evidence_groups(
        cls,
        value: object,
    ) -> tuple[str, ...]:
        if not isinstance(value, list):
            raise ValueError("must be a JSON array")  # noqa: TRY004
        if any(
            not isinstance(item, str)
            or not item.strip()
            for item in value
        ):
            raise ValueError(
                "must contain non-empty strings"
            )
        return tuple(value)

    def to_paper_alert(self) -> PaperAlert:
        return PaperAlert(
            alert_id=self.alert_id,
            exchange=self.exchange,
            symbol=self.symbol,
            created_ts_ms=self.created_ts_ms,
            reference_price=self.reference_price,
            alert_level=self.alert_level,
            score=self.score,
            confidence=self.confidence,
            data_quality_score=self.data_quality_score,
            independent_evidence_groups=(
                self.independent_evidence_groups
            ),
            analysis=self.analysis.to_analysis_output(),
        )


class AlertReplayEvent(_StrictReplayModel):
    type: Literal["alert"]
    event_ts_ms: Annotated[int, Field(ge=0)]
    alert: PaperAlertInput

    @model_validator(mode="after")
    def validate_event_time(self) -> Self:
        if self.event_ts_ms < self.alert.created_ts_ms:
            raise ValueError(
                "event_ts_ms cannot precede alert creation"
            )
        return self


class InvalidationReplayEvent(_StrictReplayModel):
    type: Literal["invalidation"]
    event_ts_ms: Annotated[int, Field(ge=0)]
    alert_id: Annotated[str, Field(min_length=1)]
    reason: Annotated[str, Field(min_length=1)]

    @field_validator("reason")
    @classmethod
    def require_normalized_reason(
        cls,
        value: str,
    ) -> str:
        if " ".join(value.split()) != value:
            raise ValueError(
                "reason must use normalized whitespace"
            )
        return value


class PriceReplayEvent(_StrictReplayModel):
    type: Literal["price"]
    event_ts_ms: Annotated[int, Field(ge=0)]
    prices_by_symbol: Mapping[str, float]

    @field_validator("prices_by_symbol")
    @classmethod
    def validate_prices(
        cls,
        value: Mapping[str, float],
    ) -> Mapping[str, float]:
        if not value:
            raise ValueError(
                "prices_by_symbol cannot be empty"
            )
        if any(
            not symbol.strip()
            or symbol.strip() != symbol
            or symbol.upper() != symbol
            for symbol in value
        ):
            raise ValueError(
                "price symbols must be non-empty uppercase strings"
            )
        if any(price <= 0 for price in value.values()):
            raise ValueError(
                "observed prices must be positive"
            )
        return MappingProxyType(
            dict(sorted(value.items()))
        )


AlertStreamEvent = AlertReplayEvent | InvalidationReplayEvent

_ALERT_EVENT_ADAPTER = TypeAdapter(
    Annotated[
        AlertStreamEvent,
        Field(discriminator="type"),
    ]
)
_PRICE_EVENT_ADAPTER = TypeAdapter(PriceReplayEvent)


@dataclass(frozen=True)
class _LoadedEvent:
    event_ts_ms: int
    source_rank: int
    line_number: int
    value: AlertStreamEvent | PriceReplayEvent


@dataclass(frozen=True)
class PaperReplayReport:
    schema_version: int
    status: ReplayStatus
    config_path: str
    alerts_path: str
    prices_path: str
    messages_path: str
    preflight_status: str
    events_processed: int
    alerts_published: int
    alerts_suppressed: int
    invalidations_published: int
    invalidations_suppressed: int
    outcomes_recorded: int
    messages_appended: int
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "config_path": self.config_path,
            "alerts_path": self.alerts_path,
            "prices_path": self.prices_path,
            "messages_path": self.messages_path,
            "preflight_status": self.preflight_status,
            "events_processed": self.events_processed,
            "alerts_published": self.alerts_published,
            "alerts_suppressed": self.alerts_suppressed,
            "invalidations_published": (
                self.invalidations_published
            ),
            "invalidations_suppressed": (
                self.invalidations_suppressed
            ),
            "outcomes_recorded": self.outcomes_recorded,
            "messages_appended": self.messages_appended,
            "errors": list(self.errors),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class LocalJsonlMessageSink:
    """Append-only local sink for formatted paper messages."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.appended = 0

    def send(self, text: str) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with self.path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {"message": text},
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
        self.appended += 1


def _display_path(
    value: str | Path,
    *,
    project_root: str | Path,
) -> str:
    candidate = Path(value)
    root = Path(project_root).resolve()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (root / candidate).resolve()
    )
    try:
        return (
            resolved.relative_to(root).as_posix()
        )
    except (OSError, ValueError):
        return candidate.name or "<path>"


def _resolve_artifact_jsonl(
    value: str,
    *,
    project_root: str | Path,
    role: str,
    must_exist: bool,
) -> Path:
    relative = PurePosixPath(value)
    if (
        value.strip() != value
        or "\\" in value
        or relative.as_posix() != value
    ):
        raise PaperReplayError(
            f"{role} path must use a normalized relative POSIX path"
        )
    if relative.is_absolute():
        raise PaperReplayError(
            f"{role} path must be relative"
        )
    if ".." in relative.parts:
        raise PaperReplayError(
            f"{role} path cannot contain parent traversal"
        )
    if (
        not relative.parts
        or relative.parts[0] != "artifacts"
    ):
        raise PaperReplayError(
            f"{role} path must be under artifacts"
        )
    if relative.suffix.lower() != ".jsonl":
        raise PaperReplayError(
            f"{role} path must end in .jsonl"
        )

    root = Path(project_root).resolve()
    artifacts_root = (root / "artifacts").resolve()
    resolved = root.joinpath(*relative.parts).resolve()
    if (
        resolved == artifacts_root
        or artifacts_root not in resolved.parents
    ):
        raise PaperReplayError(
            f"{role} path escapes the artifacts directory"
        )
    if must_exist and not resolved.is_file():
        raise PaperReplayError(
            f"{role} file not found"
        )
    if (
        not must_exist
        and resolved.exists()
        and not resolved.is_file()
    ):
        raise PaperReplayError(
            f"{role} path is not a file"
        )
    return resolved


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
    return "; ".join(sorted(messages))


def _load_jsonl_events(
    path: Path,
    *,
    source_rank: int,
    role: str,
    adapter: TypeAdapter[object],
) -> tuple[_LoadedEvent, ...]:
    try:
        lines = path.read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError as exc:
        raise PaperReplayError(
            f"{role} file cannot be read"
        ) from exc

    events: list[_LoadedEvent] = []
    for line_number, line in enumerate(
        lines,
        start=1,
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PaperReplayError(
                f"{role} JSON syntax invalid at line {line_number}"
            ) from exc
        try:
            value = adapter.validate_python(raw)
        except ValidationError as exc:
            detail = _format_validation_error(exc)
            raise PaperReplayError(
                f"{role} event invalid at line {line_number}: {detail}"
            ) from exc
        events.append(
            _LoadedEvent(
                event_ts_ms=value.event_ts_ms,
                source_rank=source_rank,
                line_number=line_number,
                value=value,
            )
        )

    if not events:
        raise PaperReplayError(
            f"{role} file contains no events"
        )
    return tuple(events)


def _validate_event_sequence(
    events: tuple[_LoadedEvent, ...],
) -> None:
    known_alerts: set[str] = set()
    for loaded in events:
        event = loaded.value
        if isinstance(event, AlertReplayEvent):
            if event.alert.alert_id in known_alerts:
                raise PaperReplayError(
                    "alert stream contains a duplicate alert_id"
                )
            known_alerts.add(event.alert.alert_id)
        elif isinstance(
            event,
            InvalidationReplayEvent,
        ):
            if event.alert_id not in known_alerts:
                raise PaperReplayError(
                    "invalidation references an unknown alert"
                )


def _failed_report(
    *,
    config_path: str,
    alerts_path: str,
    prices_path: str,
    messages_path: str,
    preflight_status: str,
    error: str,
) -> PaperReplayReport:
    return PaperReplayReport(
        schema_version=1,
        status="failed",
        config_path=config_path,
        alerts_path=alerts_path,
        prices_path=prices_path,
        messages_path=messages_path,
        preflight_status=preflight_status,
        events_processed=0,
        alerts_published=0,
        alerts_suppressed=0,
        invalidations_published=0,
        invalidations_suppressed=0,
        outcomes_recorded=0,
        messages_appended=0,
        errors=(error,),
    )


def run_paper_replay(
    alerts_path: str = (
        "artifacts/paper-alert-events.jsonl"
    ),
    prices_path: str = (
        "artifacts/paper-price-events.jsonl"
    ),
    messages_path: str = (
        "artifacts/paper-messages.jsonl"
    ),
    *,
    config_path: str | Path = (
        "configs/monitor.yaml"
    ),
    project_root: str | Path = ".",
    environ: Mapping[str, str] | None = None,
) -> PaperReplayReport:
    display_config = _display_path(
        config_path,
        project_root=project_root,
    )
    display_alerts = _display_path(
        alerts_path,
        project_root=project_root,
    )
    display_prices = _display_path(
        prices_path,
        project_root=project_root,
    )
    display_messages = _display_path(
        messages_path,
        project_root=project_root,
    )

    preflight = run_monitor_preflight(
        config_path,
        project_root=project_root,
        environ=environ,
    )
    if preflight.status != "passed":
        failed_details = tuple(
            check.detail
            for check in preflight.checks
            if check.status == "failed"
        )
        return _failed_report(
            config_path=display_config,
            alerts_path=display_alerts,
            prices_path=display_prices,
            messages_path=display_messages,
            preflight_status=preflight.status,
            error=(
                "; ".join(failed_details)
                or "paper-monitor preflight failed"
            ),
        )

    try:
        resolved_alerts = _resolve_artifact_jsonl(
            alerts_path,
            project_root=project_root,
            role="alerts input",
            must_exist=True,
        )
        resolved_prices = _resolve_artifact_jsonl(
            prices_path,
            project_root=project_root,
            role="prices input",
            must_exist=True,
        )
        resolved_messages = _resolve_artifact_jsonl(
            messages_path,
            project_root=project_root,
            role="messages output",
            must_exist=False,
        )
        if len(
            {
                resolved_alerts,
                resolved_prices,
                resolved_messages,
            }
        ) != 3:
            raise PaperReplayError(
                "alerts, prices and messages paths must be distinct"
            )

        alert_events = _load_jsonl_events(
            resolved_alerts,
            source_rank=0,
            role="alerts input",
            adapter=_ALERT_EVENT_ADAPTER,
        )
        price_events = _load_jsonl_events(
            resolved_prices,
            source_rank=1,
            role="prices input",
            adapter=_PRICE_EVENT_ADAPTER,
        )
        events = tuple(
            sorted(
                alert_events + price_events,
                key=lambda item: (
                    item.event_ts_ms,
                    item.source_rank,
                    item.line_number,
                ),
            )
        )
        _validate_event_sequence(events)

        config = load_monitor_config(config_path)
        outcome_path = resolve_outcome_path(
            config.paper_monitor.outcomes.path,
            project_root=project_root,
        )
        if outcome_path in {
            resolved_alerts,
            resolved_prices,
            resolved_messages,
        }:
            raise PaperReplayError(
                "replay paths must be distinct from the outcome store"
            )
        sink = LocalJsonlMessageSink(
            resolved_messages
        )
        try:
            monitor = build_paper_monitor(
                config,
                sink,
                project_root=project_root,
            )
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as exc:
            raise PaperReplayError(
                "configured outcome store is invalid"
            ) from exc

        existing_outcomes = len(
            monitor.outcomes.records
        )
        alerts_published = 0
        alerts_suppressed = 0
        invalidations_published = 0
        invalidations_suppressed = 0

        for loaded in events:
            event = loaded.value
            try:
                if isinstance(
                    event,
                    AlertReplayEvent,
                ):
                    delivery = monitor.publish(
                        event.alert.to_paper_alert(),
                        now_ts_ms=event.event_ts_ms,
                    )
                    if delivery.sent:
                        alerts_published += 1
                    else:
                        alerts_suppressed += 1
                elif isinstance(
                    event,
                    InvalidationReplayEvent,
                ):
                    delivery = (
                        monitor.publish_invalidation(
                            event.alert_id,
                            event.reason,
                            now_ts_ms=(
                                event.event_ts_ms
                            ),
                        )
                    )
                    if delivery.sent:
                        invalidations_published += 1
                    else:
                        invalidations_suppressed += 1
                else:
                    monitor.capture_due_outcomes(
                        now_ts_ms=event.event_ts_ms,
                        prices_by_symbol=(
                            event.prices_by_symbol
                        ),
                    )
            except (KeyError, ValueError) as exc:
                raise PaperReplayError(
                    "runtime event sequence rejected"
                ) from exc

        outcomes_recorded = (
            len(monitor.outcomes.records)
            - existing_outcomes
        )
        return PaperReplayReport(
            schema_version=1,
            status="passed",
            config_path=display_config,
            alerts_path=display_alerts,
            prices_path=display_prices,
            messages_path=display_messages,
            preflight_status=preflight.status,
            events_processed=len(events),
            alerts_published=alerts_published,
            alerts_suppressed=alerts_suppressed,
            invalidations_published=(
                invalidations_published
            ),
            invalidations_suppressed=(
                invalidations_suppressed
            ),
            outcomes_recorded=outcomes_recorded,
            messages_appended=sink.appended,
            errors=(),
        )
    except (MonitorConfigError, PaperReplayError) as exc:
        return _failed_report(
            config_path=display_config,
            alerts_path=display_alerts,
            prices_path=display_prices,
            messages_path=display_messages,
            preflight_status=preflight.status,
            error=str(exc),
        )
    except OSError:
        return _failed_report(
            config_path=display_config,
            alerts_path=display_alerts,
            prices_path=display_prices,
            messages_path=display_messages,
            preflight_status=preflight.status,
            error="local replay storage operation failed",
        )


__all__ = [
    "AlertReplayEvent",
    "InvalidationReplayEvent",
    "LocalJsonlMessageSink",
    "PaperAlertInput",
    "PaperAnalysisInput",
    "PaperReplayError",
    "PaperReplayReport",
    "PriceReplayEvent",
    "ReplayStatus",
    "run_paper_replay",
]
