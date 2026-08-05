from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Optional, Protocol, Tuple


AlertLevel = Literal["none", "yellow", "orange", "red"]
DeliveryKind = Literal["initial", "invalidation", "outcome"]


REQUIRED_ANALYSIS_OUTPUT_FIELDS = (
    "market_phase",
    "likely_pump_causes",
    "spot_role",
    "futures_role",
    "open_interest_state",
    "funding_and_leverage",
    "liquidation_state",
    "spot_order_book_evidence",
    "futures_order_book_evidence",
    "whale_limit_orders",
    "spoofing_evidence",
    "iceberg_evidence",
    "absorption_evidence",
    "liquidity_pull_evidence",
    "spot_futures_divergence",
    "technical_structure",
    "onchain_and_news_context",
    "supporting_evidence",
    "contradictions",
    "missing_data",
    "alternative_scenarios",
    "invalidation_conditions",
    "observed_outcome",
)


_SECRET_PATTERN = re.compile(
    r"\b(api[_-]?key|token|secret|password)\b"
    r"\s*[:=]\s*([^\s,;]+)",
    re.IGNORECASE,
)


def _redact_secrets(value: object) -> str:
    text = str(value)
    return _SECRET_PATTERN.sub(
        lambda match: f"{match.group(1)}=[REDACTED]",
        text,
    )


def _short_text(
    value: object,
    limit: int = 110,
) -> str:
    text = " ".join(_redact_secrets(value).split())
    if not text:
        return "not reported"
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "?"


def _display_value(value: object) -> str:
    if isinstance(value, tuple):
        if not value:
            text = "none reported"
        else:
            visible = [
                _short_text(item, 64)
                for item in value[:3]
            ]
            if len(value) > 3:
                visible.append(
                    f"+{len(value) - 3} more"
                )
            text = " | ".join(visible)
    else:
        text = _short_text(value)

    return html.escape(_short_text(text, 110))


@dataclass(frozen=True)
class PaperMonitorConfig:
    cooldown_seconds: int = 900
    outcome_offsets_minutes: Tuple[int, ...] = (
        5,
        15,
        30,
        60,
        240,
    )
    red_min_independent_groups: int = 3
    max_message_chars: int = 4096

    exchange_trading_enabled: bool = False
    production_model_updates_enabled: bool = False

    def __post_init__(self) -> None:
        if self.cooldown_seconds < 0:
            raise ValueError(
                "cooldown_seconds cannot be negative"
            )
        if self.red_min_independent_groups < 1:
            raise ValueError(
                "red_min_independent_groups must be positive"
            )
        if self.max_message_chars < 2_048:
            raise ValueError(
                "max_message_chars must be at least 2048"
            )

        offsets = self.outcome_offsets_minutes
        if not offsets:
            raise ValueError(
                "outcome_offsets_minutes cannot be empty"
            )
        if any(offset <= 0 for offset in offsets):
            raise ValueError(
                "outcome offsets must be positive"
            )
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError(
                "outcome offsets must be sorted and unique"
            )

        if self.exchange_trading_enabled:
            raise ValueError(
                "paper monitor cannot enable exchange trading"
            )
        if self.production_model_updates_enabled:
            raise ValueError(
                "paper outcomes cannot update production models"
            )


@dataclass(frozen=True)
class PaperAnalysisOutput:
    market_phase: str
    likely_pump_causes: Tuple[str, ...]
    spot_role: str
    futures_role: str
    open_interest_state: str
    funding_and_leverage: str
    liquidation_state: str
    spot_order_book_evidence: Tuple[str, ...]
    futures_order_book_evidence: Tuple[str, ...]
    whale_limit_orders: Tuple[str, ...]
    spoofing_evidence: Tuple[str, ...]
    iceberg_evidence: Tuple[str, ...]
    absorption_evidence: Tuple[str, ...]
    liquidity_pull_evidence: Tuple[str, ...]
    spot_futures_divergence: str
    technical_structure: str
    onchain_and_news_context: str
    supporting_evidence: Tuple[str, ...]
    contradictions: Tuple[str, ...]
    missing_data: Tuple[str, ...]
    alternative_scenarios: Tuple[str, ...]
    invalidation_conditions: Tuple[str, ...]
    observed_outcome: str = "pending"


@dataclass(frozen=True)
class PaperAlert:
    alert_id: str
    exchange: str
    symbol: str
    created_ts_ms: int
    reference_price: float

    alert_level: AlertLevel
    score: float
    confidence: float
    data_quality_score: float

    independent_evidence_groups: Tuple[str, ...]
    analysis: PaperAnalysisOutput

    def __post_init__(self) -> None:
        if not self.alert_id.strip():
            raise ValueError("alert_id cannot be empty")
        if not self.exchange.strip():
            raise ValueError("exchange cannot be empty")
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty")
        if self.created_ts_ms < 0:
            raise ValueError(
                "created_ts_ms cannot be negative"
            )
        if self.reference_price <= 0:
            raise ValueError(
                "reference_price must be positive"
            )

        for name in (
            "score",
            "confidence",
            "data_quality_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        object.__setattr__(
            self,
            "symbol",
            self.symbol.upper(),
        )
        object.__setattr__(
            self,
            "independent_evidence_groups",
            tuple(
                sorted(
                    set(
                        self.independent_evidence_groups
                    )
                )
            ),
        )

    def effective_alert_level(
        self,
        config: PaperMonitorConfig,
    ) -> AlertLevel:
        if (
            self.alert_level == "red"
            and len(self.independent_evidence_groups)
            < config.red_min_independent_groups
        ):
            return "orange"
        return self.alert_level


@dataclass(frozen=True)
class MonitorDelivery:
    sent: bool
    kind: DeliveryKind
    reason: str
    effective_alert_level: AlertLevel
    message: str = ""


@dataclass(frozen=True)
class OutcomeCheckpoint:
    alert_id: str
    symbol: str
    offset_minutes: int
    scheduled_ts_ms: int


@dataclass(frozen=True)
class OutcomeRecord:
    alert_id: str
    exchange: str
    symbol: str
    offset_minutes: int
    scheduled_ts_ms: int
    observed_ts_ms: int
    reference_price: float
    observed_price: float
    return_bps: float


class TelegramSink(Protocol):
    """Minimal read-only Telegram delivery boundary."""

    def send(self, text: str) -> None:
        ...


class TelegramReportFormatter:
    """Render compact HTML reports using the constitution contract."""

    def __init__(
        self,
        config: Optional[PaperMonitorConfig] = None,
    ) -> None:
        self.config = config or PaperMonitorConfig()

    def _validate_size(self, message: str) -> str:
        if len(message) > self.config.max_message_chars:
            raise ValueError(
                "Telegram report exceeds configured size"
            )
        return message

    def format_initial(
        self,
        alert: PaperAlert,
    ) -> str:
        effective = alert.effective_alert_level(
            self.config
        )

        lines = [
            "?? <b>PAPER MONITOR ? ANALYSIS ONLY</b>",
            (
                f"<b>alert_id</b>: "
                f"{html.escape(alert.alert_id)}"
            ),
            (
                f"<b>market</b>: "
                f"{html.escape(alert.exchange)} / "
                f"{html.escape(alert.symbol)}"
            ),
            (
                f"<b>requested_alert_level</b>: "
                f"{alert.alert_level.upper()}"
            ),
            (
                f"<b>effective_alert_level</b>: "
                f"{effective.upper()}"
            ),
            f"<b>score</b>: {alert.score:.3f}",
            (
                f"<b>confidence</b>: "
                f"{alert.confidence:.3f}"
            ),
            (
                f"<b>data_quality_score</b>: "
                f"{alert.data_quality_score:.3f}"
            ),
            (
                "<b>independent_evidence_groups</b>: "
                + _display_value(
                    alert.independent_evidence_groups
                )
            ),
            "",
        ]

        for field_name in REQUIRED_ANALYSIS_OUTPUT_FIELDS:
            value = getattr(
                alert.analysis,
                field_name,
            )
            lines.append(
                f"<b>{field_name}</b>: "
                f"{_display_value(value)}"
            )

        lines.extend(
            [
                "",
                (
                    "<i>Paper monitoring only. "
                    "No order placement, trading permission, "
                    "or automatic model update.</i>"
                ),
            ]
        )

        return self._validate_size(
            "\n".join(lines)
        )

    def format_invalidation(
        self,
        alert: PaperAlert,
        reason: str,
        ts_ms: int,
    ) -> str:
        message = "\n".join(
            [
                "?? <b>PAPER ALERT INVALIDATION UPDATE</b>",
                (
                    f"<b>alert_id</b>: "
                    f"{html.escape(alert.alert_id)}"
                ),
                (
                    f"<b>market</b>: "
                    f"{html.escape(alert.exchange)} / "
                    f"{html.escape(alert.symbol)}"
                ),
                f"<b>timestamp_ms</b>: {ts_ms}",
                (
                    "<b>triggered_invalidation</b>: "
                    f"{_display_value(reason)}"
                ),
                (
                    "<b>invalidation_conditions</b>: "
                    + _display_value(
                        alert.analysis
                        .invalidation_conditions
                    )
                ),
                (
                    "<i>The previous paper hypothesis is "
                    "invalidated or materially weakened.</i>"
                ),
            ]
        )
        return self._validate_size(message)

    def format_outcome(
        self,
        record: OutcomeRecord,
    ) -> str:
        message = "\n".join(
            [
                "?? <b>PAPER ALERT OUTCOME</b>",
                (
                    f"<b>alert_id</b>: "
                    f"{html.escape(record.alert_id)}"
                ),
                (
                    f"<b>market</b>: "
                    f"{html.escape(record.exchange)} / "
                    f"{html.escape(record.symbol)}"
                ),
                (
                    f"<b>checkpoint_minutes</b>: "
                    f"{record.offset_minutes}"
                ),
                (
                    f"<b>scheduled_ts_ms</b>: "
                    f"{record.scheduled_ts_ms}"
                ),
                (
                    f"<b>observed_ts_ms</b>: "
                    f"{record.observed_ts_ms}"
                ),
                (
                    f"<b>reference_price</b>: "
                    f"{record.reference_price:.12g}"
                ),
                (
                    f"<b>observed_price</b>: "
                    f"{record.observed_price:.12g}"
                ),
                (
                    f"<b>return_bps</b>: "
                    f"{record.return_bps:.4f}"
                ),
                (
                    "<i>Observed outcome is stored for "
                    "paper evaluation only.</i>"
                ),
            ]
        )
        return self._validate_size(message)


class JsonlOutcomeStore:
    """Append-only paper outcome storage."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: OutcomeRecord) -> None:
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
                    asdict(record),
                    sort_keys=True,
                )
                + "\n"
            )

    def read_all(self) -> Tuple[OutcomeRecord, ...]:
        if not self.path.exists():
            return ()

        records: list[OutcomeRecord] = []
        for line in self.path.read_text(
            encoding="utf-8"
        ).splitlines():
            if not line.strip():
                continue
            records.append(
                OutcomeRecord(**json.loads(line))
            )
        return tuple(records)


class OutcomeTracker:
    """Schedule and record immutable paper outcomes."""

    def __init__(
        self,
        offsets_minutes: Tuple[int, ...],
        store: Optional[JsonlOutcomeStore] = None,
    ) -> None:
        self.offsets_minutes = offsets_minutes
        self.store = store
        self._alerts: dict[str, PaperAlert] = {}

        existing = (
            store.read_all()
            if store is not None
            else ()
        )
        self._records: list[OutcomeRecord] = list(
            existing
        )
        self._completed = {
            (
                record.alert_id,
                record.offset_minutes,
            )
            for record in existing
        }

    @property
    def records(self) -> Tuple[OutcomeRecord, ...]:
        return tuple(self._records)

    def register(self, alert: PaperAlert) -> None:
        self._alerts[alert.alert_id] = alert

    def due_checkpoints(
        self,
        now_ts_ms: int,
    ) -> Tuple[OutcomeCheckpoint, ...]:
        due: list[OutcomeCheckpoint] = []

        for alert in self._alerts.values():
            for offset in self.offsets_minutes:
                key = (alert.alert_id, offset)
                if key in self._completed:
                    continue

                scheduled = (
                    alert.created_ts_ms
                    + offset * 60_000
                )
                if now_ts_ms >= scheduled:
                    due.append(
                        OutcomeCheckpoint(
                            alert_id=alert.alert_id,
                            symbol=alert.symbol,
                            offset_minutes=offset,
                            scheduled_ts_ms=scheduled,
                        )
                    )

        return tuple(
            sorted(
                due,
                key=lambda item: (
                    item.scheduled_ts_ms,
                    item.alert_id,
                ),
            )
        )

    def record(
        self,
        checkpoint: OutcomeCheckpoint,
        *,
        observed_ts_ms: int,
        observed_price: float,
    ) -> OutcomeRecord:
        alert = self._alerts.get(
            checkpoint.alert_id
        )
        if alert is None:
            raise KeyError(
                f"Unknown alert: {checkpoint.alert_id}"
            )
        if observed_ts_ms < checkpoint.scheduled_ts_ms:
            raise ValueError(
                "outcome cannot be recorded before checkpoint"
            )
        if observed_price <= 0:
            raise ValueError(
                "observed_price must be positive"
            )

        key = (
            checkpoint.alert_id,
            checkpoint.offset_minutes,
        )
        if key in self._completed:
            raise ValueError(
                "outcome checkpoint already recorded"
            )

        return_bps = (
            observed_price - alert.reference_price
        ) / alert.reference_price * 10_000.0

        record = OutcomeRecord(
            alert_id=alert.alert_id,
            exchange=alert.exchange,
            symbol=alert.symbol,
            offset_minutes=checkpoint.offset_minutes,
            scheduled_ts_ms=checkpoint.scheduled_ts_ms,
            observed_ts_ms=observed_ts_ms,
            reference_price=alert.reference_price,
            observed_price=observed_price,
            return_bps=return_bps,
        )

        self._records.append(record)
        self._completed.add(key)

        if self.store is not None:
            self.store.append(record)

        return record


class PaperMonitor:
    """Cooldown, invalidation, and outcome orchestration."""

    def __init__(
        self,
        sink: TelegramSink,
        *,
        config: Optional[PaperMonitorConfig] = None,
        outcome_tracker: Optional[OutcomeTracker] = None,
    ) -> None:
        self.config = config or PaperMonitorConfig()
        self.sink = sink
        self.formatter = TelegramReportFormatter(
            self.config
        )
        self.outcomes = outcome_tracker or OutcomeTracker(
            self.config.outcome_offsets_minutes
        )

        self._alerts: dict[str, PaperAlert] = {}
        self._last_initial_by_market: dict[
            tuple[str, str],
            int,
        ] = {}
        self._sent_invalidations: set[
            tuple[str, str]
        ] = set()

    def publish(
        self,
        alert: PaperAlert,
        *,
        now_ts_ms: Optional[int] = None,
    ) -> MonitorDelivery:
        now = (
            alert.created_ts_ms
            if now_ts_ms is None
            else now_ts_ms
        )
        market_key = (
            alert.exchange,
            alert.symbol,
        )
        last_sent = self._last_initial_by_market.get(
            market_key
        )
        cooldown_ms = (
            self.config.cooldown_seconds * 1_000
        )

        effective = alert.effective_alert_level(
            self.config
        )

        if (
            last_sent is not None
            and now - last_sent < cooldown_ms
        ):
            return MonitorDelivery(
                sent=False,
                kind="initial",
                reason="cooldown_active",
                effective_alert_level=effective,
            )

        message = self.formatter.format_initial(
            alert
        )
        self.sink.send(message)

        self._alerts[alert.alert_id] = alert
        self._last_initial_by_market[
            market_key
        ] = now
        self.outcomes.register(alert)

        return MonitorDelivery(
            sent=True,
            kind="initial",
            reason="sent",
            effective_alert_level=effective,
            message=message,
        )

    def publish_invalidation(
        self,
        alert_id: str,
        reason: str,
        *,
        now_ts_ms: int,
    ) -> MonitorDelivery:
        alert = self._alerts.get(alert_id)
        if alert is None:
            raise KeyError(
                f"Unknown alert: {alert_id}"
            )

        normalized_reason = " ".join(
            reason.split()
        )
        if not normalized_reason:
            raise ValueError(
                "invalidation reason cannot be empty"
            )

        key = (
            alert_id,
            normalized_reason,
        )
        effective = alert.effective_alert_level(
            self.config
        )

        if key in self._sent_invalidations:
            return MonitorDelivery(
                sent=False,
                kind="invalidation",
                reason="duplicate_invalidation",
                effective_alert_level=effective,
            )

        message = self.formatter.format_invalidation(
            alert,
            normalized_reason,
            now_ts_ms,
        )
        self.sink.send(message)
        self._sent_invalidations.add(key)

        return MonitorDelivery(
            sent=True,
            kind="invalidation",
            reason="sent",
            effective_alert_level=effective,
            message=message,
        )

    def capture_due_outcomes(
        self,
        *,
        now_ts_ms: int,
        prices_by_symbol: Mapping[str, float],
    ) -> Tuple[OutcomeRecord, ...]:
        captured: list[OutcomeRecord] = []

        for checkpoint in self.outcomes.due_checkpoints(
            now_ts_ms
        ):
            if checkpoint.symbol not in prices_by_symbol:
                continue

            record = self.outcomes.record(
                checkpoint,
                observed_ts_ms=now_ts_ms,
                observed_price=prices_by_symbol[
                    checkpoint.symbol
                ],
            )
            self.sink.send(
                self.formatter.format_outcome(record)
            )
            captured.append(record)

        return tuple(captured)


__all__ = [
    "AlertLevel",
    "JsonlOutcomeStore",
    "MonitorDelivery",
    "OutcomeCheckpoint",
    "OutcomeRecord",
    "OutcomeTracker",
    "PaperAlert",
    "PaperAnalysisOutput",
    "PaperMonitor",
    "PaperMonitorConfig",
    "REQUIRED_ANALYSIS_OUTPUT_FIELDS",
    "TelegramReportFormatter",
    "TelegramSink",
]
