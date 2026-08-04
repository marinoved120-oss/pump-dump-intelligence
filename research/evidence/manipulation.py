from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from research.live.schemas import MarketType
from research.orderbook.walls import WallLifecycle


Hypothesis = Literal[
    "spoofing_like",
    "iceberg_like",
    "absorption_like",
]


@dataclass(frozen=True)
class EvidenceConfig:
    """Thresholds for evidence-compatible behavioural hypotheses."""

    spoofing_max_lifetime_ms: int = 5_000
    spoofing_distance_reference_bps: float = 25.0
    spoofing_min_cancellation_ratio: float = 0.60
    spoofing_execution_contradiction_ratio: float = 0.25
    spoofing_repetition_reference: int = 3
    spoofing_min_repetitions: int = 2
    spoofing_min_score: float = 0.60

    iceberg_min_refill_count: int = 2
    iceberg_min_executed_to_visible: float = 1.50
    iceberg_refill_reference_ratio: float = 1.00
    iceberg_min_score: float = 0.60

    absorption_min_aggressive_volume: float = 10.0
    absorption_min_flow_to_depth_ratio: float = 1.00
    absorption_max_price_response_bps: float = 10.0

    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        positive_values = (
            "spoofing_max_lifetime_ms",
            "spoofing_distance_reference_bps",
            "spoofing_repetition_reference",
            "iceberg_min_refill_count",
            "iceberg_min_executed_to_visible",
            "iceberg_refill_reference_ratio",
            "absorption_min_aggressive_volume",
            "absorption_min_flow_to_depth_ratio",
            "absorption_max_price_response_bps",
            "epsilon",
        )
        for name in positive_values:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

        if self.spoofing_min_repetitions < 0:
            raise ValueError(
                "spoofing_min_repetitions cannot be negative"
            )

        unit_interval = (
            "spoofing_min_cancellation_ratio",
            "spoofing_execution_contradiction_ratio",
            "spoofing_min_score",
            "iceberg_min_score",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )


@dataclass(frozen=True)
class AbsorptionWindow:
    exchange: str
    symbol: str
    market_type: MarketType
    aggressive_side: Literal["buy", "sell"]

    start_ts_ms: int
    end_ts_ms: int

    aggressive_volume: float
    reference_visible_depth: float
    start_price: float
    end_price: float

    def __post_init__(self) -> None:
        if self.end_ts_ms < self.start_ts_ms:
            raise ValueError(
                "end_ts_ms cannot precede start_ts_ms"
            )
        if self.aggressive_volume < 0:
            raise ValueError(
                "aggressive_volume cannot be negative"
            )
        if self.reference_visible_depth <= 0:
            raise ValueError(
                "reference_visible_depth must be positive"
            )
        if self.start_price <= 0 or self.end_price <= 0:
            raise ValueError("prices must be positive")


@dataclass(frozen=True)
class EvidenceReport:
    """A hypothesis report, never a finding of intent."""

    hypothesis: Hypothesis
    supported: bool
    score: float
    confidence: float
    wording: str

    exchange: str
    symbol: str
    market_type: MarketType
    side: str

    start_ts_ms: int
    end_ts_ms: int

    evidence: Tuple[str, ...]
    contradictions: Tuple[str, ...]
    metrics: Tuple[Tuple[str, float], ...]

    def metric(self, name: str) -> Optional[float]:
        for metric_name, value in self.metrics:
            if metric_name == name:
                return value
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _confidence(
    score: float,
    evidence: list[str],
    contradictions: list[str],
) -> float:
    total = len(evidence) + len(contradictions)
    support_fraction = (
        len(evidence) / total
        if total
        else 0.0
    )
    return _clamp(
        0.65 * score
        + 0.35 * support_fraction
    )


def _wording(
    label: str,
    supported: bool,
) -> str:
    if supported:
        prefix = (
            "Observed evidence is compatible with "
            f"a {label} hypothesis."
        )
    else:
        prefix = (
            "Observed evidence is insufficient for "
            f"a {label} hypothesis."
        )

    return (
        f"{prefix} This does not establish intent, "
        "manipulation, or wrongdoing."
    )


def _execution_volume(wall: WallLifecycle) -> float:
    return sum(
        max(0.0, observation.executed_size)
        for observation in wall.observations
        if observation.event in (
            "partially_executed",
            "executed",
        )
    )


def _pulled_volume(wall: WallLifecycle) -> float:
    # Use liquidity_pulled only. The tracker may also emit a
    # cancelled event with the same quantity.
    return sum(
        max(0.0, observation.pulled_size)
        for observation in wall.observations
        if observation.event == "liquidity_pulled"
    )


def _refill_volume(wall: WallLifecycle) -> float:
    total = 0.0
    for observation in wall.observations:
        if observation.event != "refilled":
            continue
        if observation.previous_size is None:
            continue
        total += max(
            0.0,
            observation.size - observation.previous_size,
        )
    return total


class ManipulationEvidenceAnalyzer:
    """Build cautious spoofing, iceberg, and absorption reports."""

    def __init__(
        self,
        config: Optional[EvidenceConfig] = None,
    ) -> None:
        self.config = config or EvidenceConfig()

    def spoofing_report(
        self,
        wall: WallLifecycle,
        *,
        touch_price: float,
        repetition_count: int = 0,
    ) -> EvidenceReport:
        if touch_price <= 0:
            raise ValueError("touch_price must be positive")
        if repetition_count < 0:
            raise ValueError(
                "repetition_count cannot be negative"
            )

        cfg = self.config
        reference_size = max(
            wall.initial_size,
            wall.peak_size,
            cfg.epsilon,
        )

        lifetime_ms = float(wall.duration_ms)
        distance_bps = (
            abs(wall.initial_price - touch_price)
            / touch_price
            * 10_000.0
        )

        pulled_volume = _pulled_volume(wall)
        executed_volume = _execution_volume(wall)

        cancellation_ratio = _clamp(
            pulled_volume / reference_size
        )
        execution_ratio = _clamp(
            executed_volume / reference_size
        )

        lifetime_score = 1.0 - _clamp(
            lifetime_ms / cfg.spoofing_max_lifetime_ms
        )
        distance_score = _clamp(
            distance_bps
            / cfg.spoofing_distance_reference_bps
        )
        cancellation_score = cancellation_ratio
        execution_absence_score = 1.0 - execution_ratio
        repetition_score = _clamp(
            repetition_count
            / cfg.spoofing_repetition_reference
        )

        # All five required dimensions influence the score.
        score = _clamp(
            0.20 * lifetime_score
            + 0.15 * distance_score
            + 0.30 * cancellation_score
            + 0.20 * execution_absence_score
            + 0.15 * repetition_score
        )

        evidence: list[str] = []
        contradictions: list[str] = []

        if lifetime_ms <= cfg.spoofing_max_lifetime_ms:
            evidence.append(
                f"short_lifetime_ms={lifetime_ms:.8g}"
            )
        else:
            contradictions.append(
                f"long_lifetime_ms={lifetime_ms:.8g}"
            )

        if distance_score >= 0.5:
            evidence.append(
                f"distance_from_touch_bps={distance_bps:.8g}"
            )
        else:
            contradictions.append(
                f"near_touch_distance_bps={distance_bps:.8g}"
            )

        if (
            cancellation_ratio
            >= cfg.spoofing_min_cancellation_ratio
        ):
            evidence.append(
                "high_cancellation_ratio="
                f"{cancellation_ratio:.8g}"
            )
        else:
            contradictions.append(
                "low_cancellation_ratio="
                f"{cancellation_ratio:.8g}"
            )

        if (
            execution_ratio
            <= cfg.spoofing_execution_contradiction_ratio
        ):
            evidence.append(
                f"limited_execution_ratio={execution_ratio:.8g}"
            )
        else:
            contradictions.append(
                "substantial_execution_ratio="
                f"{execution_ratio:.8g}"
            )

        if repetition_count >= cfg.spoofing_min_repetitions:
            evidence.append(
                f"repeated_pattern_count={repetition_count}"
            )
        else:
            contradictions.append(
                f"limited_repetition_count={repetition_count}"
            )

        cancellation_observed = any(
            observation.event == "cancelled"
            for observation in wall.observations
        )

        supported = (
            cancellation_observed
            and cancellation_ratio
            >= cfg.spoofing_min_cancellation_ratio
            and score >= cfg.spoofing_min_score
        )

        return EvidenceReport(
            hypothesis="spoofing_like",
            supported=supported,
            score=score,
            confidence=_confidence(
                score,
                evidence,
                contradictions,
            ),
            wording=_wording(
                "spoofing-like",
                supported,
            ),
            exchange=wall.exchange,
            symbol=wall.symbol,
            market_type=wall.market_type,
            side=wall.side,
            start_ts_ms=wall.first_seen_ms,
            end_ts_ms=(
                wall.closed_at_ms
                if wall.closed_at_ms is not None
                else wall.last_seen_ms
            ),
            evidence=tuple(evidence),
            contradictions=tuple(contradictions),
            metrics=(
                ("lifetime_ms", lifetime_ms),
                ("distance_bps", distance_bps),
                (
                    "cancellation_ratio",
                    cancellation_ratio,
                ),
                ("execution_ratio", execution_ratio),
                (
                    "repetition_count",
                    float(repetition_count),
                ),
            ),
        )

    def iceberg_report(
        self,
        wall: WallLifecycle,
    ) -> EvidenceReport:
        cfg = self.config
        visible_size = max(
            wall.initial_size,
            cfg.epsilon,
        )

        refill_count = sum(
            1
            for observation in wall.observations
            if observation.event == "refilled"
        )
        refill_volume = _refill_volume(wall)
        executed_volume = _execution_volume(wall)

        executed_to_visible = (
            executed_volume / visible_size
        )
        refill_to_visible = refill_volume / visible_size

        refill_count_score = _clamp(
            refill_count / cfg.iceberg_min_refill_count
        )
        executed_score = _clamp(
            executed_to_visible
            / cfg.iceberg_min_executed_to_visible
        )
        refill_volume_score = _clamp(
            refill_to_visible
            / cfg.iceberg_refill_reference_ratio
        )

        score = _clamp(
            0.35 * refill_count_score
            + 0.40 * executed_score
            + 0.25 * refill_volume_score
        )

        evidence: list[str] = []
        contradictions: list[str] = []

        if refill_count >= cfg.iceberg_min_refill_count:
            evidence.append(
                f"repeated_refill_count={refill_count}"
            )
        else:
            contradictions.append(
                f"insufficient_refill_count={refill_count}"
            )

        if (
            executed_to_visible
            >= cfg.iceberg_min_executed_to_visible
        ):
            evidence.append(
                "executed_to_visible_ratio="
                f"{executed_to_visible:.8g}"
            )
        else:
            contradictions.append(
                "low_executed_to_visible_ratio="
                f"{executed_to_visible:.8g}"
            )

        if refill_volume > cfg.epsilon:
            evidence.append(
                f"visible_refill_volume={refill_volume:.8g}"
            )
        else:
            contradictions.append(
                "no_visible_refill_observed"
            )

        supported = (
            refill_count >= cfg.iceberg_min_refill_count
            and executed_to_visible
            >= cfg.iceberg_min_executed_to_visible
            and score >= cfg.iceberg_min_score
        )

        return EvidenceReport(
            hypothesis="iceberg_like",
            supported=supported,
            score=score,
            confidence=_confidence(
                score,
                evidence,
                contradictions,
            ),
            wording=_wording(
                "iceberg-like",
                supported,
            ),
            exchange=wall.exchange,
            symbol=wall.symbol,
            market_type=wall.market_type,
            side=wall.side,
            start_ts_ms=wall.first_seen_ms,
            end_ts_ms=(
                wall.closed_at_ms
                if wall.closed_at_ms is not None
                else wall.last_seen_ms
            ),
            evidence=tuple(evidence),
            contradictions=tuple(contradictions),
            metrics=(
                ("refill_count", float(refill_count)),
                ("refill_volume", refill_volume),
                ("executed_volume", executed_volume),
                (
                    "executed_to_visible_ratio",
                    executed_to_visible,
                ),
                (
                    "refill_to_visible_ratio",
                    refill_to_visible,
                ),
            ),
        )

    def absorption_report(
        self,
        window: AbsorptionWindow,
    ) -> EvidenceReport:
        cfg = self.config

        flow_to_depth = (
            window.aggressive_volume
            / window.reference_visible_depth
        )
        price_response_bps = (
            abs(window.end_price - window.start_price)
            / window.start_price
            * 10_000.0
        )

        absolute_flow_ok = (
            window.aggressive_volume
            >= cfg.absorption_min_aggressive_volume
        )
        relative_flow_ok = (
            flow_to_depth
            >= cfg.absorption_min_flow_to_depth_ratio
        )
        weak_response_ok = (
            price_response_bps
            <= cfg.absorption_max_price_response_bps
        )

        flow_score = _clamp(
            flow_to_depth
            / cfg.absorption_min_flow_to_depth_ratio
        )
        weak_response_score = (
            1.0
            - _clamp(
                price_response_bps
                / cfg.absorption_max_price_response_bps
            )
        )

        score = _clamp(
            0.60 * flow_score
            + 0.40 * weak_response_score
        )

        evidence: list[str] = []
        contradictions: list[str] = []

        if absolute_flow_ok and relative_flow_ok:
            evidence.append(
                "aggressive_flow_to_depth_ratio="
                f"{flow_to_depth:.8g}"
            )
        else:
            contradictions.append(
                "insufficient_aggressive_flow="
                f"{window.aggressive_volume:.8g}"
            )

        if weak_response_ok:
            evidence.append(
                "weak_price_response_bps="
                f"{price_response_bps:.8g}"
            )
        else:
            contradictions.append(
                "strong_price_response_bps="
                f"{price_response_bps:.8g}"
            )

        # Absorption cannot be supported unless both conditions hold.
        supported = (
            absolute_flow_ok
            and relative_flow_ok
            and weak_response_ok
        )

        return EvidenceReport(
            hypothesis="absorption_like",
            supported=supported,
            score=score,
            confidence=_confidence(
                score,
                evidence,
                contradictions,
            ),
            wording=_wording(
                "absorption-like",
                supported,
            ),
            exchange=window.exchange,
            symbol=window.symbol.upper(),
            market_type=window.market_type,
            side=window.aggressive_side,
            start_ts_ms=window.start_ts_ms,
            end_ts_ms=window.end_ts_ms,
            evidence=tuple(evidence),
            contradictions=tuple(contradictions),
            metrics=(
                (
                    "aggressive_volume",
                    window.aggressive_volume,
                ),
                (
                    "reference_visible_depth",
                    window.reference_visible_depth,
                ),
                (
                    "flow_to_depth_ratio",
                    flow_to_depth,
                ),
                (
                    "price_response_bps",
                    price_response_bps,
                ),
            ),
        )

    def wall_reports(
        self,
        wall: WallLifecycle,
        *,
        touch_price: float,
        repetition_count: int = 0,
    ) -> Tuple[EvidenceReport, EvidenceReport]:
        return (
            self.spoofing_report(
                wall,
                touch_price=touch_price,
                repetition_count=repetition_count,
            ),
            self.iceberg_report(wall),
        )


__all__ = [
    "AbsorptionWindow",
    "EvidenceConfig",
    "EvidenceReport",
    "ManipulationEvidenceAnalyzer",
]
