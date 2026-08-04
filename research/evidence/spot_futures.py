from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from research.derivatives.context import (
    DerivativesContextWindow,
    OIPriceInterpretation,
    interpret_oi_price,
)


Classification = Literal[
    "organic_spot_demand",
    "short_squeeze",
    "futures_led_pump",
    "late_long_buildup",
    "mixed_or_uncertain",
]

MovementLeader = Literal[
    "spot",
    "futures",
    "balanced",
]


@dataclass(frozen=True)
class SpotFuturesEvidenceConfig:
    min_price_move_bps: float = 30.0
    lead_margin_bps: float = 15.0
    min_flow_imbalance: float = 0.15

    oi_rise_pct: float = 0.05
    oi_fall_pct: float = 0.05

    basis_expansion_bps: float = 5.0
    high_funding_rate: float = 0.0005

    min_liquidation_volume: float = 10.0
    short_liquidation_ratio: float = 3.0

    prior_move_bps: float = 50.0
    min_score: float = 0.60

    missing_oi_penalty: float = 0.15
    missing_liquidation_penalty: float = 0.15
    missing_funding_penalty: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            "min_price_move_bps",
            "lead_margin_bps",
            "min_flow_imbalance",
            "oi_rise_pct",
            "oi_fall_pct",
            "basis_expansion_bps",
            "high_funding_rate",
            "min_liquidation_volume",
            "short_liquidation_ratio",
            "prior_move_bps",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

        unit_interval = (
            "min_score",
            "missing_oi_penalty",
            "missing_liquidation_penalty",
            "missing_funding_penalty",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )


@dataclass(frozen=True)
class SpotFuturesEvidenceReport:
    classification: Classification
    supported: bool
    score: float
    confidence: float
    movement_leader: MovementLeader

    exchange: str
    symbol: str
    start_ts_ms: int
    end_ts_ms: int

    summary: str
    evidence: Tuple[str, ...]
    contradictions: Tuple[str, ...]
    missing_data: Tuple[str, ...]

    oi_price_interpretation: OIPriceInterpretation
    counterexamples: Tuple[str, ...]
    metrics: Tuple[Tuple[str, float], ...]

    def metric(self, name: str) -> Optional[float]:
        for metric_name, value in self.metrics:
            if metric_name == name:
                return value
        return None


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ratio_score(
    value: float,
    threshold: float,
) -> float:
    return _clamp(value / threshold)


def _unique(values: list[str]) -> Tuple[str, ...]:
    return tuple(dict.fromkeys(values))


class SpotFuturesEvidenceAnalyzer:
    """Classify spot/futures leadership with derivatives context."""

    def __init__(
        self,
        config: Optional[SpotFuturesEvidenceConfig] = None,
    ) -> None:
        self.config = config or SpotFuturesEvidenceConfig()

    def _movement_leader(
        self,
        window: DerivativesContextWindow,
    ) -> MovementLeader:
        difference = (
            window.spot_return_bps
            - window.futures_return_bps
        )
        if abs(difference) < self.config.lead_margin_bps:
            return "balanced"
        return "spot" if difference > 0 else "futures"

    def _missing_data(
        self,
        window: DerivativesContextWindow,
    ) -> list[str]:
        missing: list[str] = []

        if window.open_interest_change_pct is None:
            missing.append("open_interest")
        if not window.has_liquidations:
            missing.append("liquidations")
        if window.funding_rate is None:
            missing.append("funding")

        return missing

    def _confidence(
        self,
        score: float,
        missing_data: list[str],
    ) -> float:
        penalty = 0.0
        if "open_interest" in missing_data:
            penalty += self.config.missing_oi_penalty
        if "liquidations" in missing_data:
            penalty += (
                self.config.missing_liquidation_penalty
            )
        if "funding" in missing_data:
            penalty += self.config.missing_funding_penalty

        return _clamp(score * max(0.0, 1.0 - penalty))

    def analyze(
        self,
        window: DerivativesContextWindow,
        *,
        prior: Optional[DerivativesContextWindow] = None,
    ) -> SpotFuturesEvidenceReport:
        if prior is not None:
            if prior.exchange != window.exchange:
                raise ValueError(
                    "prior exchange does not match current window"
                )
            if prior.symbol != window.symbol:
                raise ValueError(
                    "prior symbol does not match current window"
                )
            if prior.end_ts_ms > window.start_ts_ms:
                raise ValueError(
                    "prior window must not overlap current window"
                )

        cfg = self.config
        leader = self._movement_leader(window)
        missing_data = self._missing_data(window)

        oi_change = window.open_interest_change_pct
        short_ratio = window.short_liquidation_ratio

        short_liquidations = (
            window.short_liquidation_volume
            if window.short_liquidation_volume is not None
            else 0.0
        )

        price_rise = max(
            window.spot_return_bps,
            window.futures_return_bps,
        )
        spot_lead_bps = (
            window.spot_return_bps
            - window.futures_return_bps
        )
        futures_lead_bps = -spot_lead_bps

        spot_flow = window.spot_flow_imbalance
        futures_flow = window.futures_flow_imbalance
        basis_change = window.basis_change_bps

        prior_rise = (
            max(
                prior.spot_return_bps,
                prior.futures_return_bps,
            )
            if prior is not None
            else 0.0
        )

        funding = (
            window.funding_rate
            if window.funding_rate is not None
            else 0.0
        )

        short_squeeze_score = _clamp(
            0.25
            * _ratio_score(
                max(price_rise, 0.0),
                cfg.min_price_move_bps,
            )
            + 0.30
            * (
                _ratio_score(
                    max(-(oi_change or 0.0), 0.0),
                    cfg.oi_fall_pct,
                )
            )
            + 0.30
            * min(
                _ratio_score(
                    short_liquidations,
                    cfg.min_liquidation_volume,
                ),
                _ratio_score(
                    short_ratio or 0.0,
                    cfg.short_liquidation_ratio,
                ),
            )
            + 0.15
            * _ratio_score(
                max(futures_flow, 0.0),
                cfg.min_flow_imbalance,
            )
        )

        late_long_score = _clamp(
            0.20
            * _ratio_score(
                max(prior_rise, 0.0),
                cfg.prior_move_bps,
            )
            + 0.25
            * _ratio_score(
                max(oi_change or 0.0, 0.0),
                cfg.oi_rise_pct,
            )
            + 0.20
            * _ratio_score(
                max(funding, 0.0),
                cfg.high_funding_rate,
            )
            + 0.15
            * _ratio_score(
                max(basis_change, 0.0),
                cfg.basis_expansion_bps,
            )
            + 0.10
            * _ratio_score(
                max(futures_lead_bps, 0.0),
                cfg.lead_margin_bps,
            )
            + 0.10
            * (
                1.0
                if (
                    window.spot_return_bps
                    < cfg.min_price_move_bps
                    or spot_flow < cfg.min_flow_imbalance
                )
                else 0.0
            )
        )

        leverage_not_expanding_score = (
            0.5
            if oi_change is None
            else (
                1.0
                if oi_change < cfg.oi_rise_pct
                else 0.0
            )
        )
        low_liquidation_score = (
            0.5
            if short_ratio is None
            else (
                1.0
                if short_ratio < cfg.short_liquidation_ratio
                else 0.0
            )
        )

        organic_spot_score = _clamp(
            0.35
            * _ratio_score(
                max(spot_lead_bps, 0.0),
                cfg.lead_margin_bps,
            )
            + 0.30
            * _ratio_score(
                max(spot_flow, 0.0),
                cfg.min_flow_imbalance,
            )
            + 0.20 * leverage_not_expanding_score
            + 0.10
            * (
                1.0
                if basis_change < cfg.basis_expansion_bps
                else 0.0
            )
            + 0.05 * low_liquidation_score
        )

        futures_led_score = _clamp(
            0.30
            * _ratio_score(
                max(futures_lead_bps, 0.0),
                cfg.lead_margin_bps,
            )
            + 0.25
            * _ratio_score(
                max(futures_flow, 0.0),
                cfg.min_flow_imbalance,
            )
            + 0.25
            * _ratio_score(
                max(oi_change or 0.0, 0.0),
                cfg.oi_rise_pct,
            )
            + 0.20
            * _ratio_score(
                max(basis_change, 0.0),
                cfg.basis_expansion_bps,
            )
        )

        short_squeeze_supported = (
            price_rise >= cfg.min_price_move_bps
            and oi_change is not None
            and oi_change <= -cfg.oi_fall_pct
            and short_ratio is not None
            and short_ratio >= cfg.short_liquidation_ratio
            and short_liquidations
            >= cfg.min_liquidation_volume
            and short_squeeze_score >= cfg.min_score
        )

        late_long_supported = (
            prior is not None
            and prior_rise >= cfg.prior_move_bps
            and oi_change is not None
            and oi_change >= cfg.oi_rise_pct
            and window.funding_rate is not None
            and window.funding_rate
            >= cfg.high_funding_rate
            and basis_change >= cfg.basis_expansion_bps
            and leader == "futures"
            and (
                window.spot_return_bps
                < cfg.min_price_move_bps
                or spot_flow < cfg.min_flow_imbalance
            )
            and late_long_score >= cfg.min_score
        )

        organic_spot_supported = (
            leader == "spot"
            and window.spot_return_bps
            >= cfg.min_price_move_bps
            and spot_flow >= cfg.min_flow_imbalance
            and organic_spot_score >= cfg.min_score
        )

        futures_led_supported = (
            leader == "futures"
            and window.futures_return_bps
            >= cfg.min_price_move_bps
            and futures_flow >= cfg.min_flow_imbalance
            and oi_change is not None
            and oi_change >= cfg.oi_rise_pct
            and basis_change >= cfg.basis_expansion_bps
            and futures_led_score >= cfg.min_score
        )

        if short_squeeze_supported:
            classification: Classification = "short_squeeze"
            score = short_squeeze_score
        elif late_long_supported:
            classification = "late_long_buildup"
            score = late_long_score
        elif organic_spot_supported:
            classification = "organic_spot_demand"
            score = organic_spot_score
        elif futures_led_supported:
            classification = "futures_led_pump"
            score = futures_led_score
        else:
            classification = "mixed_or_uncertain"
            score = max(
                short_squeeze_score,
                late_long_score,
                organic_spot_score,
                futures_led_score,
            )

        evidence: list[str] = []
        contradictions: list[str] = []

        if classification == "short_squeeze":
            evidence.extend(
                [
                    f"price_rise_bps={price_rise:.8g}",
                    f"oi_change_pct={oi_change:.8g}",
                    (
                        "short_liquidation_ratio="
                        f"{short_ratio:.8g}"
                    ),
                    (
                        "short_liquidation_volume="
                        f"{short_liquidations:.8g}"
                    ),
                    (
                        "futures_flow_imbalance="
                        f"{futures_flow:.8g}"
                    ),
                ]
            )
            if leader == "spot":
                contradictions.append(
                    "spot_moved_more_than_futures"
                )

        elif classification == "late_long_buildup":
            evidence.extend(
                [
                    f"prior_price_rise_bps={prior_rise:.8g}",
                    f"oi_increase_pct={oi_change:.8g}",
                    f"elevated_funding_rate={funding:.8g}",
                    f"basis_expansion_bps={basis_change:.8g}",
                    f"futures_lead_bps={futures_lead_bps:.8g}",
                    "weak_spot_confirmation",
                ]
            )

        elif classification == "organic_spot_demand":
            evidence.extend(
                [
                    f"spot_lead_bps={spot_lead_bps:.8g}",
                    f"spot_flow_imbalance={spot_flow:.8g}",
                ]
            )
            if oi_change is None:
                contradictions.append(
                    "open_interest_confirmation_unavailable"
                )
            elif oi_change < cfg.oi_rise_pct:
                evidence.append(
                    "leverage_not_expanding_materially"
                )
            else:
                contradictions.append(
                    f"open_interest_rising_pct={oi_change:.8g}"
                )

            if basis_change < cfg.basis_expansion_bps:
                evidence.append(
                    "basis_not_expanding_materially"
                )
            else:
                contradictions.append(
                    f"basis_expansion_bps={basis_change:.8g}"
                )

        elif classification == "futures_led_pump":
            evidence.extend(
                [
                    f"futures_lead_bps={futures_lead_bps:.8g}",
                    (
                        "futures_flow_imbalance="
                        f"{futures_flow:.8g}"
                    ),
                    f"oi_increase_pct={oi_change:.8g}",
                    f"basis_expansion_bps={basis_change:.8g}",
                ]
            )
            if spot_flow >= cfg.min_flow_imbalance:
                contradictions.append(
                    "spot_flow_also_supportive"
                )

        else:
            contradictions.extend(
                [
                    "no_single_scenario_met_all_required_conditions",
                    f"movement_leader={leader}",
                ]
            )

        oi_interpretation = interpret_oi_price(
            window.futures_return_bps,
            oi_change,
        )

        classification_counterexamples = {
            "organic_spot_demand": (
                "Cross-venue arbitrage or delayed futures repricing "
                "can make a move appear spot-led.",
            ),
            "short_squeeze": (
                "Venue migration or voluntary position closure can "
                "reduce OI without forced short covering.",
            ),
            "futures_led_pump": (
                "Basis trades and hedging can increase futures flow "
                "and OI without directional speculation.",
            ),
            "late_long_buildup": (
                "Market-making inventory and hedging can raise OI "
                "and funding without one-sided late-long demand.",
            ),
            "mixed_or_uncertain": (
                "Different venues or sampling windows may show "
                "different leadership.",
            ),
        }

        counterexamples = _unique(
            list(oi_interpretation.counterexamples)
            + list(
                classification_counterexamples[
                    classification
                ]
            )
        )

        supported = (
            classification != "mixed_or_uncertain"
            and score >= cfg.min_score
        )

        summary = (
            "Evidence is compatible with "
            f"{classification.replace('_', ' ')}. "
            "This contextual classification does not establish "
            "a unique cause or participant intent."
        )

        return SpotFuturesEvidenceReport(
            classification=classification,
            supported=supported,
            score=score,
            confidence=self._confidence(
                score,
                missing_data,
            ),
            movement_leader=leader,
            exchange=window.exchange,
            symbol=window.symbol,
            start_ts_ms=window.start_ts_ms,
            end_ts_ms=window.end_ts_ms,
            summary=summary,
            evidence=tuple(evidence),
            contradictions=tuple(contradictions),
            missing_data=tuple(missing_data),
            oi_price_interpretation=oi_interpretation,
            counterexamples=counterexamples,
            metrics=(
                (
                    "spot_return_bps",
                    window.spot_return_bps,
                ),
                (
                    "futures_return_bps",
                    window.futures_return_bps,
                ),
                (
                    "spot_flow_imbalance",
                    spot_flow,
                ),
                (
                    "futures_flow_imbalance",
                    futures_flow,
                ),
                (
                    "basis_change_bps",
                    basis_change,
                ),
                (
                    "open_interest_change_pct",
                    oi_change
                    if oi_change is not None
                    else float("nan"),
                ),
                (
                    "short_liquidation_ratio",
                    short_ratio
                    if short_ratio is not None
                    else float("nan"),
                ),
            ),
        )


__all__ = [
    "SpotFuturesEvidenceAnalyzer",
    "SpotFuturesEvidenceConfig",
    "SpotFuturesEvidenceReport",
]
