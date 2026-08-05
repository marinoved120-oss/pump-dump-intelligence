from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


HypothesisCategory = Literal[
    "market_phase",
    "pump_cause",
    "dump_mechanism",
    "alternative",
]
EvidenceDirection = Literal["support", "contradict"]
AssessmentStatus = Literal["rejected", "weakened", "supported"]
AlertLevel = Literal["none", "yellow", "orange", "red"]


@dataclass(frozen=True)
class CausalEngineConfig:
    support_threshold: float = 0.60
    reject_below_score: float = 0.25
    orange_score: float = 0.65
    red_score: float = 0.80

    support_reference_confirmations: int = 3
    confidence_group_reference: int = 3

    min_independent_groups_for_red: int = 2
    min_independent_confirmations_for_red: int = 2

    contradiction_penalty: float = 0.50
    missing_data_penalty: float = 0.12
    invalidation_strength: float = 0.70

    def __post_init__(self) -> None:
        integer_fields = (
            "support_reference_confirmations",
            "confidence_group_reference",
            "min_independent_groups_for_red",
            "min_independent_confirmations_for_red",
        )
        for name in integer_fields:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")

        unit_interval = (
            "support_threshold",
            "reject_below_score",
            "orange_score",
            "red_score",
            "contradiction_penalty",
            "missing_data_penalty",
            "invalidation_strength",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if self.reject_below_score > self.support_threshold:
            raise ValueError(
                "reject_below_score cannot exceed support_threshold"
            )
        if self.orange_score > self.red_score:
            raise ValueError(
                "orange_score cannot exceed red_score"
            )


@dataclass(frozen=True)
class EvidenceItem:
    """One observation assigned to an evidence and correlation group."""

    evidence_id: str
    group: str
    correlation_key: str
    direction: EvidenceDirection
    strength: float
    statement: str
    source: str = ""

    def __post_init__(self) -> None:
        required = (
            "evidence_id",
            "group",
            "correlation_key",
            "statement",
        )
        for name in required:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(
                "strength must be between 0 and 1"
            )


@dataclass(frozen=True)
class InvalidationRule:
    rule_id: str
    description: str
    trigger_evidence_ids: Tuple[str, ...]
    require_all: bool = False

    def __post_init__(self) -> None:
        if not self.rule_id.strip():
            raise ValueError("rule_id cannot be empty")
        if not self.description.strip():
            raise ValueError("description cannot be empty")
        if not self.trigger_evidence_ids:
            raise ValueError(
                "trigger_evidence_ids cannot be empty"
            )


@dataclass(frozen=True)
class HypothesisSpec:
    hypothesis_id: str
    category: HypothesisCategory
    label: str
    causal_claim: str

    supporting_evidence_ids: Tuple[str, ...]
    contradicting_evidence_ids: Tuple[str, ...]

    required_data: Tuple[str, ...] = ()
    invalidation_rules: Tuple[InvalidationRule, ...] = ()
    alternative_scenarios: Tuple[str, ...] = ()

    min_independent_groups: int = 2
    min_independent_confirmations: int = 2

    def __post_init__(self) -> None:
        required = (
            "hypothesis_id",
            "label",
            "causal_claim",
        )
        for name in required:
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")

        if self.min_independent_groups < 1:
            raise ValueError(
                "min_independent_groups must be positive"
            )
        if self.min_independent_confirmations < 1:
            raise ValueError(
                "min_independent_confirmations must be positive"
            )

        overlap = set(self.supporting_evidence_ids).intersection(
            self.contradicting_evidence_ids
        )
        if overlap:
            raise ValueError(
                "Evidence cannot simultaneously support and "
                f"contradict: {sorted(overlap)}"
            )


@dataclass(frozen=True)
class CausalAssessment:
    hypothesis_id: str
    category: HypothesisCategory
    label: str

    status: AssessmentStatus
    alert_level: AlertLevel
    score: float
    confidence: float
    causal_claim_allowed: bool

    statement: str

    supporting_evidence: Tuple[EvidenceItem, ...]
    contradicting_evidence: Tuple[EvidenceItem, ...]

    independent_supporting_groups: Tuple[str, ...]
    independent_confirmation_keys: Tuple[str, ...]

    missing_data: Tuple[str, ...]
    invalidation_rules: Tuple[str, ...]
    triggered_invalidations: Tuple[str, ...]
    alternative_scenarios: Tuple[str, ...]


@dataclass(frozen=True)
class CausalAnalysisBundle:
    assessments: Tuple[CausalAssessment, ...]

    @property
    def market_phases(self) -> Tuple[CausalAssessment, ...]:
        return tuple(
            item
            for item in self.assessments
            if item.category == "market_phase"
        )

    @property
    def pump_causes(self) -> Tuple[CausalAssessment, ...]:
        return tuple(
            item
            for item in self.assessments
            if item.category == "pump_cause"
        )

    @property
    def dump_mechanisms(self) -> Tuple[CausalAssessment, ...]:
        return tuple(
            item
            for item in self.assessments
            if item.category == "dump_mechanism"
        )

    @property
    def alternatives(self) -> Tuple[CausalAssessment, ...]:
        return tuple(
            item
            for item in self.assessments
            if item.category == "alternative"
        )

    @property
    def global_alert_level(self) -> AlertLevel:
        rank = {
            "none": 0,
            "yellow": 1,
            "orange": 2,
            "red": 3,
        }
        if not self.assessments:
            return "none"
        return max(
            (
                assessment.alert_level
                for assessment in self.assessments
            ),
            key=rank.__getitem__,
        )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _deduplicate_correlated(
    items: Tuple[EvidenceItem, ...],
) -> Tuple[EvidenceItem, ...]:
    """Keep the strongest item from each correlation family."""

    strongest: dict[str, EvidenceItem] = {}

    for item in items:
        previous = strongest.get(item.correlation_key)
        if previous is None or item.strength > previous.strength:
            strongest[item.correlation_key] = item

    return tuple(
        strongest[key]
        for key in sorted(strongest)
    )


class CausalEvidenceEngine:
    """Assess causal hypotheses using independent evidence groups."""

    def __init__(
        self,
        config: Optional[CausalEngineConfig] = None,
    ) -> None:
        self.config = config or CausalEngineConfig()

    def _validate_items(
        self,
        evidence: Tuple[EvidenceItem, ...],
    ) -> dict[str, EvidenceItem]:
        indexed: dict[str, EvidenceItem] = {}

        for item in evidence:
            if item.evidence_id in indexed:
                raise ValueError(
                    "Duplicate evidence_id: "
                    f"{item.evidence_id}"
                )
            indexed[item.evidence_id] = item

        return indexed

    def _triggered_invalidations(
        self,
        spec: HypothesisSpec,
        evidence_by_id: dict[str, EvidenceItem],
    ) -> Tuple[str, ...]:
        triggered: list[str] = []

        for rule in spec.invalidation_rules:
            matches = [
                evidence_by_id[evidence_id]
                for evidence_id in rule.trigger_evidence_ids
                if evidence_id in evidence_by_id
                and evidence_by_id[evidence_id].strength
                >= self.config.invalidation_strength
            ]

            if rule.require_all:
                active = len(matches) == len(
                    rule.trigger_evidence_ids
                )
            else:
                active = bool(matches)

            if active:
                triggered.append(rule.description)

        return tuple(triggered)

    def assess(
        self,
        spec: HypothesisSpec,
        evidence: Tuple[EvidenceItem, ...],
        *,
        available_data: Tuple[str, ...] = (),
    ) -> CausalAssessment:
        evidence_by_id = self._validate_items(evidence)

        supporting = tuple(
            evidence_by_id[evidence_id]
            for evidence_id in spec.supporting_evidence_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].direction == "support"
        )
        contradicting = tuple(
            evidence_by_id[evidence_id]
            for evidence_id in spec.contradicting_evidence_ids
            if evidence_id in evidence_by_id
            and evidence_by_id[evidence_id].direction
            == "contradict"
        )

        independent_support = _deduplicate_correlated(
            supporting
        )
        independent_contradictions = _deduplicate_correlated(
            contradicting
        )

        independent_groups = tuple(
            sorted(
                {
                    item.group
                    for item in independent_support
                }
            )
        )
        confirmation_keys = tuple(
            sorted(
                item.correlation_key
                for item in independent_support
            )
        )

        missing_data = tuple(
            sorted(
                set(spec.required_data)
                - set(available_data)
            )
        )

        triggered_invalidations = (
            self._triggered_invalidations(
                spec,
                evidence_by_id,
            )
        )

        if independent_support:
            mean_support = sum(
                item.strength
                for item in independent_support
            ) / len(independent_support)

            coverage = min(
                1.0,
                len(independent_support)
                / self.config.support_reference_confirmations,
            )

            support_score = mean_support * (
                0.5 + 0.5 * coverage
            )
        else:
            support_score = 0.0

        if independent_contradictions:
            contradiction_strength = sum(
                item.strength
                for item in independent_contradictions
            ) / len(independent_contradictions)
        else:
            contradiction_strength = 0.0

        score = _clamp(
            support_score
            * (
                1.0
                - self.config.contradiction_penalty
                * contradiction_strength
            )
        )

        group_diversity = min(
            1.0,
            len(independent_groups)
            / self.config.confidence_group_reference,
        )
        diversity_factor = 0.5 + 0.5 * group_diversity

        missing_penalty = min(
            0.80,
            len(missing_data)
            * self.config.missing_data_penalty,
        )

        confidence = _clamp(
            score
            * diversity_factor
            * (1.0 - missing_penalty)
        )

        enough_groups = (
            len(independent_groups)
            >= spec.min_independent_groups
        )
        enough_confirmations = (
            len(confirmation_keys)
            >= spec.min_independent_confirmations
        )

        if triggered_invalidations:
            status: AssessmentStatus = "rejected"
        elif not independent_support:
            status = "rejected"
        elif score < self.config.reject_below_score:
            status = "rejected"
        elif (
            not enough_groups
            or not enough_confirmations
            or score < self.config.support_threshold
        ):
            status = "weakened"
        else:
            status = "supported"

        causal_claim_allowed = status == "supported"

        if status == "rejected":
            alert_level: AlertLevel = "none"
            statement = (
                f"The causal claim '{spec.causal_claim}' is rejected "
                "for the current evidence set. Supporting evidence "
                "is absent, materially contradicted, or invalidated."
            )
        elif status == "weakened":
            alert_level = "yellow"
            statement = (
                f"Observed evidence is associated with '{spec.label}', "
                "but independent support is insufficient for a causal "
                "claim. Treat this only as a weakened hypothesis."
            )
        else:
            red_allowed = (
                score >= self.config.red_score
                and len(independent_groups)
                >= self.config.min_independent_groups_for_red
                and len(confirmation_keys)
                >= self.config.min_independent_confirmations_for_red
            )

            if red_allowed:
                alert_level = "red"
            elif score >= self.config.orange_score:
                alert_level = "orange"
            else:
                alert_level = "yellow"

            statement = (
                f"Evidence from {len(independent_groups)} independent "
                f"groups is compatible with '{spec.causal_claim}'. "
                "This remains a causal hypothesis, not proof."
            )

        return CausalAssessment(
            hypothesis_id=spec.hypothesis_id,
            category=spec.category,
            label=spec.label,
            status=status,
            alert_level=alert_level,
            score=score,
            confidence=confidence,
            causal_claim_allowed=causal_claim_allowed,
            statement=statement,
            supporting_evidence=supporting,
            contradicting_evidence=contradicting,
            independent_supporting_groups=independent_groups,
            independent_confirmation_keys=confirmation_keys,
            missing_data=missing_data,
            invalidation_rules=tuple(
                rule.description
                for rule in spec.invalidation_rules
            ),
            triggered_invalidations=triggered_invalidations,
            alternative_scenarios=spec.alternative_scenarios,
        )

    def assess_many(
        self,
        specs: Tuple[HypothesisSpec, ...],
        evidence: Tuple[EvidenceItem, ...],
        *,
        available_data: Tuple[str, ...] = (),
    ) -> CausalAnalysisBundle:
        return CausalAnalysisBundle(
            assessments=tuple(
                self.assess(
                    spec,
                    evidence,
                    available_data=available_data,
                )
                for spec in specs
            )
        )


__all__ = [
    "AlertLevel",
    "AssessmentStatus",
    "CausalAnalysisBundle",
    "CausalAssessment",
    "CausalEngineConfig",
    "CausalEvidenceEngine",
    "EvidenceItem",
    "HypothesisCategory",
    "HypothesisSpec",
    "InvalidationRule",
]
