import pytest

from research.evidence.causal import (
    CausalEvidenceEngine,
    EvidenceItem,
    HypothesisSpec,
    InvalidationRule,
)


def support(
    evidence_id,
    group,
    correlation_key,
    strength=1.0,
):
    return EvidenceItem(
        evidence_id=evidence_id,
        group=group,
        correlation_key=correlation_key,
        direction="support",
        strength=strength,
        statement=f"supporting observation: {evidence_id}",
        source="test",
    )


def contradict(
    evidence_id,
    group,
    correlation_key,
    strength=1.0,
):
    return EvidenceItem(
        evidence_id=evidence_id,
        group=group,
        correlation_key=correlation_key,
        direction="contradict",
        strength=strength,
        statement=f"contradicting observation: {evidence_id}",
        source="test",
    )


def spec(**overrides):
    values = {
        "hypothesis_id": "pump-cause-1",
        "category": "pump_cause",
        "label": "futures-led leverage expansion",
        "causal_claim": (
            "futures leverage expansion contributed to the pump"
        ),
        "supporting_evidence_ids": (
            "oi_growth",
            "spot_buying",
        ),
        "contradicting_evidence_ids": (
            "spot_reversal",
        ),
        "required_data": (
            "open_interest",
            "spot_flow",
        ),
        "invalidation_rules": (),
        "alternative_scenarios": (
            "organic spot demand",
            "cross-venue repricing",
        ),
    }
    values.update(overrides)
    return HypothesisSpec(**values)


def test_correlated_features_count_as_one_confirmation():
    engine = CausalEvidenceEngine()

    hypothesis = spec(
        supporting_evidence_ids=(
            "oi_growth",
            "funding_rise",
            "basis_expansion",
        ),
        contradicting_evidence_ids=(),
        required_data=(),
    )

    evidence = (
        support(
            "oi_growth",
            "derivatives",
            "leverage_expansion",
        ),
        support(
            "funding_rise",
            "derivatives",
            "leverage_expansion",
        ),
        support(
            "basis_expansion",
            "derivatives",
            "leverage_expansion",
        ),
    )

    result = engine.assess(hypothesis, evidence)

    assert len(result.supporting_evidence) == 3
    assert result.independent_supporting_groups == (
        "derivatives",
    )
    assert result.independent_confirmation_keys == (
        "leverage_expansion",
    )

    assert result.status == "weakened"
    assert result.alert_level != "red"
    assert not result.causal_claim_allowed


def test_red_alert_requires_multiple_independent_groups():
    engine = CausalEvidenceEngine()

    evidence = (
        support(
            "oi_growth",
            "derivatives",
            "leverage_expansion",
        ),
        support(
            "spot_buying",
            "spot_flow",
            "spot_aggressive_demand",
        ),
    )

    result = engine.assess(
        spec(contradicting_evidence_ids=()),
        evidence,
        available_data=(
            "open_interest",
            "spot_flow",
        ),
    )

    assert result.status == "supported"
    assert result.alert_level == "red"
    assert result.causal_claim_allowed
    assert set(result.independent_supporting_groups) == {
        "derivatives",
        "spot_flow",
    }


def test_assessment_lists_evidence_missing_data_and_rules():
    engine = CausalEvidenceEngine()

    rule = InvalidationRule(
        rule_id="invalidate-on-reversal",
        description=(
            "Invalidate when price fully reverses "
            "without continued flow."
        ),
        trigger_evidence_ids=("full_reversal",),
    )

    hypothesis = spec(
        invalidation_rules=(rule,),
        required_data=(
            "open_interest",
            "liquidations",
        ),
    )

    evidence = (
        support(
            "oi_growth",
            "derivatives",
            "leverage_expansion",
            0.9,
        ),
        support(
            "spot_buying",
            "spot_flow",
            "spot_aggressive_demand",
            0.9,
        ),
        contradict(
            "spot_reversal",
            "price_action",
            "price_reversal",
            0.4,
        ),
    )

    result = engine.assess(
        hypothesis,
        evidence,
        available_data=("open_interest",),
    )

    assert result.supporting_evidence
    assert result.contradicting_evidence
    assert result.missing_data == ("liquidations",)
    assert result.invalidation_rules == (
        rule.description,
    )
    assert result.alternative_scenarios == (
        "organic spot demand",
        "cross-venue repricing",
    )


def test_unsupported_causal_claim_is_weakened():
    engine = CausalEvidenceEngine()

    result = engine.assess(
        spec(
            supporting_evidence_ids=("oi_growth",),
            contradicting_evidence_ids=(),
            required_data=(),
        ),
        (
            support(
                "oi_growth",
                "derivatives",
                "leverage_expansion",
            ),
        ),
    )

    assert result.status == "weakened"
    assert result.alert_level == "yellow"
    assert not result.causal_claim_allowed
    assert "insufficient for a causal claim" in result.statement


def test_triggered_invalidation_rejects_claim():
    engine = CausalEvidenceEngine()

    rule = InvalidationRule(
        rule_id="full-reversal",
        description="Full reversal invalidates continuation.",
        trigger_evidence_ids=("full_reversal",),
    )

    hypothesis = spec(
        contradicting_evidence_ids=(),
        invalidation_rules=(rule,),
        required_data=(),
    )

    evidence = (
        support(
            "oi_growth",
            "derivatives",
            "leverage_expansion",
        ),
        support(
            "spot_buying",
            "spot_flow",
            "spot_aggressive_demand",
        ),
        contradict(
            "full_reversal",
            "price_action",
            "full_reversal",
            0.95,
        ),
    )

    result = engine.assess(hypothesis, evidence)

    assert result.status == "rejected"
    assert result.alert_level == "none"
    assert not result.causal_claim_allowed
    assert result.triggered_invalidations == (
        rule.description,
    )


def test_bundle_contains_all_hypothesis_categories():
    engine = CausalEvidenceEngine()

    common = {
        "supporting_evidence_ids": (
            "oi_growth",
            "spot_buying",
        ),
        "contradicting_evidence_ids": (),
        "required_data": (),
    }

    specs = (
        spec(
            hypothesis_id="phase",
            category="market_phase",
            label="markup phase",
            causal_claim="the market is in a markup phase",
            **common,
        ),
        spec(
            hypothesis_id="pump",
            category="pump_cause",
            label="leveraged pump",
            causal_claim="leverage contributed to the pump",
            **common,
        ),
        spec(
            hypothesis_id="dump",
            category="dump_mechanism",
            label="long liquidation",
            causal_claim="long liquidation accelerated the dump",
            **common,
        ),
        spec(
            hypothesis_id="alternative",
            category="alternative",
            label="spot repricing",
            causal_claim="spot repricing explains the movement",
            **common,
        ),
    )

    bundle = engine.assess_many(
        specs,
        (
            support(
                "oi_growth",
                "derivatives",
                "leverage_expansion",
            ),
            support(
                "spot_buying",
                "spot_flow",
                "spot_aggressive_demand",
            ),
        ),
    )

    assert len(bundle.market_phases) == 1
    assert len(bundle.pump_causes) == 1
    assert len(bundle.dump_mechanisms) == 1
    assert len(bundle.alternatives) == 1
    assert bundle.global_alert_level == "red"


def test_missing_data_reduces_confidence():
    engine = CausalEvidenceEngine()

    evidence = (
        support(
            "oi_growth",
            "derivatives",
            "leverage_expansion",
            0.9,
        ),
        support(
            "spot_buying",
            "spot_flow",
            "spot_aggressive_demand",
            0.9,
        ),
    )

    complete = engine.assess(
        spec(contradicting_evidence_ids=()),
        evidence,
        available_data=(
            "open_interest",
            "spot_flow",
        ),
    )
    incomplete = engine.assess(
        spec(contradicting_evidence_ids=()),
        evidence,
        available_data=(),
    )

    assert incomplete.score == pytest.approx(complete.score)
    assert incomplete.confidence < complete.confidence
    assert set(incomplete.missing_data) == {
        "open_interest",
        "spot_flow",
    }
