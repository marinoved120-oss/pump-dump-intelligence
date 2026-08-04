import pytest

from research.evidence.manipulation import (
    AbsorptionWindow,
    EvidenceConfig,
    ManipulationEvidenceAnalyzer,
)
from research.orderbook.walls import (
    WallLifecycle,
    WallObservation,
)


def observation(
    wall,
    *,
    event,
    ts,
    size,
    previous_size=None,
    executed_size=0.0,
    pulled_size=0.0,
):
    return WallObservation(
        wall_id=wall.wall_id,
        event=event,
        exchange=wall.exchange,
        symbol=wall.symbol,
        market_type=wall.market_type,
        side=wall.side,
        ts_ms=ts,
        price=wall.price,
        size=size,
        previous_size=previous_size,
        executed_size=executed_size,
        pulled_size=pulled_size,
        evidence=("test_observation",),
    )


def make_wall(
    *,
    market_type="spot",
    side="ask",
    initial_size=100.0,
    first_seen_ms=0,
    closed_at_ms=1_000,
):
    wall = WallLifecycle(
        wall_id=(
            f"binance:{market_type}:BTCUSDT:"
            f"{side}:000001"
        ),
        exchange="binance",
        symbol="BTCUSDT",
        market_type=market_type,
        side=side,
        initial_price=101.0,
        initial_size=initial_size,
        price=101.0,
        current_size=0.0,
        peak_size=initial_size,
        first_seen_ms=first_seen_ms,
        last_seen_ms=closed_at_ms,
        observation_count=3,
        persisted=False,
        status="closed",
        closed_at_ms=closed_at_ms,
        closed_reason="cancelled",
    )
    return wall


def test_spoofing_score_uses_all_required_dimensions():
    analyzer = ManipulationEvidenceAnalyzer()

    wall = make_wall()
    wall.observations.extend(
        [
            observation(
                wall,
                event="liquidity_pulled",
                ts=1_000,
                size=0.0,
                previous_size=100.0,
                pulled_size=100.0,
            ),
            observation(
                wall,
                event="cancelled",
                ts=1_000,
                size=0.0,
                previous_size=100.0,
                pulled_size=100.0,
            ),
        ]
    )

    repeated = analyzer.spoofing_report(
        wall,
        touch_price=100.0,
        repetition_count=3,
    )
    first_occurrence = analyzer.spoofing_report(
        wall,
        touch_price=100.0,
        repetition_count=0,
    )

    metrics = dict(repeated.metrics)
    assert set(metrics) == {
        "lifetime_ms",
        "distance_bps",
        "cancellation_ratio",
        "execution_ratio",
        "repetition_count",
    }

    assert repeated.supported
    assert repeated.score > first_occurrence.score
    assert metrics["lifetime_ms"] == pytest.approx(1_000)
    assert metrics["distance_bps"] == pytest.approx(100.0)
    assert metrics["cancellation_ratio"] == pytest.approx(1.0)
    assert metrics["execution_ratio"] == pytest.approx(0.0)
    assert metrics["repetition_count"] == pytest.approx(3.0)


def test_substantial_execution_contradicts_spoofing_hypothesis():
    analyzer = ManipulationEvidenceAnalyzer()

    wall = make_wall()
    wall.closed_reason = "mixed_execution_and_cancellation"
    wall.observations.extend(
        [
            observation(
                wall,
                event="partially_executed",
                ts=1_000,
                size=20.0,
                previous_size=100.0,
                executed_size=80.0,
            ),
            observation(
                wall,
                event="liquidity_pulled",
                ts=1_000,
                size=0.0,
                previous_size=20.0,
                pulled_size=20.0,
            ),
            observation(
                wall,
                event="cancelled",
                ts=1_000,
                size=0.0,
                previous_size=20.0,
                pulled_size=20.0,
            ),
        ]
    )

    report = analyzer.spoofing_report(
        wall,
        touch_price=100.0,
        repetition_count=3,
    )

    assert not report.supported
    assert any(
        item.startswith("substantial_execution_ratio=")
        for item in report.contradictions
    )
    assert report.metric("execution_ratio") == pytest.approx(0.8)


def test_iceberg_proxy_requires_refill_and_executed_to_visible():
    analyzer = ManipulationEvidenceAnalyzer()

    wall = make_wall(initial_size=10.0)
    wall.closed_reason = "executed"
    wall.observations.extend(
        [
            observation(
                wall,
                event="refilled",
                ts=500,
                size=10.0,
                previous_size=5.0,
            ),
            observation(
                wall,
                event="refilled",
                ts=700,
                size=10.0,
                previous_size=5.0,
            ),
            observation(
                wall,
                event="executed",
                ts=1_000,
                size=0.0,
                previous_size=10.0,
                executed_size=20.0,
            ),
        ]
    )

    report = analyzer.iceberg_report(wall)

    assert report.supported
    assert report.metric("refill_count") == pytest.approx(2.0)
    assert report.metric("refill_volume") == pytest.approx(10.0)
    assert report.metric(
        "executed_to_visible_ratio"
    ) == pytest.approx(2.0)
    assert any(
        item.startswith("repeated_refill_count=")
        for item in report.evidence
    )


def test_execution_without_refill_is_not_iceberg_support():
    analyzer = ManipulationEvidenceAnalyzer()

    wall = make_wall(initial_size=10.0)
    wall.observations.append(
        observation(
            wall,
            event="executed",
            ts=1_000,
            size=0.0,
            previous_size=10.0,
            executed_size=30.0,
        )
    )

    report = analyzer.iceberg_report(wall)

    assert not report.supported
    assert "no_visible_refill_observed" in report.contradictions
    assert any(
        item.startswith("insufficient_refill_count=")
        for item in report.contradictions
    )


def test_absorption_requires_flow_and_weak_price_response():
    analyzer = ManipulationEvidenceAnalyzer(
        EvidenceConfig(
            absorption_min_aggressive_volume=10.0,
            absorption_min_flow_to_depth_ratio=1.0,
            absorption_max_price_response_bps=10.0,
        )
    )

    supported = analyzer.absorption_report(
        AbsorptionWindow(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="futures",
            aggressive_side="buy",
            start_ts_ms=0,
            end_ts_ms=1_000,
            aggressive_volume=200.0,
            reference_visible_depth=100.0,
            start_price=100.0,
            end_price=100.05,
        )
    )

    strong_response = analyzer.absorption_report(
        AbsorptionWindow(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="futures",
            aggressive_side="buy",
            start_ts_ms=0,
            end_ts_ms=1_000,
            aggressive_volume=200.0,
            reference_visible_depth=100.0,
            start_price=100.0,
            end_price=101.0,
        )
    )

    weak_flow = analyzer.absorption_report(
        AbsorptionWindow(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="futures",
            aggressive_side="buy",
            start_ts_ms=0,
            end_ts_ms=1_000,
            aggressive_volume=5.0,
            reference_visible_depth=100.0,
            start_price=100.0,
            end_price=100.01,
        )
    )

    assert supported.supported
    assert not strong_response.supported
    assert not weak_flow.supported

    assert any(
        item.startswith("strong_price_response_bps=")
        for item in strong_response.contradictions
    )
    assert any(
        item.startswith("insufficient_aggressive_flow=")
        for item in weak_flow.contradictions
    )


def test_reports_are_cautious_and_preserve_market_identity():
    analyzer = ManipulationEvidenceAnalyzer()

    spot_wall = make_wall(market_type="spot")
    futures_wall = make_wall(market_type="futures")

    for wall in (spot_wall, futures_wall):
        wall.observations.extend(
            [
                observation(
                    wall,
                    event="liquidity_pulled",
                    ts=1_000,
                    size=0.0,
                    previous_size=100.0,
                    pulled_size=100.0,
                ),
                observation(
                    wall,
                    event="cancelled",
                    ts=1_000,
                    size=0.0,
                    previous_size=100.0,
                    pulled_size=100.0,
                ),
            ]
        )

    spot_report = analyzer.spoofing_report(
        spot_wall,
        touch_price=100.0,
        repetition_count=3,
    )
    futures_report = analyzer.spoofing_report(
        futures_wall,
        touch_price=100.0,
        repetition_count=3,
    )

    assert spot_report.market_type == "spot"
    assert futures_report.market_type == "futures"

    for report in (spot_report, futures_report):
        assert 0.0 <= report.score <= 1.0
        assert 0.0 <= report.confidence <= 1.0
        assert isinstance(report.evidence, tuple)
        assert isinstance(report.contradictions, tuple)

        wording = report.wording.lower()
        assert "hypothesis" in wording
        assert "does not establish intent" in wording
        assert "wrongdoing" in wording
