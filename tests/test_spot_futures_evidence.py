import pytest

from research.derivatives.context import (
    DerivativesContextWindow,
    interpret_oi_price,
)
from research.evidence.spot_futures import (
    SpotFuturesEvidenceAnalyzer,
)


def make_window(**overrides):
    values = {
        "exchange": "binance",
        "symbol": "BTCUSDT",
        "start_ts_ms": 0,
        "end_ts_ms": 1_000,
        "spot_return_bps": 20.0,
        "futures_return_bps": 20.0,
        "spot_buy_volume": 500.0,
        "spot_sell_volume": 500.0,
        "futures_buy_volume": 500.0,
        "futures_sell_volume": 500.0,
        "spot_visible_depth": 1_000.0,
        "futures_visible_depth": 1_000.0,
        "basis_start_bps": 5.0,
        "basis_end_bps": 5.0,
        "open_interest_start": 100.0,
        "open_interest_end": 100.0,
        "funding_rate": 0.0,
        "long_liquidation_volume": 1.0,
        "short_liquidation_volume": 1.0,
    }
    values.update(overrides)
    return DerivativesContextWindow(**values)


def test_oi_price_combinations_have_interpretations_and_counterexamples():
    cases = (
        (100.0, 0.10, "price_up_oi_up", "leveraged"),
        (100.0, -0.10, "price_up_oi_down", "short covering"),
        (-100.0, 0.10, "price_down_oi_up", "short"),
        (-100.0, -0.10, "price_down_oi_down", "deleveraging"),
    )

    for price_return, oi_change, combination, phrase in cases:
        result = interpret_oi_price(
            price_return,
            oi_change,
        )

        assert result.combination == combination
        assert phrase in result.interpretation.lower()
        assert result.counterexamples
        assert all(result.counterexamples)


def test_short_squeeze_sequence_is_classified():
    analyzer = SpotFuturesEvidenceAnalyzer()

    prior = make_window(
        start_ts_ms=0,
        end_ts_ms=1_000,
        spot_return_bps=-20.0,
        futures_return_bps=-30.0,
        open_interest_start=100.0,
        open_interest_end=115.0,
    )
    current = make_window(
        start_ts_ms=1_000,
        end_ts_ms=2_000,
        spot_return_bps=80.0,
        futures_return_bps=120.0,
        futures_buy_volume=900.0,
        futures_sell_volume=100.0,
        open_interest_start=115.0,
        open_interest_end=100.0,
        short_liquidation_volume=500.0,
        long_liquidation_volume=10.0,
    )

    report = analyzer.analyze(current, prior=prior)

    assert report.supported
    assert report.classification == "short_squeeze"
    assert report.movement_leader == "futures"
    assert (
        report.oi_price_interpretation.combination
        == "price_up_oi_down"
    )
    assert any(
        item.startswith("short_liquidation_ratio=")
        for item in report.evidence
    )
    assert report.counterexamples


def test_late_long_buildup_sequence_is_classified():
    analyzer = SpotFuturesEvidenceAnalyzer()

    prior = make_window(
        start_ts_ms=0,
        end_ts_ms=1_000,
        spot_return_bps=100.0,
        futures_return_bps=150.0,
        open_interest_start=100.0,
        open_interest_end=105.0,
    )
    current = make_window(
        start_ts_ms=1_000,
        end_ts_ms=2_000,
        spot_return_bps=10.0,
        futures_return_bps=90.0,
        spot_buy_volume=520.0,
        spot_sell_volume=480.0,
        futures_buy_volume=900.0,
        futures_sell_volume=100.0,
        basis_start_bps=5.0,
        basis_end_bps=25.0,
        open_interest_start=105.0,
        open_interest_end=125.0,
        funding_rate=0.001,
        long_liquidation_volume=1.0,
        short_liquidation_volume=1.0,
    )

    report = analyzer.analyze(current, prior=prior)

    assert report.supported
    assert report.classification == "late_long_buildup"
    assert report.movement_leader == "futures"
    assert "weak_spot_confirmation" in report.evidence
    assert any(
        item.startswith("elevated_funding_rate=")
        for item in report.evidence
    )


def test_spot_led_and_futures_led_movements_are_separate():
    analyzer = SpotFuturesEvidenceAnalyzer()

    spot_led = analyzer.analyze(
        make_window(
            spot_return_bps=120.0,
            futures_return_bps=50.0,
            spot_buy_volume=900.0,
            spot_sell_volume=100.0,
            basis_start_bps=5.0,
            basis_end_bps=6.0,
            open_interest_start=100.0,
            open_interest_end=101.0,
        )
    )

    futures_led = analyzer.analyze(
        make_window(
            spot_return_bps=30.0,
            futures_return_bps=120.0,
            futures_buy_volume=900.0,
            futures_sell_volume=100.0,
            basis_start_bps=5.0,
            basis_end_bps=25.0,
            open_interest_start=100.0,
            open_interest_end=115.0,
        )
    )

    assert spot_led.classification == "organic_spot_demand"
    assert spot_led.movement_leader == "spot"

    assert futures_led.classification == "futures_led_pump"
    assert futures_led.movement_leader == "futures"


def test_missing_oi_and_liquidations_lower_confidence():
    analyzer = SpotFuturesEvidenceAnalyzer()

    complete = analyzer.analyze(
        make_window(
            spot_return_bps=120.0,
            futures_return_bps=50.0,
            spot_buy_volume=900.0,
            spot_sell_volume=100.0,
            basis_start_bps=5.0,
            basis_end_bps=6.0,
            open_interest_start=100.0,
            open_interest_end=100.0,
            funding_rate=0.0,
            long_liquidation_volume=1.0,
            short_liquidation_volume=1.0,
        )
    )

    incomplete = analyzer.analyze(
        make_window(
            spot_return_bps=120.0,
            futures_return_bps=50.0,
            spot_buy_volume=900.0,
            spot_sell_volume=100.0,
            basis_start_bps=5.0,
            basis_end_bps=6.0,
            open_interest_start=None,
            open_interest_end=None,
            funding_rate=None,
            long_liquidation_volume=None,
            short_liquidation_volume=None,
        )
    )

    assert complete.classification == "organic_spot_demand"
    assert incomplete.classification == "organic_spot_demand"

    assert incomplete.confidence < complete.confidence
    assert set(incomplete.missing_data) == {
        "open_interest",
        "liquidations",
        "funding",
    }


def test_report_contains_explicit_caveats_and_metrics():
    analyzer = SpotFuturesEvidenceAnalyzer()

    report = analyzer.analyze(
        make_window(
            spot_return_bps=120.0,
            futures_return_bps=50.0,
            spot_buy_volume=900.0,
            spot_sell_volume=100.0,
        )
    )

    assert "does not establish a unique cause" in report.summary
    assert report.oi_price_interpretation.interpretation
    assert report.counterexamples

    assert report.metric("spot_return_bps") == pytest.approx(120.0)
    assert report.metric("futures_return_bps") == pytest.approx(50.0)
    assert 0.0 <= report.score <= 1.0
    assert 0.0 <= report.confidence <= 1.0
