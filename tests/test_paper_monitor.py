from pathlib import Path

import pytest

from research.monitor.paper import (
    REQUIRED_ANALYSIS_OUTPUT_FIELDS,
    JsonlOutcomeStore,
    OutcomeTracker,
    PaperAlert,
    PaperAnalysisOutput,
    PaperMonitor,
    PaperMonitorConfig,
    TelegramReportFormatter,
)


class MemorySink:
    def __init__(self):
        self.messages = []

    def send(self, text):
        self.messages.append(text)


def make_output():
    return PaperAnalysisOutput(
        market_phase="markup",
        likely_pump_causes=(
            "spot demand",
            "short covering",
        ),
        spot_role="spot-led demand",
        futures_role="confirming but not leading",
        open_interest_state="OI falling",
        funding_and_leverage="funding neutral",
        liquidation_state="short liquidations elevated",
        spot_order_book_evidence=(
            "persistent bid support",
        ),
        futures_order_book_evidence=(
            "moderate ask liquidity",
        ),
        whale_limit_orders=(
            "large bid persisted",
        ),
        spoofing_evidence=(
            "insufficient for spoofing-like hypothesis",
        ),
        iceberg_evidence=(
            "visible refill observed",
        ),
        absorption_evidence=(
            "buy flow with weak price response",
        ),
        liquidity_pull_evidence=(
            "ask liquidity pulled",
        ),
        spot_futures_divergence=(
            "spot moved before futures"
        ),
        technical_structure="range breakout",
        onchain_and_news_context=(
            "token=abc123 no verified catalyst"
        ),
        supporting_evidence=(
            "spot aggressive demand",
            "short liquidations",
            "price breakout",
        ),
        contradictions=(
            "futures basis did not expand",
        ),
        missing_data=(
            "onchain data",
        ),
        alternative_scenarios=(
            "cross-venue repricing",
        ),
        invalidation_conditions=(
            "full return below breakout level",
        ),
        observed_outcome="pending",
    )


def make_alert(**overrides):
    values = {
        "alert_id": "paper-001",
        "exchange": "binance",
        "symbol": "btcusdt",
        "created_ts_ms": 0,
        "reference_price": 100.0,
        "alert_level": "red",
        "score": 0.91,
        "confidence": 0.82,
        "data_quality_score": 0.88,
        "independent_evidence_groups": (
            "derivatives",
            "order_flow",
            "price_structure",
        ),
        "analysis": make_output(),
    }
    values.update(overrides)
    return PaperAlert(**values)


def test_telegram_report_follows_constitution_contract():
    formatter = TelegramReportFormatter()
    message = formatter.format_initial(
        make_alert()
    )

    for field_name in REQUIRED_ANALYSIS_OUTPUT_FIELDS:
        assert f"<b>{field_name}</b>:" in message

    assert "<b>data_quality_score</b>:" in message
    assert "<b>independent_evidence_groups</b>:" in message
    assert "PAPER MONITOR" in message
    assert "No order placement" in message

    assert "abc123" not in message
    assert "[REDACTED]" in message


def test_red_alert_requires_three_independent_groups():
    config = PaperMonitorConfig(
        red_min_independent_groups=3
    )
    alert = make_alert(
        independent_evidence_groups=(
            "derivatives",
            "price_structure",
        )
    )

    assert (
        alert.effective_alert_level(config)
        == "orange"
    )

    sink = MemorySink()
    monitor = PaperMonitor(
        sink,
        config=config,
    )
    delivery = monitor.publish(alert)

    assert delivery.sent
    assert delivery.effective_alert_level == "orange"
    assert (
        "<b>effective_alert_level</b>: ORANGE"
        in delivery.message
    )


def test_cooldown_and_invalidation_updates():
    sink = MemorySink()
    monitor = PaperMonitor(
        sink,
        config=PaperMonitorConfig(
            cooldown_seconds=900
        ),
    )

    first = monitor.publish(
        make_alert(alert_id="paper-001"),
        now_ts_ms=0,
    )
    repeated = monitor.publish(
        make_alert(
            alert_id="paper-002",
            created_ts_ms=1_000,
        ),
        now_ts_ms=1_000,
    )

    invalidation = monitor.publish_invalidation(
        "paper-001",
        "Full return below breakout level",
        now_ts_ms=2_000,
    )
    duplicate = monitor.publish_invalidation(
        "paper-001",
        "Full return below breakout level",
        now_ts_ms=3_000,
    )

    assert first.sent
    assert not repeated.sent
    assert repeated.reason == "cooldown_active"

    assert invalidation.sent
    assert invalidation.kind == "invalidation"
    assert not duplicate.sent
    assert duplicate.reason == "duplicate_invalidation"

    assert len(sink.messages) == 2
    assert "INVALIDATION UPDATE" in sink.messages[1]


def test_outcomes_are_recorded_at_required_checkpoints(
    tmp_path,
):
    sink = MemorySink()
    config = PaperMonitorConfig()

    store = JsonlOutcomeStore(
        tmp_path / "paper-outcomes.jsonl"
    )
    tracker = OutcomeTracker(
        config.outcome_offsets_minutes,
        store,
    )
    monitor = PaperMonitor(
        sink,
        config=config,
        outcome_tracker=tracker,
    )

    monitor.publish(make_alert(), now_ts_ms=0)

    checkpoints = (
        (5, 101.0),
        (15, 102.0),
        (30, 99.0),
        (60, 103.0),
        (240, 105.0),
    )

    for minutes, price in checkpoints:
        captured = monitor.capture_due_outcomes(
            now_ts_ms=minutes * 60_000,
            prices_by_symbol={
                "BTCUSDT": price,
            },
        )
        assert len(captured) == 1
        assert captured[0].offset_minutes == minutes

    records = store.read_all()

    assert tuple(
        record.offset_minutes
        for record in records
    ) == (5, 15, 30, 60, 240)

    assert records[0].return_bps == pytest.approx(
        100.0
    )
    assert records[-1].return_bps == pytest.approx(
        500.0
    )

    assert len(sink.messages) == 6
    assert all(
        record.alert_id == "paper-001"
        for record in records
    )


def test_outcomes_do_not_enable_model_updates():
    with pytest.raises(
        ValueError,
        match="cannot update production models",
    ):
        PaperMonitorConfig(
            production_model_updates_enabled=True
        )

    config = PaperMonitorConfig()

    assert not config.production_model_updates_enabled
    assert not config.exchange_trading_enabled
    assert not hasattr(PaperMonitor, "update_model")
    assert not hasattr(PaperMonitor, "fit_model")


def test_no_exchange_trading_endpoint_exists():
    source = Path(
        "/workspace/research/monitor/paper.py"
    ).read_text(encoding="utf-8").lower()

    forbidden = (
        "create_order",
        "place_order",
        "send_order",
        "cancel_order",
        "/api/v3/order",
        "/fapi/v1/order",
    )

    for token in forbidden:
        assert token not in source

    assert not hasattr(PaperMonitor, "buy")
    assert not hasattr(PaperMonitor, "sell")
    assert not hasattr(PaperMonitor, "withdraw")


def test_missing_data_contradictions_and_invalidations_visible():
    message = TelegramReportFormatter().format_initial(
        make_alert()
    )

    assert "<b>contradictions</b>:" in message
    assert "futures basis did not expand" in message

    assert "<b>missing_data</b>:" in message
    assert "onchain data" in message

    assert "<b>alternative_scenarios</b>:" in message
    assert "cross-venue repricing" in message

    assert "<b>invalidation_conditions</b>:" in message
    assert "full return below breakout level" in message
