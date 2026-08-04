from dataclasses import fields

import pytest

from research.live.schemas import DepthUpdate, Trade
from research.orderbook.walls import (
    WallLifecycle,
    WallTracker,
    WallTrackerConfig,
    WallTrackerEngine,
)


def make_depth(
    *,
    ts: int,
    sequence: int,
    market_type: str = "spot",
    snapshot: bool = False,
    bids=(),
    asks=(),
):
    return DepthUpdate(
        exchange="binance",
        symbol="BTCUSDT",
        market_type=market_type,
        exchange_ts=ts,
        sequence=sequence,
        is_snapshot=snapshot,
        bids=tuple(bids),
        asks=tuple(asks),
    )


def make_config(**overrides):
    values = {
        "local_depth_levels": 3,
        "relative_depth_fraction": 0.50,
        "historical_percentile": 0.75,
        "history_size": 100,
        "min_history_observations": 3,
        "min_absolute_size": 0.0,
        "min_persistence_observations": 2,
        "min_persistence_ms": 0,
        "liquidity_pull_fraction": 0.20,
        "refill_fraction": 0.20,
        "execution_match_bps": 5.0,
        "reposition_max_bps": 20.0,
        "reposition_size_tolerance": 0.20,
    }
    values.update(overrides)
    return WallTrackerConfig(**values)


def test_threshold_uses_local_depth_and_historical_percentile():
    tracker = WallTracker(
        "binance",
        "BTCUSDT",
        "spot",
        make_config(),
    )

    first = tracker.apply_depth(
        make_depth(
            ts=1_000,
            sequence=1,
            snapshot=True,
            bids=((100.0, 8.0), (99.0, 1.0), (98.0, 1.0)),
            asks=((101.0, 1.0), (102.0, 1.0), (103.0, 1.0)),
        )
    )

    detected = [
        event
        for event in first
        if event.event == "detected"
        and event.side == "bid"
    ]
    assert len(detected) == 1
    assert "local_depth_threshold=5" in detected[0].evidence
    assert (
        "historical_percentile_threshold=0"
        in detected[0].evidence
    )

    second = tracker.apply_depth(
        make_depth(
            ts=2_000,
            sequence=2,
            snapshot=True,
            bids=((97.0, 4.0), (96.0, 1.0), (95.0, 1.0)),
            asks=((101.0, 1.0), (102.0, 1.0), (103.0, 1.0)),
        )
    )

    threshold = tracker.last_thresholds["bid"]
    assert threshold.local_threshold == pytest.approx(3.0)
    assert threshold.historical_threshold == pytest.approx(4.5)
    assert threshold.effective_threshold == pytest.approx(4.5)

    assert not any(
        event.event == "detected"
        and event.price == 97.0
        for event in second
    )


def test_size_does_not_assign_real_or_fake_label():
    tracker = WallTracker(
        "binance",
        "BTCUSDT",
        "spot",
        make_config(min_history_observations=100),
    )

    tracker.apply_depth(
        make_depth(
            ts=1_000,
            sequence=1,
            snapshot=True,
            bids=((100.0, 100.0), (99.0, 1.0), (98.0, 1.0)),
            asks=((101.0, 1.0), (102.0, 1.0), (103.0, 1.0)),
        )
    )

    wall = tracker.active_walls[0]
    lifecycle_fields = {item.name for item in fields(WallLifecycle)}

    assert "real" not in lifecycle_fields
    assert "fake" not in lifecycle_fields
    assert "verdict" not in lifecycle_fields

    assert wall.status == "active"
    assert wall.closed_reason is None
    assert wall.observations
    assert all(
        observation.evidence
        for observation in wall.observations
    )


def test_persistence_execution_and_refill_lifecycle():
    tracker = WallTracker(
        "binance",
        "BTCUSDT",
        "spot",
        make_config(min_history_observations=100),
    )

    tracker.apply_depth(
        make_depth(
            ts=1_000,
            sequence=1,
            snapshot=True,
            bids=((100.0, 1.0), (99.0, 1.0), (98.0, 1.0)),
            asks=((101.0, 10.0), (102.0, 1.0), (103.0, 1.0)),
        )
    )

    persisted_events = tracker.apply_depth(
        make_depth(
            ts=2_000,
            sequence=2,
            asks=((101.0, 10.0),),
        )
    )
    assert any(
        event.event == "persisted"
        for event in persisted_events
    )

    tracker.apply_trade(
        Trade(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="spot",
            exchange_ts=2_500,
            trade_id="trade-1",
            price=101.0,
            size=4.0,
            side="buy",
        )
    )

    execution_events = tracker.apply_depth(
        make_depth(
            ts=3_000,
            sequence=3,
            asks=((101.0, 6.0),),
        )
    )
    partial = next(
        event
        for event in execution_events
        if event.event == "partially_executed"
    )
    assert partial.executed_size == pytest.approx(4.0)

    refill_events = tracker.apply_depth(
        make_depth(
            ts=4_000,
            sequence=4,
            asks=((101.0, 10.0),),
        )
    )
    refill = next(
        event
        for event in refill_events
        if event.event == "refilled"
    )
    assert refill.size == pytest.approx(10.0)
    assert refill.previous_size == pytest.approx(6.0)

    tracker.apply_trade(
        Trade(
            exchange="binance",
            symbol="BTCUSDT",
            market_type="spot",
            exchange_ts=4_500,
            trade_id="trade-2",
            price=101.0,
            size=10.0,
            side="buy",
        )
    )

    final_events = tracker.apply_depth(
        make_depth(
            ts=5_000,
            sequence=5,
            asks=((101.0, 0.0),),
        )
    )

    assert any(
        event.event == "executed"
        for event in final_events
    )
    assert tracker.closed_walls[0].closed_reason == "executed"


def test_cancellation_and_liquidity_pull_are_separate_events():
    tracker = WallTracker(
        "binance",
        "BTCUSDT",
        "spot",
        make_config(min_history_observations=100),
    )

    tracker.apply_depth(
        make_depth(
            ts=1_000,
            sequence=1,
            snapshot=True,
            bids=((100.0, 10.0), (99.0, 1.0), (98.0, 1.0)),
            asks=((101.0, 1.0), (102.0, 1.0), (103.0, 1.0)),
        )
    )

    events = tracker.apply_depth(
        make_depth(
            ts=2_000,
            sequence=2,
            bids=((100.0, 0.0),),
        )
    )

    event_types = {event.event for event in events}
    assert "liquidity_pulled" in event_types
    assert "cancelled" in event_types

    wall = tracker.closed_walls[0]
    assert wall.closed_reason == "cancelled"


def test_repositioning_keeps_the_same_wall_lifecycle():
    tracker = WallTracker(
        "binance",
        "BTCUSDT",
        "spot",
        make_config(min_history_observations=100),
    )

    tracker.apply_depth(
        make_depth(
            ts=1_000,
            sequence=1,
            snapshot=True,
            bids=((100.0, 10.0), (99.0, 1.0), (98.0, 1.0)),
            asks=((101.0, 1.0), (102.0, 1.0), (103.0, 1.0)),
        )
    )

    original_wall_id = tracker.active_walls[0].wall_id

    events = tracker.apply_depth(
        make_depth(
            ts=2_000,
            sequence=2,
            bids=((100.0, 0.0), (99.9, 9.5)),
        )
    )

    reposition = next(
        event
        for event in events
        if event.event == "repositioned"
    )

    assert reposition.wall_id == original_wall_id
    assert reposition.previous_price == pytest.approx(100.0)
    assert reposition.price == pytest.approx(99.9)

    assert len(tracker.lifecycles) == 1
    assert tracker.active_walls[0].price == pytest.approx(99.9)


def test_spot_and_futures_results_are_separate_and_comparable():
    engine = WallTrackerEngine(
        make_config(min_history_observations=100)
    )

    for market_type in ("spot", "futures"):
        engine.process_depth(
            make_depth(
                ts=1_000,
                sequence=1,
                market_type=market_type,
                snapshot=True,
                bids=((100.0, 10.0), (99.0, 1.0), (98.0, 1.0)),
                asks=((101.0, 1.0), (102.0, 1.0), (103.0, 1.0)),
            )
        )

    spot = engine.tracker("binance", "BTCUSDT", "spot")
    futures = engine.tracker("binance", "BTCUSDT", "futures")

    assert spot is not futures
    assert len(spot.active_walls) == 1
    assert len(futures.active_walls) == 1
    assert (
        spot.active_walls[0].wall_id
        != futures.active_walls[0].wall_id
    )

    summaries = {
        summary.market_type: summary
        for summary in engine.summaries()
    }

    assert set(summaries) == {"spot", "futures"}
    assert summaries["spot"].active_wall_count == 1
    assert summaries["futures"].active_wall_count == 1
    assert (
        summaries["spot"].event_counts
        == summaries["futures"].event_counts
    )
