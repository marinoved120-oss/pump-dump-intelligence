import pytest

from research.live.schemas import DepthUpdate
from research.live.sequence import SequenceApplier


def make_snapshot(seq: int, ts: int) -> DepthUpdate:
    return DepthUpdate(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="spot",
        exchange_ts=ts,
        sequence=seq,
        is_snapshot=True,
        bids=((30000.0, 1.0),),
        asks=((30010.0, 1.0),),
    )


def make_inc(seq: int, ts: int) -> DepthUpdate:
    return DepthUpdate(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="spot",
        exchange_ts=ts,
        sequence=seq,
        is_snapshot=False,
        bids=((30000.0, 0.5),),
        asks=((30010.0, 0.7),),
    )


def test_ordered_updates_apply_and_advance_sequence():
    applier = SequenceApplier(require_snapshot_first=True)

    # Before any snapshot, incrementals are rejected
    res0 = applier.apply_depth(make_inc(10, 1000))
    assert not res0.applied and res0.reason == "requires_snapshot"

    # Initial snapshot
    res1 = applier.apply_depth(make_snapshot(100, 1100))
    assert res1.applied and res1.state.in_sync and res1.state.last_sequence == 100

    # Next ordered incrementals
    res2 = applier.apply_depth(make_inc(101, 1200))
    assert res2.applied and res2.state.last_sequence == 101
    res3 = applier.apply_depth(make_inc(102, 1300))
    assert res3.applied and res3.state.last_sequence == 102


def test_duplicate_and_out_of_order_handling():
    applier = SequenceApplier(require_snapshot_first=True)
    applier.apply_depth(make_snapshot(200, 2000))
    applier.apply_depth(make_inc(201, 2100))

    # Duplicate sequence
    dup = applier.apply_depth(make_inc(201, 2150))
    assert not dup.applied and dup.reason == "duplicate"
    assert dup.state.duplicate_count == 1

    # Out-of-order (older than last)
    ooo = applier.apply_depth(make_inc(200, 2200))
    assert not ooo.applied and ooo.reason == "stale_sequence"
    assert ooo.state.out_of_order_count == 1


def test_gap_detection_and_blocking_incrementals_until_resync():
    applier = SequenceApplier(require_snapshot_first=True)
    applier.apply_depth(make_snapshot(300, 3000))
    applier.apply_depth(make_inc(301, 3100))

    # Gap appears (jump from 301 to 305)
    gap = applier.apply_depth(make_inc(305, 3200))
    assert not gap.applied and gap.reason == "gap"
    assert not gap.state.in_sync and gap.state.gap_detected
    assert gap.state.data_quality_score == 0.0

    # Subsequent incrementals are blocked while out_of_sync
    blocked = applier.apply_depth(make_inc(306, 3300))
    assert not blocked.applied and blocked.reason == "out_of_sync"

    # Resynchronize with a fresh snapshot
    resnap = applier.apply_depth(make_snapshot(400, 3400))
    assert resnap.applied and resnap.state.in_sync and resnap.state.last_sequence == 400
    assert resnap.state.data_quality_score == 1.0


def test_staleness_reduces_data_quality():
    applier = SequenceApplier(require_snapshot_first=True, staleness_warn_ms=100, staleness_critical_ms=300)
    res = applier.apply_depth(make_snapshot(10, 1000), now_ms=1000)
    assert res.applied and res.state.data_quality_score == 1.0

    # After some staleness over warn threshold, quality degrades to <= 0.7
    applier.recompute_quality(now_ms=1150)
    assert applier.state.data_quality_score <= 0.7

    # Critical staleness further degrades to <= 0.3
    applier.recompute_quality(now_ms=1400)
    assert applier.state.data_quality_score <= 0.3


def test_missing_source_marks_quality_lower():
    applier = SequenceApplier()
    applier.apply_depth(make_snapshot(1, 1000))
    before = applier.state.data_quality_score
    applier.mark_missing_source()
    assert applier.state.data_quality_score <= before


def test_resnapshot_same_sequence_is_duplicate_not_applied():
    applier = SequenceApplier()
    applier.apply_depth(make_snapshot(50, 1000))
    # resend snapshot with same sequence, should be duplicate
    dup = applier.apply_depth(make_snapshot(50, 1010))
    assert not dup.applied and dup.reason == "duplicate"
