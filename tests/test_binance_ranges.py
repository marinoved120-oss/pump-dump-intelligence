from __future__ import annotations

from datetime import UTC, datetime, timedelta

from research.data.binance import DateRange


def test_open_interest_range_is_clamped_to_under_29_days() -> None:
    end = datetime(2026, 8, 3, tzinfo=UTC)
    original = DateRange(
        int((end - timedelta(days=90)).timestamp() * 1000),
        int(end.timestamp() * 1000),
    )
    clamped = original.clamp_to_recent_days(29)
    span = timedelta(milliseconds=clamped.end_ms - clamped.start_ms)
    assert span < timedelta(days=29)
    assert clamped.end_ms == original.end_ms


def test_short_range_is_not_extended() -> None:
    end = datetime(2026, 8, 3, tzinfo=UTC)
    original = DateRange(
        int((end - timedelta(days=7)).timestamp() * 1000),
        int(end.timestamp() * 1000),
    )
    clamped = original.clamp_to_recent_days(29)
    assert clamped == original
