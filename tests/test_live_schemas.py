from dataclasses import FrozenInstanceError

import pytest

from research.live.schemas import (
    DepthUpdate,
    Trade,
    OpenInterest,
    FundingUpdate,
    Liquidation,
)


def test_depth_update_is_immutable_and_has_required_fields():
    upd = DepthUpdate(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="spot",
        exchange_ts=1700000000000,
        sequence=123,
        is_snapshot=True,
        bids=[(30000.0, 1.5)],
        asks=((30010.0, 2.0),),
    )

    assert upd.market_type == "spot"
    assert isinstance(upd.exchange_ts, int)
    # normalized to tuple-of-tuples
    assert isinstance(upd.bids, tuple) and isinstance(upd.bids[0], tuple)
    assert isinstance(upd.asks, tuple) and isinstance(upd.asks[0], tuple)

    with pytest.raises(FrozenInstanceError):
        upd.sequence = 124  # type: ignore[attr-defined]

    with pytest.raises(AttributeError):
        upd.bids.append((29999.0, 1.0))  # type: ignore[attr-defined]


def test_trade_and_derivative_records_are_immutable_and_typed():
    tr = Trade(
        exchange="binance",
        symbol="BTCUSDT",
        market_type="spot",
        exchange_ts=1700000000100,
        trade_id="t1",
        price=30005.0,
        size=0.25,
        side="buy",
    )
    assert tr.market_type == "spot"
    with pytest.raises(FrozenInstanceError):
        tr.price = 1.0  # type: ignore[attr-defined]

    oi = OpenInterest(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_ts=1700000000200,
        open_interest_contracts=1000.0,
        open_interest_usd=30000000.0,
    )
    assert oi.market_type == "futures"
    with pytest.raises(FrozenInstanceError):
        oi.open_interest_usd = 0.0  # type: ignore[attr-defined]

    fu = FundingUpdate(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_ts=1700000000300,
        funding_ts=1700000000000,
        funding_rate=0.0001,
        interval_seconds=28800,
    )
    assert fu.market_type == "futures"
    with pytest.raises(FrozenInstanceError):
        fu.funding_rate = 0.0  # type: ignore[attr-defined]

    liq = Liquidation(
        exchange="binance",
        symbol="BTCUSDT",
        exchange_ts=1700000000400,
        price=29900.0,
        size=10.0,
        side="long",
    )
    assert liq.market_type == "futures"
    with pytest.raises(FrozenInstanceError):
        liq.size = 0.0  # type: ignore[attr-defined]


def test_market_type_supports_spot_and_futures():
    s = DepthUpdate(
        exchange="binance",
        symbol="ETHUSDT",
        market_type="spot",
        exchange_ts=1700000000000,
        sequence=1,
        is_snapshot=True,
    )
    f = DepthUpdate(
        exchange="binance",
        symbol="ETHUSDT",
        market_type="futures",
        exchange_ts=1700000000000,
        sequence=2,
        is_snapshot=True,
    )
    assert s.market_type == "spot"
    assert f.market_type == "futures"
