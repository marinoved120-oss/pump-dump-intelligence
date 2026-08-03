from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple


MarketType = Literal["spot", "futures"]


def _to_tuple_2d(
    pairs: Tuple[Tuple[float, float], ...]
    | list[tuple[float, float]]
    | None,
) -> Tuple[Tuple[float, float], ...]:
    if pairs is None:
        return tuple()
    if isinstance(pairs, tuple):
        # ensure inner tuples
        return tuple(tuple(p) for p in pairs)
    return tuple(tuple(p) for p in pairs)


@dataclass(frozen=True)
class DepthUpdate:
    """Immutable depth update for spot/futures order books.

    - market_type: explicit 'spot' or 'futures'.
    - exchange_ts: millisecond timestamp from the exchange feed.
    - sequence: exchange-provided sequence/lastUpdateId for gap detection.
    - is_snapshot: True for a full/safe snapshot that can re-initialize state.
    - bids/asks: tuple-of-tuples [(price, size), ...] representing deltas (size=0 => remove).
    """

    exchange: str
    symbol: str
    market_type: MarketType
    exchange_ts: int
    sequence: int
    is_snapshot: bool
    bids: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)
    asks: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    def __post_init__(self):  # type: ignore[override]
        # Normalize lists to tuples to keep deep immutability of collections
        object.__setattr__(self, "bids", _to_tuple_2d(self.bids))
        object.__setattr__(self, "asks", _to_tuple_2d(self.asks))


@dataclass(frozen=True)
class Trade:
    """Immutable trade record.

    - market_type: explicit 'spot' or 'futures'.
    - exchange_ts: millisecond timestamp from the exchange.
    - side: 'buy' or 'sell' aggressor side.
    """

    exchange: str
    symbol: str
    market_type: MarketType
    exchange_ts: int
    trade_id: str
    price: float
    size: float
    side: Literal["buy", "sell", "unknown"] = "unknown"


@dataclass(frozen=True)
class OpenInterest:
    """Immutable open interest point for a futures instrument."""

    exchange: str
    symbol: str
    market_type: Literal["futures"] = "futures"
    exchange_ts: int = 0
    open_interest_contracts: Optional[float] = None
    open_interest_usd: Optional[float] = None


@dataclass(frozen=True)
class FundingUpdate:
    """Immutable funding update for a perpetual swap or futures.

    funding_ts is the effective timestamp for the funding period.
    """

    exchange: str
    symbol: str
    market_type: Literal["futures"] = "futures"
    exchange_ts: int = 0
    funding_ts: int = 0
    funding_rate: float = 0.0
    interval_seconds: Optional[int] = None


@dataclass(frozen=True)
class Liquidation:
    """Immutable liquidation print from derivatives venue."""

    exchange: str
    symbol: str
    market_type: Literal["futures"] = "futures"
    exchange_ts: int = 0
    price: float = 0.0
    size: float = 0.0
    side: Literal["long", "short", "unknown"] = "unknown"


__all__ = [
    "MarketType",
    "DepthUpdate",
    "Trade",
    "OpenInterest",
    "FundingUpdate",
    "Liquidation",
]
