from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


PriceDirection = Literal["up", "down", "flat"]
OIDirection = Literal["up", "down", "flat", "missing"]


@dataclass(frozen=True)
class OIPriceInterpretation:
    """Explicit price/OI interpretation with alternative explanations."""

    combination: str
    price_direction: PriceDirection
    oi_direction: OIDirection
    interpretation: str
    counterexamples: Tuple[str, ...]


@dataclass(frozen=True)
class DerivativesContextWindow:
    """Comparable spot/futures evidence for one market window."""

    exchange: str
    symbol: str
    start_ts_ms: int
    end_ts_ms: int

    spot_return_bps: float
    futures_return_bps: float

    spot_buy_volume: float
    spot_sell_volume: float
    futures_buy_volume: float
    futures_sell_volume: float

    spot_visible_depth: float
    futures_visible_depth: float

    basis_start_bps: float
    basis_end_bps: float

    open_interest_start: Optional[float] = None
    open_interest_end: Optional[float] = None
    funding_rate: Optional[float] = None

    long_liquidation_volume: Optional[float] = None
    short_liquidation_volume: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())

        if self.end_ts_ms < self.start_ts_ms:
            raise ValueError(
                "end_ts_ms cannot precede start_ts_ms"
            )

        volumes = (
            "spot_buy_volume",
            "spot_sell_volume",
            "futures_buy_volume",
            "futures_sell_volume",
        )
        for name in volumes:
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} cannot be negative")

        if self.spot_visible_depth <= 0:
            raise ValueError(
                "spot_visible_depth must be positive"
            )
        if self.futures_visible_depth <= 0:
            raise ValueError(
                "futures_visible_depth must be positive"
            )

        optional_nonnegative = (
            "open_interest_start",
            "open_interest_end",
            "long_liquidation_volume",
            "short_liquidation_volume",
        )
        for name in optional_nonnegative:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

    @staticmethod
    def _imbalance(buy: float, sell: float) -> float:
        total = buy + sell
        if total <= 0:
            return 0.0
        return (buy - sell) / total

    @property
    def spot_flow_imbalance(self) -> float:
        return self._imbalance(
            self.spot_buy_volume,
            self.spot_sell_volume,
        )

    @property
    def futures_flow_imbalance(self) -> float:
        return self._imbalance(
            self.futures_buy_volume,
            self.futures_sell_volume,
        )

    @property
    def basis_change_bps(self) -> float:
        return self.basis_end_bps - self.basis_start_bps

    @property
    def open_interest_change_pct(self) -> Optional[float]:
        if (
            self.open_interest_start is None
            or self.open_interest_end is None
            or self.open_interest_start <= 0
        ):
            return None

        return (
            self.open_interest_end
            - self.open_interest_start
        ) / self.open_interest_start

    @property
    def has_liquidations(self) -> bool:
        return (
            self.long_liquidation_volume is not None
            and self.short_liquidation_volume is not None
        )

    @property
    def short_liquidation_ratio(self) -> Optional[float]:
        if not self.has_liquidations:
            return None

        assert self.long_liquidation_volume is not None
        assert self.short_liquidation_volume is not None

        return self.short_liquidation_volume / max(
            self.long_liquidation_volume,
            1e-12,
        )


def _direction(
    value: float,
    flat_threshold: float,
) -> Literal["up", "down", "flat"]:
    if value > flat_threshold:
        return "up"
    if value < -flat_threshold:
        return "down"
    return "flat"


def interpret_oi_price(
    price_return_bps: float,
    oi_change_pct: Optional[float],
    *,
    flat_price_bps: float = 5.0,
    flat_oi_pct: float = 0.01,
) -> OIPriceInterpretation:
    """Interpret price/OI combinations without claiming a unique cause."""

    price_direction = _direction(
        price_return_bps,
        flat_price_bps,
    )

    if oi_change_pct is None:
        return OIPriceInterpretation(
            combination=f"price_{price_direction}_oi_missing",
            price_direction=price_direction,
            oi_direction="missing",
            interpretation=(
                "Open-interest data is unavailable, so leverage "
                "participation cannot be distinguished reliably."
            ),
            counterexamples=(
                "Price can move through spot demand without material "
                "changes in derivatives positioning.",
                "Venue-specific OI can be missing while positioning "
                "changes on other venues.",
            ),
        )

    oi_direction = _direction(
        oi_change_pct,
        flat_oi_pct,
    )
    combination = (
        f"price_{price_direction}_oi_{oi_direction}"
    )

    interpretations = {
        ("up", "up"): (
            "Price rising with OI rising is compatible with new "
            "leveraged position buildup, including fresh longs.",
            (
                "Market-neutral basis trades can increase OI "
                "without directional conviction.",
                "Hedging or cross-venue migration can increase "
                "venue OI during a price rise.",
            ),
        ),
        ("up", "down"): (
            "Price rising with OI falling is compatible with "
            "short covering or squeeze-driven deleveraging.",
            (
                "Broad position closure can reduce OI without "
                "forced short liquidations.",
                "Positions may migrate to another venue while "
                "price continues rising.",
            ),
        ),
        ("down", "up"): (
            "Price falling with OI rising is compatible with "
            "fresh short positioning or leveraged longs becoming trapped.",
            (
                "Protective hedging can increase OI without a "
                "directional bearish view.",
                "Spread trades can add OI while outright price falls.",
            ),
        ),
        ("down", "down"): (
            "Price falling with OI falling is compatible with "
            "long liquidation or broader deleveraging.",
            (
                "Voluntary position reduction can lower OI "
                "without forced liquidation.",
                "Venue migration can reduce observed OI during a decline.",
            ),
        ),
        ("flat", "up"): (
            "OI rising while price is flat is compatible with "
            "position buildup before directional resolution.",
            (
                "Balanced long and short additions can increase OI "
                "without directional pressure.",
            ),
        ),
        ("flat", "down"): (
            "OI falling while price is flat is compatible with "
            "position reduction without immediate price impact.",
            (
                "Liquidity providers may reduce inventory while "
                "spot demand offsets the price effect.",
            ),
        ),
    }

    interpretation, counterexamples = interpretations.get(
        (price_direction, oi_direction),
        (
            "Stable OI does not provide decisive evidence about "
            "new leverage entering or leaving the market.",
            (
                "Offsetting position changes can leave aggregate "
                "OI approximately unchanged.",
            ),
        ),
    )

    return OIPriceInterpretation(
        combination=combination,
        price_direction=price_direction,
        oi_direction=oi_direction,
        interpretation=interpretation,
        counterexamples=counterexamples,
    )


__all__ = [
    "DerivativesContextWindow",
    "OIPriceInterpretation",
    "interpret_oi_price",
]
