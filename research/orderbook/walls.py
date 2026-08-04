from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, Literal, Optional, Tuple

from research.live.schemas import DepthUpdate, MarketType, Trade


Side = Literal["bid", "ask"]
WallStatus = Literal["active", "closed"]
WallEvent = Literal[
    "detected",
    "observed",
    "persisted",
    "partially_executed",
    "executed",
    "refilled",
    "repositioned",
    "liquidity_pulled",
    "cancelled",
]


@dataclass(frozen=True)
class WallTrackerConfig:
    """Configuration for relative wall detection and lifecycle tracking."""

    local_depth_levels: int = 10
    relative_depth_fraction: float = 0.25
    historical_percentile: float = 0.95
    history_size: int = 2_000
    min_history_observations: int = 30
    min_absolute_size: float = 0.0

    min_persistence_observations: int = 3
    min_persistence_ms: int = 1_000

    liquidity_pull_fraction: float = 0.20
    refill_fraction: float = 0.20

    execution_match_bps: float = 5.0
    reposition_max_bps: float = 20.0
    reposition_size_tolerance: float = 0.35

    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.local_depth_levels < 1:
            raise ValueError("local_depth_levels must be positive")
        if self.history_size < 1:
            raise ValueError("history_size must be positive")
        if self.min_history_observations < 0:
            raise ValueError(
                "min_history_observations cannot be negative"
            )
        if self.min_persistence_observations < 1:
            raise ValueError(
                "min_persistence_observations must be positive"
            )
        if self.min_persistence_ms < 0:
            raise ValueError("min_persistence_ms cannot be negative")
        if self.min_absolute_size < 0:
            raise ValueError("min_absolute_size cannot be negative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

        unit_interval = (
            "relative_depth_fraction",
            "historical_percentile",
            "liquidity_pull_fraction",
            "refill_fraction",
            "reposition_size_tolerance",
        )
        for name in unit_interval:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if self.execution_match_bps < 0:
            raise ValueError("execution_match_bps cannot be negative")
        if self.reposition_max_bps < 0:
            raise ValueError("reposition_max_bps cannot be negative")


@dataclass(frozen=True)
class WallThreshold:
    side: Side
    local_depth_total: float
    local_threshold: float
    historical_threshold: float
    effective_threshold: float
    history_count: int


@dataclass(frozen=True)
class WallObservation:
    wall_id: str
    event: WallEvent
    exchange: str
    symbol: str
    market_type: MarketType
    side: Side
    ts_ms: int

    price: float
    size: float

    previous_price: Optional[float] = None
    previous_size: Optional[float] = None
    executed_size: float = 0.0
    pulled_size: float = 0.0

    evidence: Tuple[str, ...] = field(default_factory=tuple)


@dataclass
class WallLifecycle:
    """Lifecycle of one wall without assigning a real/fake verdict."""

    wall_id: str
    exchange: str
    symbol: str
    market_type: MarketType
    side: Side

    initial_price: float
    initial_size: float
    price: float
    current_size: float
    peak_size: float

    first_seen_ms: int
    last_seen_ms: int
    observation_count: int = 1

    persisted: bool = False
    status: WallStatus = "active"
    closed_at_ms: Optional[int] = None
    closed_reason: Optional[str] = None

    observations: list[WallObservation] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        end = (
            self.closed_at_ms
            if self.closed_at_ms is not None
            else self.last_seen_ms
        )
        return max(0, end - self.first_seen_ms)


@dataclass(frozen=True)
class MarketWallSummary:
    exchange: str
    symbol: str
    market_type: MarketType
    active_wall_count: int
    closed_wall_count: int
    event_counts: Tuple[Tuple[str, int], ...]


@dataclass
class _TradeLot:
    trade: Trade
    remaining: float


def _percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower

    return (
        ordered[lower]
        + (ordered[upper] - ordered[lower]) * fraction
    )


class WallTracker:
    """Track wall detection and lifecycle for one market and symbol."""

    def __init__(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType,
        config: Optional[WallTrackerConfig] = None,
    ) -> None:
        if market_type not in ("spot", "futures"):
            raise ValueError("market_type must be spot or futures")

        self.exchange = exchange
        self.symbol = symbol.upper()
        self.market_type = market_type
        self.config = config or WallTrackerConfig()

        self._book: Dict[Side, Dict[float, float]] = {
            "bid": {},
            "ask": {},
        }
        self._history: Dict[Side, Deque[float]] = {
            "bid": deque(maxlen=self.config.history_size),
            "ask": deque(maxlen=self.config.history_size),
        }

        self._lifecycles: Dict[str, WallLifecycle] = {}
        self._event_log: list[WallObservation] = []
        self._pending_trades: list[Trade] = []
        self._last_ts_ms: Optional[int] = None
        self._next_wall_number = 1

        self._last_thresholds: Dict[Side, WallThreshold] = {
            "bid": WallThreshold("bid", 0, 0, 0, 0, 0),
            "ask": WallThreshold("ask", 0, 0, 0, 0, 0),
        }

    @property
    def lifecycles(self) -> Tuple[WallLifecycle, ...]:
        return tuple(self._lifecycles.values())

    @property
    def active_walls(self) -> Tuple[WallLifecycle, ...]:
        return tuple(
            wall
            for wall in self._lifecycles.values()
            if wall.status == "active"
        )

    @property
    def closed_walls(self) -> Tuple[WallLifecycle, ...]:
        return tuple(
            wall
            for wall in self._lifecycles.values()
            if wall.status == "closed"
        )

    @property
    def event_log(self) -> Tuple[WallObservation, ...]:
        return tuple(self._event_log)

    @property
    def last_thresholds(self) -> Dict[Side, WallThreshold]:
        return dict(self._last_thresholds)

    @property
    def book(self) -> Dict[Side, Dict[float, float]]:
        return {
            "bid": dict(self._book["bid"]),
            "ask": dict(self._book["ask"]),
        }

    def _validate_identity(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType,
    ) -> None:
        if exchange != self.exchange:
            raise ValueError("exchange does not match tracker")
        if symbol.upper() != self.symbol:
            raise ValueError("symbol does not match tracker")
        if market_type != self.market_type:
            raise ValueError("market_type does not match tracker")

    def apply_trade(self, trade: Trade) -> None:
        self._validate_identity(
            trade.exchange,
            trade.symbol,
            trade.market_type,
        )
        if trade.size <= self.config.epsilon:
            return
        self._pending_trades.append(trade)

    def apply_depth(
        self,
        update: DepthUpdate,
    ) -> Tuple[WallObservation, ...]:
        self._validate_identity(
            update.exchange,
            update.symbol,
            update.market_type,
        )

        if update.is_snapshot:
            self._book["bid"].clear()
            self._book["ask"].clear()

        self._apply_levels("bid", update.bids)
        self._apply_levels("ask", update.asks)

        return self._evaluate(update.exchange_ts)

    def _apply_levels(
        self,
        side: Side,
        levels: Iterable[Tuple[float, float]],
    ) -> None:
        book_side = self._book[side]
        for raw_price, raw_size in levels:
            price = float(raw_price)
            size = float(raw_size)
            if size <= self.config.epsilon:
                book_side.pop(price, None)
            else:
                book_side[price] = size

    def _ordered_levels(
        self,
        side: Side,
    ) -> list[Tuple[float, float]]:
        return sorted(
            self._book[side].items(),
            key=lambda item: item[0],
            reverse=side == "bid",
        )

    def _compute_threshold(self, side: Side) -> WallThreshold:
        local = self._ordered_levels(side)[
            : self.config.local_depth_levels
        ]
        local_total = sum(size for _, size in local)
        local_threshold = (
            local_total * self.config.relative_depth_fraction
        )

        history = self._history[side]
        historical_threshold = 0.0
        if len(history) >= self.config.min_history_observations:
            historical_threshold = _percentile(
                history,
                self.config.historical_percentile,
            )

        effective = max(
            self.config.min_absolute_size,
            local_threshold,
            historical_threshold,
        )

        return WallThreshold(
            side=side,
            local_depth_total=local_total,
            local_threshold=local_threshold,
            historical_threshold=historical_threshold,
            effective_threshold=effective,
            history_count=len(history),
        )

    def _candidate_levels(
        self,
        side: Side,
        threshold: WallThreshold,
    ) -> Dict[float, float]:
        candidates: Dict[float, float] = {}
        for price, size in self._ordered_levels(side)[
            : self.config.local_depth_levels
        ]:
            if (
                size + self.config.epsilon
                >= threshold.effective_threshold
            ):
                candidates[price] = size
        return candidates

    def _trade_pool(self, ts_ms: int) -> list[_TradeLot]:
        lower = (
            self._last_ts_ms
            if self._last_ts_ms is not None
            else -1
        )
        return [
            _TradeLot(trade=trade, remaining=float(trade.size))
            for trade in self._pending_trades
            if lower < trade.exchange_ts <= ts_ms
        ]

    def _consume_execution(
        self,
        side: Side,
        wall_price: float,
        requested_size: float,
        pool: list[_TradeLot],
    ) -> float:
        expected_trade_side = "sell" if side == "bid" else "buy"
        consumed = 0.0

        for lot in pool:
            if lot.remaining <= self.config.epsilon:
                continue
            if lot.trade.side != expected_trade_side:
                continue

            distance_bps = (
                abs(float(lot.trade.price) - wall_price)
                / wall_price
                * 10_000.0
            )
            if distance_bps > self.config.execution_match_bps:
                continue

            take = min(
                lot.remaining,
                requested_size - consumed,
            )
            lot.remaining -= take
            consumed += take

            if (
                consumed + self.config.epsilon
                >= requested_size
            ):
                break

        return consumed

    def _record(
        self,
        wall: WallLifecycle,
        event: WallEvent,
        ts_ms: int,
        *,
        price: Optional[float] = None,
        size: Optional[float] = None,
        previous_price: Optional[float] = None,
        previous_size: Optional[float] = None,
        executed_size: float = 0.0,
        pulled_size: float = 0.0,
        evidence: Tuple[str, ...] = (),
    ) -> WallObservation:
        observation = WallObservation(
            wall_id=wall.wall_id,
            event=event,
            exchange=wall.exchange,
            symbol=wall.symbol,
            market_type=wall.market_type,
            side=wall.side,
            ts_ms=ts_ms,
            price=wall.price if price is None else price,
            size=wall.current_size if size is None else size,
            previous_price=previous_price,
            previous_size=previous_size,
            executed_size=executed_size,
            pulled_size=pulled_size,
            evidence=evidence,
        )
        wall.observations.append(observation)
        self._event_log.append(observation)
        return observation

    def _threshold_evidence(
        self,
        threshold: WallThreshold,
    ) -> Tuple[str, ...]:
        return (
            f"local_depth_total={threshold.local_depth_total:.8g}",
            f"local_depth_threshold={threshold.local_threshold:.8g}",
            (
                "historical_percentile_threshold="
                f"{threshold.historical_threshold:.8g}"
            ),
            (
                "effective_threshold="
                f"{threshold.effective_threshold:.8g}"
            ),
            f"history_count={threshold.history_count}",
        )

    def _new_wall(
        self,
        side: Side,
        price: float,
        size: float,
        ts_ms: int,
        threshold: WallThreshold,
    ) -> Tuple[WallLifecycle, WallObservation]:
        wall_id = (
            f"{self.exchange}:{self.market_type}:"
            f"{self.symbol}:{side}:"
            f"{self._next_wall_number:06d}"
        )
        self._next_wall_number += 1

        wall = WallLifecycle(
            wall_id=wall_id,
            exchange=self.exchange,
            symbol=self.symbol,
            market_type=self.market_type,
            side=side,
            initial_price=price,
            initial_size=size,
            price=price,
            current_size=size,
            peak_size=size,
            first_seen_ms=ts_ms,
            last_seen_ms=ts_ms,
        )
        self._lifecycles[wall_id] = wall

        observation = self._record(
            wall,
            "detected",
            ts_ms,
            evidence=self._threshold_evidence(threshold),
        )
        return wall, observation

    def _maybe_mark_persisted(
        self,
        wall: WallLifecycle,
        ts_ms: int,
        events: list[WallObservation],
    ) -> None:
        if wall.persisted:
            return

        duration = ts_ms - wall.first_seen_ms
        if (
            wall.observation_count
            >= self.config.min_persistence_observations
            and duration >= self.config.min_persistence_ms
        ):
            wall.persisted = True
            events.append(
                self._record(
                    wall,
                    "persisted",
                    ts_ms,
                    evidence=(
                        f"observation_count={wall.observation_count}",
                        f"duration_ms={duration}",
                    ),
                )
            )

    def _observe_existing(
        self,
        wall: WallLifecycle,
        current_size: float,
        ts_ms: int,
        trade_pool: list[_TradeLot],
        events: list[WallObservation],
    ) -> None:
        previous_size = wall.current_size
        wall.observation_count += 1
        wall.last_seen_ms = ts_ms

        events.append(
            self._record(
                wall,
                "observed",
                ts_ms,
                size=current_size,
                previous_size=previous_size,
                evidence=("price_level_present",),
            )
        )

        if (
            current_size
            > previous_size + self.config.epsilon
        ):
            refill = current_size - previous_size
            refill_threshold = max(
                previous_size * self.config.refill_fraction,
                self.config.epsilon,
            )
            if refill + self.config.epsilon >= refill_threshold:
                events.append(
                    self._record(
                        wall,
                        "refilled",
                        ts_ms,
                        size=current_size,
                        previous_size=previous_size,
                        evidence=(
                            f"refill_size={refill:.8g}",
                            (
                                "refill_threshold="
                                f"{refill_threshold:.8g}"
                            ),
                        ),
                    )
                )

        elif (
            current_size
            < previous_size - self.config.epsilon
        ):
            removed = previous_size - current_size
            executed = self._consume_execution(
                wall.side,
                wall.price,
                removed,
                trade_pool,
            )
            pulled = max(0.0, removed - executed)

            if executed > self.config.epsilon:
                events.append(
                    self._record(
                        wall,
                        "partially_executed",
                        ts_ms,
                        size=current_size,
                        previous_size=previous_size,
                        executed_size=executed,
                        evidence=(
                            "matching_aggressive_trade",
                            f"removed_size={removed:.8g}",
                        ),
                    )
                )

            pull_threshold = max(
                previous_size
                * self.config.liquidity_pull_fraction,
                self.config.epsilon,
            )
            if (
                pulled > self.config.epsilon
                and pulled + self.config.epsilon
                >= pull_threshold
            ):
                events.append(
                    self._record(
                        wall,
                        "liquidity_pulled",
                        ts_ms,
                        size=current_size,
                        previous_size=previous_size,
                        pulled_size=pulled,
                        evidence=(
                            "removal_not_explained_by_trades",
                            (
                                "liquidity_pull_threshold="
                                f"{pull_threshold:.8g}"
                            ),
                        ),
                    )
                )

        wall.current_size = current_size
        wall.peak_size = max(wall.peak_size, current_size)
        self._maybe_mark_persisted(wall, ts_ms, events)

    def _find_reposition(
        self,
        wall: WallLifecycle,
        candidates: Dict[float, float],
        consumed: set[Tuple[Side, float]],
    ) -> Optional[Tuple[float, float, float, float]]:
        matches: list[Tuple[float, float, float, float]] = []

        for price, size in candidates.items():
            if (wall.side, price) in consumed:
                continue

            distance_bps = (
                abs(price - wall.price)
                / wall.price
                * 10_000.0
            )
            if distance_bps > self.config.reposition_max_bps:
                continue

            size_difference = (
                abs(size - wall.current_size)
                / max(wall.current_size, self.config.epsilon)
            )
            if (
                size_difference
                > self.config.reposition_size_tolerance
            ):
                continue

            matches.append(
                (
                    distance_bps,
                    size_difference,
                    price,
                    size,
                )
            )

        if not matches:
            return None

        distance_bps, size_difference, price, size = min(matches)
        return price, size, distance_bps, size_difference

    def _reposition(
        self,
        wall: WallLifecycle,
        match: Tuple[float, float, float, float],
        ts_ms: int,
        events: list[WallObservation],
    ) -> None:
        new_price, new_size, distance_bps, size_difference = match
        previous_price = wall.price
        previous_size = wall.current_size

        wall.observation_count += 1
        wall.last_seen_ms = ts_ms

        events.append(
            self._record(
                wall,
                "repositioned",
                ts_ms,
                price=new_price,
                size=new_size,
                previous_price=previous_price,
                previous_size=previous_size,
                evidence=(
                    f"distance_bps={distance_bps:.8g}",
                    f"relative_size_change={size_difference:.8g}",
                ),
            )
        )

        wall.price = new_price
        wall.current_size = new_size
        wall.peak_size = max(wall.peak_size, new_size)
        self._maybe_mark_persisted(wall, ts_ms, events)

    def _close_missing_wall(
        self,
        wall: WallLifecycle,
        ts_ms: int,
        trade_pool: list[_TradeLot],
        events: list[WallObservation],
    ) -> None:
        previous_size = wall.current_size
        executed = self._consume_execution(
            wall.side,
            wall.price,
            previous_size,
            trade_pool,
        )
        pulled = max(0.0, previous_size - executed)

        if executed > self.config.epsilon:
            execution_event: WallEvent = (
                "executed"
                if pulled <= self.config.epsilon
                else "partially_executed"
            )
            events.append(
                self._record(
                    wall,
                    execution_event,
                    ts_ms,
                    size=0.0,
                    previous_size=previous_size,
                    executed_size=executed,
                    evidence=(
                        "matching_aggressive_trade",
                        f"removed_size={previous_size:.8g}",
                    ),
                )
            )

        if pulled > self.config.epsilon:
            events.append(
                self._record(
                    wall,
                    "liquidity_pulled",
                    ts_ms,
                    size=0.0,
                    previous_size=previous_size,
                    pulled_size=pulled,
                    evidence=(
                        "removed_liquidity_not_explained_by_trades",
                    ),
                )
            )
            events.append(
                self._record(
                    wall,
                    "cancelled",
                    ts_ms,
                    size=0.0,
                    previous_size=previous_size,
                    pulled_size=pulled,
                    evidence=("price_level_removed",),
                )
            )

        wall.current_size = 0.0
        wall.status = "closed"
        wall.closed_at_ms = ts_ms

        if executed > self.config.epsilon and pulled <= self.config.epsilon:
            wall.closed_reason = "executed"
        elif executed > self.config.epsilon:
            wall.closed_reason = "mixed_execution_and_cancellation"
        else:
            wall.closed_reason = "cancelled"

    def _update_history(self) -> None:
        for side in ("bid", "ask"):
            for _, size in self._ordered_levels(side)[
                : self.config.local_depth_levels
            ]:
                self._history[side].append(size)

    def _evaluate(
        self,
        ts_ms: int,
    ) -> Tuple[WallObservation, ...]:
        if (
            self._last_ts_ms is not None
            and ts_ms < self._last_ts_ms
        ):
            raise ValueError("depth timestamps must be non-decreasing")

        events: list[WallObservation] = []
        trade_pool = self._trade_pool(ts_ms)

        thresholds = {
            "bid": self._compute_threshold("bid"),
            "ask": self._compute_threshold("ask"),
        }
        self._last_thresholds = thresholds

        candidates = {
            "bid": self._candidate_levels(
                "bid",
                thresholds["bid"],
            ),
            "ask": self._candidate_levels(
                "ask",
                thresholds["ask"],
            ),
        }

        consumed: set[Tuple[Side, float]] = set()
        missing: list[WallLifecycle] = []

        for wall in tuple(self.active_walls):
            current_size = self._book[wall.side].get(wall.price)
            if (
                current_size is not None
                and current_size > self.config.epsilon
            ):
                consumed.add((wall.side, wall.price))
                self._observe_existing(
                    wall,
                    current_size,
                    ts_ms,
                    trade_pool,
                    events,
                )
            else:
                missing.append(wall)

        for wall in missing:
            match = self._find_reposition(
                wall,
                candidates[wall.side],
                consumed,
            )
            if match is not None:
                self._reposition(wall, match, ts_ms, events)
                consumed.add((wall.side, match[0]))
            else:
                self._close_missing_wall(
                    wall,
                    ts_ms,
                    trade_pool,
                    events,
                )

        for side in ("bid", "ask"):
            ordered_candidates = sorted(
                candidates[side].items(),
                key=lambda item: item[0],
                reverse=side == "bid",
            )
            for price, size in ordered_candidates:
                if (side, price) in consumed:
                    continue
                _, observation = self._new_wall(
                    side,
                    price,
                    size,
                    ts_ms,
                    thresholds[side],
                )
                events.append(observation)
                consumed.add((side, price))

        self._update_history()
        self._pending_trades = [
            trade
            for trade in self._pending_trades
            if trade.exchange_ts > ts_ms
        ]
        self._last_ts_ms = ts_ms

        return tuple(events)

    def summary(self) -> MarketWallSummary:
        counts = Counter(
            observation.event
            for observation in self._event_log
        )
        return MarketWallSummary(
            exchange=self.exchange,
            symbol=self.symbol,
            market_type=self.market_type,
            active_wall_count=len(self.active_walls),
            closed_wall_count=len(self.closed_walls),
            event_counts=tuple(sorted(counts.items())),
        )


class WallTrackerEngine:
    """Route spot and futures events to independent comparable trackers."""

    def __init__(
        self,
        config: Optional[WallTrackerConfig] = None,
    ) -> None:
        self.config = config or WallTrackerConfig()
        self._trackers: Dict[
            Tuple[str, str, MarketType],
            WallTracker,
        ] = {}

    def tracker(
        self,
        exchange: str,
        symbol: str,
        market_type: MarketType,
    ) -> WallTracker:
        key = (exchange, symbol.upper(), market_type)
        tracker = self._trackers.get(key)
        if tracker is None:
            tracker = WallTracker(
                exchange=exchange,
                symbol=symbol,
                market_type=market_type,
                config=self.config,
            )
            self._trackers[key] = tracker
        return tracker

    def process_depth(
        self,
        update: DepthUpdate,
    ) -> Tuple[WallObservation, ...]:
        return self.tracker(
            update.exchange,
            update.symbol,
            update.market_type,
        ).apply_depth(update)

    def process_trade(self, trade: Trade) -> None:
        self.tracker(
            trade.exchange,
            trade.symbol,
            trade.market_type,
        ).apply_trade(trade)

    def summaries(self) -> Tuple[MarketWallSummary, ...]:
        return tuple(
            self._trackers[key].summary()
            for key in sorted(self._trackers)
        )


__all__ = [
    "MarketWallSummary",
    "WallLifecycle",
    "WallObservation",
    "WallThreshold",
    "WallTracker",
    "WallTrackerConfig",
    "WallTrackerEngine",
]
