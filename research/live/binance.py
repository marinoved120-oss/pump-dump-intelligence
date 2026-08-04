from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple

import httpx
from websockets.asyncio.client import connect

from .schemas import DepthUpdate, MarketType
from .sequence import SequenceApplier


class Transport(Protocol):
    async def rest_get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ...

    async def ws_listen(self, url: str) -> AsyncIterator[str]:
        ...

    async def close(self) -> None:  # optional for some transports
        ...


class BinanceProductionTransport:
    """Production HTTP and WebSocket transport for Binance collectors."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._timeout = timeout

    async def rest_get_json(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            payload = response.json()

        if not isinstance(payload, dict):
            raise TypeError("Binance REST response must be a JSON object")

        return payload

    async def ws_listen(self, url: str) -> AsyncIterator[str]:
        websocket = await connect(
            url,
            open_timeout=self._timeout,
            ping_interval=30.0,
            ping_timeout=30.0,
        )
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    yield message.decode("utf-8")
                else:
                    yield message
        finally:
            await websocket.close()

    async def close(self) -> None:
        # HTTP clients and WebSocket connections are scoped per operation.
        return None


class NDJSONWriter:
    def __init__(self, path: str):
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(self._path, "a", encoding="utf-8")

    def write(self, obj: Dict[str, Any]) -> None:
        line = json.dumps(obj, separators=(",", ":"))
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


@dataclass
class BookState:
    bids: Dict[float, float]
    asks: Dict[float, float]

    @staticmethod
    def empty() -> "BookState":
        return BookState(bids={}, asks={})


def _apply_snapshot(book: BookState, bids: List[List[str]] | List[Tuple[str, str]], asks: List[List[str]] | List[Tuple[str, str]]):
    book.bids.clear()
    book.asks.clear()
    for p, q in bids:
        price = float(p)
        qty = float(q)
        if qty > 0.0:
            book.bids[price] = qty
    for p, q in asks:
        price = float(p)
        qty = float(q)
        if qty > 0.0:
            book.asks[price] = qty


def _apply_deltas(book: BookState, bids: List[List[str]] | List[Tuple[str, str]], asks: List[List[str]] | List[Tuple[str, str]]):
    for p, q in bids:
        price = float(p)
        qty = float(q)
        if qty == 0.0:
            book.bids.pop(price, None)
        else:
            book.bids[price] = qty
    for p, q in asks:
        price = float(p)
        qty = float(q)
        if qty == 0.0:
            book.asks.pop(price, None)
        else:
            book.asks[price] = qty


class BinanceCollector:
    """Base collector for Binance spot/futures depth and public trades.

    - Independent per-market-type instance per symbol.
    - Snapshot bootstrap via REST depth.
    - Depth update ID validation using SequenceApplier with sequence=U.
    - Gap triggers resynchronization via fresh snapshot.
    - Durable NDJSON raw-event logging prior to aggregation.
    - Reconnect/backoff on WebSocket disconnects.
    """

    def __init__(
        self,
        exchange: str,
        market_type: MarketType,
        symbol: str,
        transport: Optional[Transport] = None,
        data_dir: Optional[str] = None,
        backoff_initial: float = 0.1,
        backoff_max: float = 2.0,
        depth_limit: int = 1000,
    ) -> None:
        self.exchange = exchange
        self.market_type = market_type
        self.symbol = symbol.upper()
        self._transport = transport or BinanceProductionTransport()
        self._depth_limit = depth_limit
        self._applier = SequenceApplier(require_snapshot_first=True)
        self._book = BookState.empty()
        if data_dir is None:
            base_dir = os.path.join("data", "live", self.exchange, self.market_type, self.symbol)
        else:
            base_dir = os.path.join(data_dir, "data", "live", self.exchange, self.market_type, self.symbol)
        self._depth_log = NDJSONWriter(os.path.join(base_dir, "depth.ndjson"))
        self._trades_log = NDJSONWriter(os.path.join(base_dir, "trades.ndjson"))
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._applied_events = 0

    def rest_depth_url(self) -> str:
        raise NotImplementedError

    def ws_depth_url(self) -> str:
        raise NotImplementedError

    def ws_trades_url(self) -> str:
        raise NotImplementedError

    @property
    def book(self) -> BookState:
        return self._book

    @property
    def applier(self) -> SequenceApplier:
        return self._applier

    async def _bootstrap_snapshot(self) -> None:
        params = {"symbol": self.symbol, "limit": self._depth_limit}
        snap = await self._transport.rest_get_json(self.rest_depth_url(), params=params)
        last_id = int(snap["lastUpdateId"])  # Binance snapshot field
        self._depth_log.write(
            {
                "type": "snapshot",
                "exchange": self.exchange,
                "market": self.market_type,
                "symbol": self.symbol,
                "lastUpdateId": last_id,
                "bids": snap.get("bids", []),
                "asks": snap.get("asks", []),
                "ts": int(time.time() * 1000),
            }
        )
        du = DepthUpdate(
            exchange=self.exchange,
            symbol=self.symbol,
            market_type=self.market_type,
            exchange_ts=int(time.time() * 1000),
            sequence=last_id,
            is_snapshot=True,
            bids=tuple((float(p), float(q)) for p, q in snap.get("bids", [])),
            asks=tuple((float(p), float(q)) for p, q in snap.get("asks", [])),
        )
        res = self._applier.apply_depth(du, now_ms=du.exchange_ts)
        if res.applied:
            _apply_snapshot(self._book, snap.get("bids", []), snap.get("asks", []))
            self._applied_events += 1

    async def _handle_depth_message(self, msg_obj: Dict[str, Any]) -> None:
        U = int(msg_obj.get("U"))
        u = int(msg_obj.get("u"))  # noqa: F841 - retained for completeness
        bids = msg_obj.get("b", [])
        asks = msg_obj.get("a", [])
        exch_ts = int(msg_obj.get("E", int(time.time() * 1000)))

        self._depth_log.write(
            {
                "type": "depth",
                "exchange": self.exchange,
                "market": self.market_type,
                "symbol": self.symbol,
                "U": U,
                "u": int(msg_obj.get("u", 0)),
                "b": bids,
                "a": asks,
                "ts": exch_ts,
            }
        )

        du = DepthUpdate(
            exchange=self.exchange,
            symbol=self.symbol,
            market_type=self.market_type,
            exchange_ts=exch_ts,
            sequence=U,
            is_snapshot=False,
            bids=tuple((float(p), float(q)) for p, q in bids),
            asks=tuple((float(p), float(q)) for p, q in asks),
        )
        res = self._applier.apply_depth(du, now_ms=exch_ts)
        if res.applied:
            _apply_deltas(self._book, bids, asks)
            self._applied_events += 1
        else:
            if res.reason in ("gap", "requires_snapshot", "out_of_sync"):
                await self._bootstrap_snapshot()

    async def _handle_trade_message(self, msg_obj: Dict[str, Any]) -> None:
        exch_ts = int(msg_obj.get("E", int(time.time() * 1000)))
        side = "unknown"
        if "m" in msg_obj:
            side = "sell" if bool(msg_obj.get("m")) else "buy"
        rec: Dict[str, Any] = {
            "type": "trade",
            "exchange": self.exchange,
            "market": self.market_type,
            "symbol": self.symbol,
            "trade_id": str(msg_obj.get("t", "")),
            "price": msg_obj.get("p"),
            "size": msg_obj.get("q"),
            "side": side,
            "ts": exch_ts,
        }
        self._trades_log.write(rec)

    async def _depth_loop(self, stop_event: asyncio.Event, max_depth_events: Optional[int]) -> None:
        backoff = self._backoff_initial
        while not stop_event.is_set():
            try:
                await self._bootstrap_snapshot()
                url = self.ws_depth_url()
                async for raw in self._transport.ws_listen(url):
                    if stop_event.is_set():
                        break
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    await self._handle_depth_message(obj)
                    if max_depth_events is not None and self._applied_events >= max_depth_events:
                        stop_event.set()
                        break
                await asyncio.sleep(min(backoff, self._backoff_max))
                backoff = min(self._backoff_max, backoff * 2)
            except Exception:
                await asyncio.sleep(min(backoff, self._backoff_max))
                backoff = min(self._backoff_max, backoff * 2)
                continue

    async def _trades_loop(self, stop_event: asyncio.Event) -> None:
        backoff = self._backoff_initial
        while not stop_event.is_set():
            try:
                url = self.ws_trades_url()
                async for raw in self._transport.ws_listen(url):
                    if stop_event.is_set():
                        break
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    self._trades_log.write(
                        {
                            "type": "raw_trade",
                            "exchange": self.exchange,
                            "market": self.market_type,
                            "symbol": self.symbol,
                            "payload": obj,
                            "ts": int(time.time() * 1000),
                        }
                    )
                    await self._handle_trade_message(obj)
                await asyncio.sleep(min(backoff, self._backoff_max))
                backoff = min(self._backoff_max, backoff * 2)
            except Exception:
                await asyncio.sleep(min(backoff, self._backoff_max))
                backoff = min(self._backoff_max, backoff * 2)
                continue

    async def run_for(self, max_depth_events: int, run_trades: bool = True) -> None:
        """Run collector until a number of applied depth events (including snapshots) is reached.

        Intended for tests and controlled runs. Trades stream runs concurrently by default.
        """
        if self._running:
            raise RuntimeError("already running")
        self._running = True
        stop_event = asyncio.Event()
        self._applied_events = 0
        tasks: List[asyncio.Task] = []
        tasks.append(asyncio.create_task(self._depth_loop(stop_event, max_depth_events)))
        if run_trades:
            tasks.append(asyncio.create_task(self._trades_loop(stop_event)))
        self._tasks = tasks
        try:
            await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        finally:
            stop_event.set()
            for t in tasks:
                if not t.done():
                    t.cancel()
            self._running = False

    async def shutdown(self) -> None:
        try:
            for t in self._tasks:
                if not t.done():
                    t.cancel()
        finally:
            self._tasks = []
            self._depth_log.close()
            self._trades_log.close()
            try:
                await self._transport.close()
            except Exception:
                pass


class BinanceSpotCollector(BinanceCollector):
    def __init__(
        self,
        symbol: str,
        transport: Optional[Transport] = None,
        data_dir: Optional[str] = None,
        backoff_initial: float = 0.05,
        backoff_max: float = 1.0,
        depth_limit: int = 1000,
    ) -> None:
        super().__init__(
            exchange="binance",
            market_type="spot",
            symbol=symbol,
            transport=transport,
            data_dir=data_dir,
            backoff_initial=backoff_initial,
            backoff_max=backoff_max,
            depth_limit=depth_limit,
        )

    def rest_depth_url(self) -> str:
        return "https://api.binance.com/api/v3/depth"

    def ws_depth_url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@depth@100ms"

    def ws_trades_url(self) -> str:
        return f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"


class BinanceFuturesCollector(BinanceCollector):
    def __init__(
        self,
        symbol: str,
        transport: Optional[Transport] = None,
        data_dir: Optional[str] = None,
        backoff_initial: float = 0.05,
        backoff_max: float = 1.0,
        depth_limit: int = 1000,
    ) -> None:
        super().__init__(
            exchange="binance",
            market_type="futures",
            symbol=symbol,
            transport=transport,
            data_dir=data_dir,
            backoff_initial=backoff_initial,
            backoff_max=backoff_max,
            depth_limit=depth_limit,
        )

    def rest_depth_url(self) -> str:
        return "https://fapi.binance.com/fapi/v1/depth"

    def ws_depth_url(self) -> str:
        return f"wss://fstream.binance.com/ws/{self.symbol.lower()}@depth@100ms"

    def ws_trades_url(self) -> str:
        return f"wss://fstream.binance.com/ws/{self.symbol.lower()}@trade"


__all__ = [
    "Transport",
    "BinanceProductionTransport",
    "NDJSONWriter",
    "BookState",
    "BinanceCollector",
    "BinanceSpotCollector",
    "BinanceFuturesCollector",
]

"""
Implementation notes:
- SequenceApplier enforces strict continuity on U (first update ID). Any skip triggers resnapshot.
- Durable logs allow deterministic replay and integrity auditing without secrets.
"""
