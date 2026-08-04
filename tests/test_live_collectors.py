import asyncio
import json
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest



@pytest.fixture
def anyio_backend():
    return "asyncio"

from research.live.binance import (
    BinanceSpotCollector,
    BinanceFuturesCollector,
    BookState,
)
from research.live.sequence import SequenceApplier


class MockTransport:
    def __init__(self):
        self._ws_payloads: Dict[str, List[Any]] = {}
        self._rest_payloads: Dict[str, List[Dict[str, Any]]] = {}

    def add_rest_response(self, url: str, params: Dict[str, Any], resp: Dict[str, Any]):
        key = self._rest_key(url, params)
        self._rest_payloads.setdefault(key, []).append(resp)

    def _rest_key(self, url: str, params: Optional[Dict[str, Any]]):
        p = params or {}
        return url + "?" + json.dumps(sorted(p.items()))

    def add_ws_messages(self, url: str, messages: List[Any]):
        self._ws_payloads.setdefault(url, []).extend(messages)

    async def rest_get_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        key = self._rest_key(url, params)
        if key not in self._rest_payloads or not self._rest_payloads[key]:
            raise RuntimeError(f"No REST payload for {key}")
        return self._rest_payloads[key].pop(0)

    async def ws_listen(self, url: str) -> AsyncIterator[str]:
        if url not in self._ws_payloads:
            if False:
                yield ""  # pragma: no cover
            return
        payloads = self._ws_payloads[url]
        while payloads:
            item = payloads.pop(0)
            if isinstance(item, Exception):
                raise item
            yield json.dumps(item)

    async def close(self) -> None:
        pass


def reconstruct_book_from_depth_log(path: str) -> BookState:
    applier = SequenceApplier(require_snapshot_first=True)
    book = BookState.empty()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("type") == "snapshot":
                last_id = int(obj["lastUpdateId"])  # sequence
                du = {
                    "exchange": obj["exchange"],
                    "symbol": obj["symbol"],
                    "market_type": obj["market"],
                    "exchange_ts": obj.get("ts", 0),
                    "sequence": last_id,
                    "is_snapshot": True,
                    "bids": tuple((float(p), float(q)) for p, q in obj.get("bids", [])),
                    "asks": tuple((float(p), float(q)) for p, q in obj.get("asks", [])),
                }
                from research.live.schemas import DepthUpdate as DU

                res = applier.apply_depth(DU(**du), now_ms=du["exchange_ts"])  # type: ignore[arg-type]
                if res.applied:
                    book.bids = {float(p): float(q) for p, q in obj.get("bids", []) if float(q) > 0}
                    book.asks = {float(p): float(q) for p, q in obj.get("asks", []) if float(q) > 0}
            elif obj.get("type") == "depth":
                U = int(obj["U"])  # sequence
                bids = obj.get("b", [])
                asks = obj.get("a", [])
                from research.live.schemas import DepthUpdate as DU

                du = DU(
                    exchange=obj["exchange"],
                    symbol=obj["symbol"],
                    market_type=obj["market"],
                    exchange_ts=obj.get("ts", 0),
                    sequence=U,
                    is_snapshot=False,
                    bids=tuple((float(p), float(q)) for p, q in bids),
                    asks=tuple((float(p), float(q)) for p, q in asks),
                )
                res = applier.apply_depth(du, now_ms=du.exchange_ts)
                if res.applied:
                    for p, q in bids:
                        p = float(p)
                        q = float(q)
                        if q == 0.0:
                            book.bids.pop(p, None)
                        else:
                            book.bids[p] = q
                    for p, q in asks:
                        p = float(p)
                        q = float(q)
                        if q == 0.0:
                            book.asks.pop(p, None)
                        else:
                            book.asks[p] = q
    return book


@pytest.mark.anyio
@pytest.mark.parametrize(
    "collector_cls,rest_url,ws_base",
    [
        (BinanceSpotCollector, "https://api.binance.com/api/v3/depth", "wss://stream.binance.com:9443/ws"),
        (BinanceFuturesCollector, "https://fapi.binance.com/fapi/v1/depth", "wss://fstream.binance.com/ws"),
    ],
)
async def test_connect_bootstrap_process_and_replay(collector_cls, rest_url, ws_base, tmp_path):
    symbol = "BTCUSDT"
    t = MockTransport()

    t.add_rest_response(
        rest_url,
        {"symbol": symbol, "limit": 1000},
        {
            "lastUpdateId": 100,
            "bids": [["100.0", "1.0"]],
            "asks": [["101.0", "2.0"]],
        },
    )

    depth_url = f"{ws_base}/{symbol.lower()}@depth@100ms"
    t.add_ws_messages(
        depth_url,
        [
            {"e": "depthUpdate", "E": 1, "s": symbol, "U": 101, "u": 101, "b": [["100.0", "2.0"]], "a": []},
            {"e": "depthUpdate", "E": 2, "s": symbol, "U": 102, "u": 102, "b": [["99.5", "1.0"]], "a": [["101.0", "0"]]},
        ],
    )

    trades_url = f"{ws_base}/{symbol.lower()}@trade"
    t.add_ws_messages(
        trades_url,
        [
            {"e": "trade", "E": 1, "s": symbol, "t": 1, "p": "100.0", "q": "0.1", "m": True},
            {"e": "trade", "E": 2, "s": symbol, "t": 2, "p": "99.9", "q": "0.2", "m": False},
        ],
    )

    data_dir = tmp_path.as_posix()
    collector = collector_cls(symbol, t, data_dir=data_dir, backoff_initial=0.0, backoff_max=0.0)
    await collector.run_for(max_depth_events=3, run_trades=True)
    await collector.shutdown()

    assert collector.book.bids.get(100.0) == 2.0
    assert collector.book.bids.get(99.5) == 1.0
    assert 101.0 not in collector.book.asks

    depth_log = os.path.join(
        data_dir,
        "data",
        "live",
        "binance",
        "spot" if collector_cls is BinanceSpotCollector else "futures",
        symbol,
        "depth.ndjson",
    )
    assert os.path.exists(depth_log)

    replayed = reconstruct_book_from_depth_log(depth_log)
    assert replayed.bids == collector.book.bids
    assert replayed.asks == collector.book.asks


@pytest.mark.anyio
@pytest.mark.parametrize(
    "collector_cls,rest_url,ws_base",
    [
        (BinanceSpotCollector, "https://api.binance.com/api/v3/depth", "wss://stream.binance.com:9443/ws"),
        (BinanceFuturesCollector, "https://fapi.binance.com/fapi/v1/depth", "wss://fstream.binance.com/ws"),
    ],
)
async def test_gap_triggers_resnapshot_and_recovery(collector_cls, rest_url, ws_base, tmp_path):
    symbol = "ETHUSDT"
    t = MockTransport()

    t.add_rest_response(
        rest_url,
        {"symbol": symbol, "limit": 1000},
        {
            "lastUpdateId": 200,
            "bids": [["200.0", "1.0"]],
            "asks": [["201.0", "1.0"]],
        },
    )
    t.add_rest_response(
        rest_url,
        {"symbol": symbol, "limit": 1000},
        {
            "lastUpdateId": 205,
            "bids": [["200.0", "2.0"], ["199.5", "1.0"]],
            "asks": [["201.5", "1.0"]],
        },
    )

    depth_url = f"{ws_base}/{symbol.lower()}@depth@100ms"
    t.add_ws_messages(
        depth_url,
        [
            {"e": "depthUpdate", "E": 10, "s": symbol, "U": 201, "u": 201, "b": [["200.0", "1.5"]], "a": []},
            {"e": "depthUpdate", "E": 11, "s": symbol, "U": 205, "u": 205, "b": [["199.0", "1.0"]], "a": []},
            {"e": "depthUpdate", "E": 12, "s": symbol, "U": 206, "u": 206, "b": [["200.0", "2.5"]], "a": [["201.5", "0"]]},
        ],
    )

    trades_url = f"{ws_base}/{symbol.lower()}@trade"
    t.add_ws_messages(trades_url, [])

    data_dir = tmp_path.as_posix()
    collector = collector_cls(symbol, t, data_dir=data_dir, backoff_initial=0.0, backoff_max=0.0)
    await collector.run_for(max_depth_events=4, run_trades=False)
    await collector.shutdown()

    assert collector.book.bids.get(200.0) == 2.5
    assert collector.book.bids.get(199.5) == 1.0
    assert 201.5 not in collector.book.asks
    assert collector.applier.state.gap_count >= 1 or "sequence_gap" in collector.applier.state.issues


@pytest.mark.anyio
@pytest.mark.parametrize(
    "collector_cls,rest_url,ws_base",
    [
        (BinanceSpotCollector, "https://api.binance.com/api/v3/depth", "wss://stream.binance.com:9443/ws"),
        (BinanceFuturesCollector, "https://fapi.binance.com/fapi/v1/depth", "wss://fstream.binance.com/ws"),
    ],
)
async def test_disconnect_reconnect_with_resnapshot(collector_cls, rest_url, ws_base, tmp_path):
    symbol = "BNBUSDT"
    t = MockTransport()

    t.add_rest_response(
        rest_url,
        {"symbol": symbol, "limit": 1000},
        {
            "lastUpdateId": 300,
            "bids": [["300.0", "1.0"]],
            "asks": [["301.0", "1.0"]],
        },
    )
    t.add_rest_response(
        rest_url,
        {"symbol": symbol, "limit": 1000},
        {
            "lastUpdateId": 302,
            "bids": [["300.0", "1.5"]],
            "asks": [["301.0", "1.0"]],
        },
    )

    depth_url = f"{ws_base}/{symbol.lower()}@depth@100ms"
    t.add_ws_messages(
        depth_url,
        [
            {"e": "depthUpdate", "E": 20, "s": symbol, "U": 301, "u": 301, "b": [["300.0", "2.0"]], "a": []},
            RuntimeError("disconnect"),
            {"e": "depthUpdate", "E": 21, "s": symbol, "U": 303, "u": 303, "b": [["299.0", "1.0"]], "a": [["301.0", "0"]]},
        ],
    )

    trades_url = f"{ws_base}/{symbol.lower()}@trade"
    t.add_ws_messages(trades_url, [])

    data_dir = tmp_path.as_posix()
    collector = collector_cls(symbol, t, data_dir=data_dir, backoff_initial=0.0, backoff_max=0.0)
    await collector.run_for(max_depth_events=4, run_trades=False)
    await collector.shutdown()

    assert collector.book.bids.get(299.0) == 1.0
    assert collector.book.bids.get(300.0) == 1.5
    assert 301.0 not in collector.book.asks
    depth_log = os.path.join(
        data_dir,
        "data",
        "live",
        "binance",
        "spot" if collector_cls is BinanceSpotCollector else "futures",
        symbol,
        "depth.ndjson",
    )
    trades_log = os.path.join(
        data_dir,
        "data",
        "live",
        "binance",
        "spot" if collector_cls is BinanceSpotCollector else "futures",
        symbol,
        "trades.ndjson",
    )
    assert os.path.exists(depth_log)
    assert os.path.exists(trades_log)


@pytest.mark.anyio
async def test_collectors_default_to_production_transport(tmp_path):
    import research.live.binance as module

    spot = module.BinanceSpotCollector(
        "BTCUSDT",
        data_dir=str(tmp_path / "spot"),
    )
    futures = module.BinanceFuturesCollector(
        "BTCUSDT",
        data_dir=str(tmp_path / "futures"),
    )

    try:
        assert isinstance(
            spot._transport,
            module.BinanceProductionTransport,
        )
        assert isinstance(
            futures._transport,
            module.BinanceProductionTransport,
        )
    finally:
        await spot.shutdown()
        await futures.shutdown()


@pytest.mark.anyio
async def test_production_transport_uses_websocket_and_http(
    monkeypatch,
):
    import research.live.binance as module

    calls = {}

    class FakeWebSocket:
        def __init__(self):
            self.messages = iter(
                [
                    '{"event":"text"}',
                    b'{"event":"binary"}',
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.messages)
            except StopIteration:
                raise StopAsyncIteration

        async def close(self):
            calls["websocket_closed"] = True

    async def fake_connect(url, **kwargs):
        calls["websocket_url"] = url
        calls["websocket_kwargs"] = kwargs
        return FakeWebSocket()

    class FakeResponse:
        def raise_for_status(self):
            calls["status_checked"] = True

        def json(self):
            return {"lastUpdateId": 123}

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client_kwargs"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None):
            calls["http_url"] = url
            calls["http_params"] = params
            return FakeResponse()

    monkeypatch.setattr(module, "connect", fake_connect)
    monkeypatch.setattr(module.httpx, "AsyncClient", FakeClient)

    transport = module.BinanceProductionTransport(timeout=7.0)

    messages = [
        message
        async for message in transport.ws_listen(
            "wss://example.test/ws"
        )
    ]

    payload = await transport.rest_get_json(
        "https://example.test/depth",
        {"symbol": "BTCUSDT"},
    )

    assert messages == [
        '{"event":"text"}',
        '{"event":"binary"}',
    ]
    assert payload == {"lastUpdateId": 123}

    assert calls["websocket_url"] == "wss://example.test/ws"
    assert calls["websocket_kwargs"]["open_timeout"] == 7.0
    assert calls["websocket_kwargs"]["ping_interval"] == 30.0
    assert calls["websocket_closed"] is True

    assert calls["http_url"] == "https://example.test/depth"
    assert calls["http_params"] == {"symbol": "BTCUSDT"}
    assert calls["client_kwargs"]["timeout"] == 7.0
    assert calls["status_checked"] is True

