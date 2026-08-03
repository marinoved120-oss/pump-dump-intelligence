from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Iterable
import difflib
import re

import httpx
import pandas as pd

from research.config import BinanceConfig


class BinanceDataError(RuntimeError):
    """Raised when Binance public market data cannot be retrieved safely."""


@dataclass(frozen=True)
class DateRange:
    start_ms: int
    end_ms: int

    @classmethod
    def recent_days(cls, days: int) -> "DateRange":
        if days < 1:
            raise ValueError("days must be >= 1")
        end = datetime.now(tz=UTC)
        start = end - timedelta(days=days)
        return cls(int(start.timestamp() * 1000), int(end.timestamp() * 1000))

    def clamp_to_recent_days(self, days: int, safety_minutes: int = 10) -> "DateRange":
        """Clamp a request to a provider's trailing-history limit.

        Binance documents only the latest month for open-interest statistics.
        A small safety margin avoids boundary failures caused by server/client
        clock differences and the ambiguous length of a calendar month.
        """
        if days < 1:
            raise ValueError("days must be >= 1")
        maximum_span = timedelta(days=days) - timedelta(minutes=safety_minutes)
        minimum_start = self.end_ms - int(maximum_span.total_seconds() * 1000)
        return DateRange(max(self.start_ms, minimum_start), self.end_ms)


class BinanceFuturesClient:
    """Public USD-M Futures REST client with retries and pagination.

    No API key is used. The implementation intentionally limits itself to public
    market-data endpoints.
    """

    def __init__(self, config: BinanceConfig) -> None:
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            headers={"User-Agent": "pump-dump-research/0.2"},
        )

    async def __aenter__(self) -> "BinanceFuturesClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self._client.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.get(path, params=params)
                if response.status_code in {418, 429}:
                    retry_after = float(response.headers.get("Retry-After", "2"))
                    await asyncio.sleep(max(retry_after, 1.0))
                    continue
                if response.status_code >= 400:
                    try:
                        detail: Any = response.json()
                    except ValueError:
                        detail = response.text[:500]
                    raise BinanceDataError(
                        f"HTTP {response.status_code} for {path}; response: {detail}"
                    )
                payload = response.json()
                if isinstance(payload, dict) and "code" in payload and int(payload["code"]) < 0:
                    raise BinanceDataError(f"Binance error: {payload}")
                return payload
            except (httpx.HTTPError, ValueError, BinanceDataError) as exc:
                last_error = exc
                if attempt + 1 >= self.config.max_retries:
                    break
                delay = self.config.backoff_seconds * (2**attempt) + random.random() * 0.25
                await asyncio.sleep(delay)
        raise BinanceDataError(f"Request failed: {path}; last error: {last_error}")

    async def exchange_info(self) -> dict[str, Any]:
        payload = await self._get("/fapi/v1/exchangeInfo")
        if not isinstance(payload, dict):
            raise BinanceDataError("Unexpected exchangeInfo response")
        return payload

    async def active_usdt_perpetuals(self) -> list[str]:
        payload = await self.exchange_info()
        symbols: list[str] = []
        for item in payload.get("symbols", []):
            if (
                item.get("status") == "TRADING"
                and item.get("contractType") == "PERPETUAL"
                and item.get("quoteAsset") == "USDT"
            ):
                symbols.append(str(item["symbol"]))
        return sorted(symbols)

    async def top_symbols(self, limit: int = 30) -> pd.DataFrame:
        active = set(await self.active_usdt_perpetuals())
        tickers = await self._get("/fapi/v1/ticker/24hr")
        rows = []
        for item in tickers:
            symbol = str(item.get("symbol", ""))
            if symbol not in active:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "last_price": float(item.get("lastPrice", 0.0)),
                    "price_change_pct": float(item.get("priceChangePercent", 0.0)),
                    "quote_volume": float(item.get("quoteVolume", 0.0)),
                    "trades": int(item.get("count", 0)),
                }
            )
        return (
            pd.DataFrame(rows)
            .sort_values("quote_volume", ascending=False)
            .head(limit)
            .reset_index(drop=True)
        )

    async def klines(
        self,
        symbol: str,
        interval: str,
        date_range: DateRange,
        limit: int = 1500,
    ) -> pd.DataFrame:
        rows: list[list[Any]] = []
        cursor = date_range.start_ms
        while cursor < date_range.end_ms:
            batch = await self._get(
                "/fapi/v1/klines",
                {
                    "symbol": symbol.upper(),
                    "interval": interval,
                    "startTime": cursor,
                    "endTime": date_range.end_ms,
                    "limit": limit,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1][6]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < limit:
                break
            await asyncio.sleep(0.03)

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume_base",
            "close_time",
            "volume_quote",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        if frame.empty:
            raise BinanceDataError(f"No klines returned for {symbol}")
        numeric = [column for column in columns if column not in {"open_time", "close_time", "ignore"}]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        frame["symbol"] = symbol.upper()
        return frame.drop(columns=["ignore"]).drop_duplicates("open_time").sort_values("open_time")

    async def funding_rates(self, symbol: str, date_range: DateRange) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        cursor = date_range.start_ms
        while cursor < date_range.end_ms:
            batch = await self._get(
                "/fapi/v1/fundingRate",
                {
                    "symbol": symbol.upper(),
                    "startTime": cursor,
                    "endTime": date_range.end_ms,
                    "limit": 1000,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1]["fundingTime"]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
            await asyncio.sleep(0.03)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
        frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
        frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
        frame["mark_price"] = pd.to_numeric(frame.get("markPrice"), errors="coerce")
        return frame[["timestamp", "funding_rate", "mark_price"]].drop_duplicates("timestamp")

    async def open_interest_history(
        self,
        symbol: str,
        date_range: DateRange,
        period: str = "5m",
    ) -> pd.DataFrame:
        # Binance exposes only approximately the latest month for this endpoint.
        # Keep 29 days plus a small safety margin to prevent HTTP 400 at the edge.
        effective_range = date_range.clamp_to_recent_days(29)
        rows: list[dict[str, Any]] = []
        cursor = effective_range.start_ms
        while cursor < effective_range.end_ms:
            batch = await self._get(
                "/futures/data/openInterestHist",
                {
                    "symbol": symbol.upper(),
                    "period": period,
                    "startTime": cursor,
                    "endTime": effective_range.end_ms,
                    "limit": 500,
                },
            )
            if not batch:
                break
            rows.extend(batch)
            next_cursor = int(batch[-1]["timestamp"]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 500:
                break
            await asyncio.sleep(0.03)
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "open_interest", "open_interest_value"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        frame["open_interest"] = pd.to_numeric(frame["sumOpenInterest"], errors="coerce")
        frame["open_interest_value"] = pd.to_numeric(
            frame["sumOpenInterestValue"], errors="coerce"
        )
        return frame[["timestamp", "open_interest", "open_interest_value"]].drop_duplicates(
            "timestamp"
        )


async def collect_symbol(
    config: BinanceConfig,
    symbol: str,
    days: int,
    interval: str = "1m",
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Collect required candles and best-effort derivative histories.

    Candles are mandatory. Funding and open-interest histories are optional:
    if Binance rejects one auxiliary endpoint, the research dataset is still
    saved and the missing feature family is represented by NaNs later.
    """
    date_range = DateRange.recent_days(days)
    notices: list[str] = []
    async with BinanceFuturesClient(config) as client:
        results = await asyncio.gather(
            client.klines(symbol, interval, date_range),
            client.funding_rates(symbol, date_range),
            client.open_interest_history(symbol, date_range),
            return_exceptions=True,
        )

    klines_result, funding_result, oi_result = results
    if isinstance(klines_result, Exception):
        raise BinanceDataError(f"Required kline collection failed: {klines_result}")

    if isinstance(funding_result, Exception):
        notices.append(f"Funding history unavailable: {funding_result}")
        funding = pd.DataFrame(columns=["timestamp", "funding_rate", "mark_price"])
    else:
        funding = funding_result

    if isinstance(oi_result, Exception):
        notices.append(f"Open-interest history unavailable: {oi_result}")
        open_interest = pd.DataFrame(
            columns=["timestamp", "open_interest", "open_interest_value"]
        )
    else:
        open_interest = oi_result

    if days > 29:
        notices.append(
            "Binance open-interest statistics are limited to the latest month; "
            "candles/funding cover the requested period, while OI covers about 29 days."
        )

    return {
        "klines": klines_result,
        "funding": funding,
        "open_interest": open_interest,
    }, notices


def _clean_symbol(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper().strip())


def resolve_symbol(value: str, active: Iterable[str]) -> str:
    """Resolve user-friendly asset names to Binance USD-M contract symbols.

    Binance uses quantity-prefixed contract names for some assets, for example
    PEPE -> 1000PEPEUSDT. The resolver prefers an exact contract, then a single
    numeric-prefix variant. Ambiguous matches are rejected rather than guessed.
    """
    active_set = {_clean_symbol(symbol) for symbol in active}
    requested = _clean_symbol(value)
    if not requested:
        raise BinanceDataError("Empty symbol")

    direct = requested if requested.endswith("USDT") else f"{requested}USDT"
    if direct in active_set:
        return direct

    base = requested[:-4] if requested.endswith("USDT") else requested
    scaled = sorted(
        symbol
        for symbol in active_set
        if re.fullmatch(rf"\d+{re.escape(base)}USDT", symbol)
    )
    if len(scaled) == 1:
        return scaled[0]
    if len(scaled) > 1:
        raise BinanceDataError(
            f"Ambiguous Binance Futures symbol {value!r}. Candidates: {', '.join(scaled)}"
        )

    close = difflib.get_close_matches(direct, sorted(active_set), n=5, cutoff=0.45)
    suggestion = f" Closest active contracts: {', '.join(close)}." if close else ""
    raise BinanceDataError(
        f"Not an active USD-M USDT perpetual: {direct}.{suggestion} "
        "Run the symbols command to inspect active contracts."
    )


def validate_symbols(requested: Iterable[str], active: Iterable[str]) -> list[str]:
    active_list = list(active)
    return [resolve_symbol(value, active_list) for value in requested]
