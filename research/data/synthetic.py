from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    days: int = 30
    seed: int = 42
    symbol: str = "SYNTHUSDT"
    event_spacing_hours: int = 30


def generate_synthetic_market(config: SyntheticConfig = SyntheticConfig()) -> pd.DataFrame:
    """Generate minute data containing repeated pump, distribution and dump regimes."""
    rng = np.random.default_rng(config.seed)
    periods = config.days * 24 * 60
    timestamp = pd.date_range("2025-01-01", periods=periods, freq="min", tz="UTC")

    log_returns = rng.normal(0.0, 0.00055, periods)
    baseline_volume = rng.lognormal(mean=10.4, sigma=0.45, size=periods)
    trade_count = rng.poisson(85, periods).astype(float)
    taker_buy_ratio = np.clip(rng.normal(0.50, 0.055, periods), 0.20, 0.80)
    open_interest = 50_000_000 + np.cumsum(rng.normal(0, 18_000, periods))
    funding = np.clip(rng.normal(0.00008, 0.00004, periods), -0.0002, 0.0004)

    event_starts = range(24 * 60, periods - 180, config.event_spacing_hours * 60)
    for start in event_starts:
        pump_minutes = int(rng.integers(18, 45))
        plateau_minutes = int(rng.integers(8, 25))
        dump_minutes = int(rng.integers(12, 35))
        amplitude = float(rng.uniform(0.22, 0.65))
        event_type = float(rng.random())
        if event_type < 0.30:
            # Pump that fades sideways: a required negative class for dump models.
            dump_fraction = float(rng.uniform(0.01, 0.035))
        elif event_type < 0.60:
            # Ordinary post-pump correction.
            dump_fraction = float(rng.uniform(0.04, 0.075))
        else:
            dump_fraction = float(rng.uniform(0.10, min(amplitude * 0.85, 0.35)))

        pump = slice(start, start + pump_minutes)
        plateau = slice(start + pump_minutes, start + pump_minutes + plateau_minutes)
        dump = slice(
            start + pump_minutes + plateau_minutes,
            start + pump_minutes + plateau_minutes + dump_minutes,
        )

        log_returns[pump] += np.log1p(amplitude) / pump_minutes
        log_returns[plateau] += rng.normal(0, 0.0015, plateau_minutes)
        log_returns[dump] -= abs(np.log1p(-dump_fraction)) / dump_minutes

        baseline_volume[pump] *= rng.uniform(7, 16)
        baseline_volume[plateau] *= rng.uniform(5, 11)
        baseline_volume[dump] *= rng.uniform(9, 20)
        trade_count[pump] *= rng.uniform(4, 8)
        trade_count[plateau] *= rng.uniform(3, 6)
        trade_count[dump] *= rng.uniform(5, 10)
        taker_buy_ratio[pump] = np.clip(rng.normal(0.76, 0.04, pump_minutes), 0.55, 0.95)
        taker_buy_ratio[plateau] = np.linspace(0.62, 0.40, plateau_minutes)
        taker_buy_ratio[dump] = np.clip(rng.normal(0.26, 0.05, dump_minutes), 0.05, 0.45)

        open_interest[pump] += np.linspace(0, 6_000_000, pump_minutes)
        open_interest[plateau] += 6_000_000 + np.linspace(0, 2_000_000, plateau_minutes)
        open_interest[dump] += np.linspace(8_000_000, -2_000_000, dump_minutes)
        funding[pump] += np.linspace(0.00005, 0.00035, pump_minutes)
        funding[plateau] += 0.00035

    close = 1.0 * np.exp(np.cumsum(log_returns))
    open_price = np.r_[close[0], close[:-1]]
    candle_noise = np.abs(rng.normal(0, 0.0012, periods))
    high = np.maximum(open_price, close) * (1 + candle_noise)
    low = np.minimum(open_price, close) * (1 - candle_noise)
    volume_base = baseline_volume / np.maximum(close, 1e-9)
    taker_buy_quote = baseline_volume * taker_buy_ratio

    frame = pd.DataFrame(
        {
            "timestamp": timestamp,
            "open_time": (timestamp.astype("int64") // 1_000_000).astype("int64"),
            "close_time": ((timestamp + pd.Timedelta(minutes=1)).astype("int64") // 1_000_000).astype(
                "int64"
            ),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume_base": volume_base,
            "volume_quote": baseline_volume,
            "trade_count": trade_count.astype(int),
            "taker_buy_base": volume_base * taker_buy_ratio,
            "taker_buy_quote": taker_buy_quote,
            "funding_rate": funding,
            "mark_price": close,
            "open_interest": open_interest,
            "open_interest_value": open_interest,
            "symbol": config.symbol,
        }
    )
    return frame
