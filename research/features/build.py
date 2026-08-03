from __future__ import annotations

import numpy as np
import pandas as pd

from research.config import FeatureConfig


EPSILON = 1e-12


def robust_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or max(30, window // 10)
    median = series.rolling(window, min_periods=min_periods).median()
    absolute_deviation = (series - median).abs()
    mad = absolute_deviation.rolling(window, min_periods=min_periods).median()
    return (series - median) / (1.4826 * mad.replace(0, np.nan))


def rolling_time_since_high(series: pd.Series, window: int) -> pd.Series:
    values = series.to_numpy(dtype=float)
    output = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        segment = values[index - window + 1 : index + 1]
        if np.isnan(segment).all():
            continue
        position = int(np.nanargmax(segment))
        output[index] = window - 1 - position
    return pd.Series(output, index=series.index)


def merge_derivatives(
    klines: pd.DataFrame,
    funding: pd.DataFrame | None,
    open_interest: pd.DataFrame | None,
) -> pd.DataFrame:
    frame = klines.sort_values("timestamp").copy()
    for derivative_frame in (funding, open_interest):
        if derivative_frame is None or derivative_frame.empty:
            continue
        right = derivative_frame.sort_values("timestamp").copy()
        frame = pd.merge_asof(frame, right, on="timestamp", direction="backward")
    return frame


def build_features(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume_quote",
        "trade_count",
        "taker_buy_quote",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for feature engineering: {sorted(missing)}")

    data = frame.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True).copy()
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume_quote",
        "trade_count",
        "taker_buy_quote",
    ]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")

    for minutes in (1, 5, 15, 60, 240, 1440):
        data[f"return_{minutes}m"] = data["close"].pct_change(minutes, fill_method=None)
        data[f"log_return_{minutes}m"] = np.log(data["close"] / data["close"].shift(minutes))

    one_minute_log_return = np.log(data["close"] / data["close"].shift(1))
    for minutes in (5, 15, 60, 240):
        data[f"realized_vol_{minutes}m"] = (
            one_minute_log_return.rolling(minutes, min_periods=max(3, minutes // 3)).std()
            * np.sqrt(minutes)
        )

    robust_window = config.robust_window_minutes
    data["volume_robust_z"] = robust_zscore(data["volume_quote"], robust_window)
    data["trade_count_robust_z"] = robust_zscore(data["trade_count"], robust_window)

    ewma_span = config.ewma_days * 24 * 60
    data["volume_ewma_20d"] = data["volume_quote"].ewm(span=ewma_span, adjust=False).mean()
    data["volume_ratio_ewma"] = data["volume_quote"] / data["volume_ewma_20d"].replace(0, np.nan)
    data["price_ewma_20d"] = data["close"].ewm(span=ewma_span, adjust=False).mean()
    data["distance_price_ewma"] = data["close"] / data["price_ewma_20d"] - 1

    typical_price = (data["high"] + data["low"] + data["close"]) / 3
    for minutes in (15, 60):
        weighted = (typical_price * data["volume_quote"]).rolling(minutes, min_periods=3).sum()
        volume = data["volume_quote"].rolling(minutes, min_periods=3).sum()
        data[f"vwap_{minutes}m"] = weighted / volume.replace(0, np.nan)
        data[f"distance_vwap_{minutes}m"] = data["close"] / data[f"vwap_{minutes}m"] - 1

    high_window = config.high_window_minutes
    rolling_high = data["high"].rolling(high_window, min_periods=max(5, high_window // 4)).max()
    data["distance_high_60m"] = data["close"] / rolling_high - 1
    data["time_since_high_60m"] = rolling_time_since_high(data["high"], high_window)

    candle_range = (data["high"] - data["low"]).replace(0, np.nan)
    upper_wick = data["high"] - data[["open", "close"]].max(axis=1)
    data["upper_wick_ratio"] = upper_wick / candle_range

    data["taker_buy_ratio"] = data["taker_buy_quote"] / data["volume_quote"].replace(0, np.nan)
    data["taker_imbalance"] = 2 * data["taker_buy_ratio"] - 1
    signed_quote = data["volume_quote"] * data["taker_imbalance"].fillna(0)
    data["cvd_5m"] = signed_quote.rolling(5, min_periods=1).sum()
    data["cvd_15m"] = signed_quote.rolling(15, min_periods=1).sum()
    data["cvd_slope_15m"] = data["cvd_15m"].diff(5)

    buy_volume_5m = data["taker_buy_quote"].rolling(5, min_periods=1).sum()
    data["buy_efficiency_5m"] = data["return_5m"] / np.log1p(buy_volume_5m.clip(lower=0))
    data["buy_efficiency_change"] = data["buy_efficiency_5m"].diff(5)
    data["sell_volume_5m"] = (
        (data["volume_quote"] - data["taker_buy_quote"]).clip(lower=0).rolling(5).sum()
    )
    data["sell_acceleration"] = data["sell_volume_5m"].pct_change(5, fill_method=None)

    breakout = data["high"] >= rolling_high.shift(1)
    close_failed = data["close"] < rolling_high.shift(1) * 0.995
    data["failed_breakout"] = (breakout & close_failed).astype(int)
    data["failed_breakouts_15m"] = data["failed_breakout"].rolling(15).sum()

    if "open_interest" in data.columns:
        data["open_interest"] = pd.to_numeric(data["open_interest"], errors="coerce").ffill()
        for minutes in (5, 15, 60):
            data[f"oi_change_{minutes}m"] = data["open_interest"].pct_change(
                minutes, fill_method=None
            )
        data["price_oi_divergence_15m"] = data["return_15m"] - data["oi_change_15m"]
    else:
        for name in ("oi_change_5m", "oi_change_15m", "oi_change_60m", "price_oi_divergence_15m"):
            data[name] = np.nan

    if "funding_rate" in data.columns:
        data["funding_rate"] = pd.to_numeric(data["funding_rate"], errors="coerce").ffill()
        data["funding_robust_z"] = robust_zscore(data["funding_rate"], 30 * 24 * 60)
    else:
        data["funding_rate"] = np.nan
        data["funding_robust_z"] = np.nan

    data["data_quality"] = compute_data_quality(data)
    return data


def compute_data_quality(data: pd.DataFrame) -> pd.Series:
    required = ["close", "volume_quote", "trade_count", "taker_buy_quote"]
    completeness = data[required].notna().mean(axis=1)
    positive_price = (data["close"] > 0).astype(float)
    nonnegative_volume = (data["volume_quote"] >= 0).astype(float)
    timestamp_gap = data["timestamp"].diff().dt.total_seconds().fillna(60)
    fresh = (timestamp_gap <= 90).astype(float)
    activity = ((data["trade_count"] > 0) & (data["volume_quote"] > 0)).astype(float)
    quality = (
        0.35 * completeness
        + 0.15 * positive_price
        + 0.15 * nonnegative_volume
        + 0.20 * fresh
        + 0.15 * activity
    )
    return quality.clip(0, 1)


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    candidates = [
        "return_1m",
        "return_5m",
        "return_15m",
        "return_60m",
        "realized_vol_5m",
        "realized_vol_15m",
        "realized_vol_60m",
        "volume_robust_z",
        "trade_count_robust_z",
        "volume_ratio_ewma",
        "distance_price_ewma",
        "distance_vwap_15m",
        "distance_vwap_60m",
        "distance_high_60m",
        "time_since_high_60m",
        "upper_wick_ratio",
        "taker_buy_ratio",
        "taker_imbalance",
        "cvd_slope_15m",
        "buy_efficiency_5m",
        "buy_efficiency_change",
        "sell_acceleration",
        "failed_breakouts_15m",
        "oi_change_5m",
        "oi_change_15m",
        "oi_change_60m",
        "price_oi_divergence_15m",
        "funding_rate",
        "funding_robust_z",
        "data_quality",
    ]
    return [column for column in candidates if column in frame.columns]
