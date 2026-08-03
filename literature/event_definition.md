# Event definition v0.2.2

## Raw trigger

A raw pump candidate is produced when a 15m, 60m or 240m return exceeds both a fixed threshold and its volatility-adjusted threshold, with abnormal volume/trade activity and sufficient data quality.

## Independent event

Raw candidates are not automatically independent. Their pump-to-peak intervals are merged when they overlap or share the same peak. The merged event is then recalculated from the earliest trigger to the highest price in the combined interval.

## Event-level acceptance

An event is retained only when:

```text
pump_return_to_peak >= max(0.03, 2 × realized_volatility_60m_at_start)
```

This prevents a momentary statistical anomaly with negligible final appreciation from being labelled as a pump.

## Pump regimes

```text
FAST    minutes_to_peak <= 30
MEDIUM  30 < minutes_to_peak <= 120
SLOW    minutes_to_peak > 120
```

## Warning labels

Every warning snapshot is labelled from its own close:

```text
forward_drawdown_H(t) = min(low[t+1 : t+H]) / close[t] - 1
```

The future event peak is not the reference price for the model target.
