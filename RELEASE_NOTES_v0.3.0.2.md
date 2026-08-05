# v0.3.0.2 - Paper Monitoring Packaging Release

This release publishes the completed V030 public-market evidence and paper-monitoring pipeline with corrected Python packaging and release metadata.

## Included pipeline

- Public Binance spot and futures WebSocket recorders with independent order books.
- Immutable live schemas and order-book sequence handling.
- Whale-wall lifecycle tracking for execution, refill, liquidity pull, cancellation and repositioning.
- Spoofing-like, iceberg-like and absorption-like evidence scoring with explicit contradictions and confidence.
- Spot/futures divergence context covering organic spot demand, short squeezes, futures-led pumps and late-long buildup.
- A causal evidence engine with supporting evidence, contradictions, missing-data statements and invalidation rules.
- Telegram HTML paper-monitor reports with cooldowns, invalidation updates and outcomes at 5, 15, 30, 60 and 240 minutes.
- Append-only JSONL monitoring storage.
- A minimum of three independent evidence groups for red alerts.

## Packaging correction

The Python wheel now includes the default research configuration.

`pd-research` continues to prefer `configs/research.yaml` from the working project. When that default path is absent, it falls back to the packaged configuration. Explicitly supplied missing paths still fail rather than silently using a different file.

## Safety boundaries

- Public/read-only market data only.
- No order placement, cancellation or withdrawal endpoints.
- No exchange trading permissions.
- No automatic production-model updates.
- Evidence is expressed cautiously and does not accuse market participants.

## Validation

- 113 automated tests passed.
- Python compilation completed successfully for `research` and `orchestrator`.
- The wheel built successfully and all new modules imported from the installed wheel.
- Packaged-config fallback was verified from an empty working directory.
- The trading-endpoint safety scan returned no matches.
- 2525 existing NumPy/Pandas timedelta deprecation warnings remain; the warning count did not increase.
