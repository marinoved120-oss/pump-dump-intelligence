# Validation v0.2.4

## Automated checks

The included suite contains 24 tests covering:

- event extraction and cleaning;
- leakage-safe grouped chronological splits;
- snapshot-relative labels;
- event-level alert collapsing;
- grouped bootstrap confidence intervals;
- all-null feature removal;
- external-symbol evaluation;
- prospective event partitioning;
- event-end calculation from the future peak metadata used only for purging;
- one global calibration threshold;
- strict past-only and other-symbol fold construction;
- macro and symbol-day summary generation;
- Binance ranges and aliases;
- complete synthetic training and report generation.

Run:

```powershell
docker compose run --rm research pytest
```

Expected result: `24 passed`.

## Prospective empirical protocol

```powershell
docker compose run --rm research prospective SYMBOLS... --target dump_8_15m --model random_forest
```

The protocol is nested and chronological:

1. Independent events are sorted by event start.
2. The latest configured fraction is reserved for final evaluation.
3. The tail of the earlier development interval becomes calibration.
4. For every calibration event, a model is fitted only on completed events from other symbols in the past.
5. Calibration probabilities are pooled and one event-F1 threshold is selected.
6. For every final evaluation event, a fresh model is fitted only on completed events from other symbols in the past.
7. The frozen global threshold is applied without per-symbol or per-event retuning.

Audit condition for each successful fold:

```text
train_max_event_end < event_start - purge_minutes
holdout_symbol not in train_symbols
```

## Primary acceptance metrics

Review these together rather than selecting a model from ROC-AUC alone:

- pooled PR-AUC versus target prevalence;
- event precision / recall / F1 and grouped-bootstrap confidence intervals;
- macro event metrics on positive symbols;
- symbol detection coverage;
- false events per 100 symbol-days;
- lead-time distribution and fraction of alerts with at least 5/10 minutes lead.

## Remaining limitations

- The final period is historical, not live.
- Event definitions and pump-context filters remain predeclared hypotheses.
- Sparse historical OI limits derivatives-feature evaluation.
- Retraining a model before each event is an evaluation protocol, not yet an optimized online serving architecture.
- Threshold and feature stability must be monitored prospectively before any production use.
