# Changelog

## Unreleased - V031 operational hardening

- Parse the complete paper-monitor YAML into immutable strict typed configuration.
- Reject missing fields, unknown fields, malformed YAML, unsafe values and unsafe JSONL paths.
- Add the read-only paper-preflight command with deterministic machine-readable output.
- Exit non-zero when Binance credential variables or unsafe monitor settings are detected.
- Keep credential values redacted and perform no Binance or Telegram network requests.
- Connect validated configuration to the existing paper monitor and append-only outcome tracker.
- Add the deterministic local paper-replay command for strict alert, invalidation and price JSONL streams.
- Run fail-closed preflight before reading replay inputs and emit deterministic redacted JSON reports.
- Reuse the existing PaperMonitor for cooldown, invalidation and append-only outcome tracking.
- Prevent duplicate stored outcomes after restart and perform no Binance or Telegram requests.

## 0.3.0.2 - Paper monitoring packaging release

- Package the completed V030 public-data and paper-monitoring pipeline.
- Include independent Binance spot and futures collectors and order books.
- Track whale-wall lifecycle evidence without assigning real/fake labels.
- Add manipulation, spot/futures divergence and causal contradiction evidence.
- Add Telegram HTML paper reports, cooldowns, invalidations and outcome tracking.
- Require at least three independent evidence groups for red alerts.
- Include the default research configuration in the Python wheel.
- Fall back to the packaged default only when the normal repository default is absent.
- Preserve strict handling for explicitly supplied configuration paths.
- Keep the architecture read-only: no order placement, cancellation or withdrawal endpoints.
- Validation total: 113 automated tests passed; 2525 pre-existing warnings remain.

## 0.3.0.1 — Patch and recovery hardening

- Normalize model-generated unified diffs and remove Markdown wrappers before validation.
- Reject incomplete patches that contain file headers but no valid `@@` hunks.
- Retry malformed model output and repair patches that fail `git apply --check`.
- Keep failed roadmap tasks retryable instead of silently advancing to the next task.
- Delete failed proposal branches and only clean files named by the proposal.
- Remove the dangerous broad `git clean -fd` failure path.
- Preserve unrelated untracked files, local notes and `.env`.
- Added six regression tests; automated total: 48 tests.

## 0.2.4.1

- Prospective evaluation now reads only per-symbol timestamp coverage instead of loading every processed feature row into memory.
- Prevents silent Docker OOM termination on year-long multi-symbol datasets.
- Keeps exposure and false-events-per-100-symbol-days calculations unchanged.


## 0.2.4 — Prospective Protocol

- Added `prospective` command for purged, past-only walk-forward leave-one-symbol-out evaluation.
- Every evaluation event is scored by a freshly fitted model trained only on completed events from other symbols before the event cutoff.
- Added one global threshold selected on earlier pooled out-of-symbol calibration predictions and frozen for the final evaluation window.
- Added fold audit fields including holdout symbol, training symbols, train event count, train positive-event count and maximum training event end.
- Added macro metrics across all symbols and across symbols with positive events.
- Added symbol detection coverage so one event-heavy contract cannot hide weak cross-symbol transfer.
- Added lead-time P25, median, P75 and shares with at least 5/10 minutes of warning.
- Added false events and false snapshot alerts per 100 symbol-days using processed minute-data exposure.
- Added prospective CSV/JSON/HTML artifacts and a dedicated report.
- Added a built-in `pytest` CLI command and changed the Dockerfile to install the `dev` dependency group.
- Added four prospective-protocol regression tests; total automated checks: 24.

## 0.2.3 — Evaluation Pack

- Added event-level precision, recall and F1: repeated snapshot alerts inside one `event_id` are one market alert.
- Added event false-alert rate, alerts per predicted event, first-alert lead time and minutes before the event peak.
- Added vectorized grouped bootstrap confidence intervals by resampling complete independent events.
- Added `availability` command and automatic removal of features that are completely empty in the training split.
- Added feature-availability and effective-feature manifests to the HTML report and artifacts.
- Added exploratory leave-one-symbol-out evaluation with fold-specific validation thresholds and pooled metrics.
- Added event prediction export for normal chronological experiments.

## 0.2.2

- Merge raw pump candidates when their pump-to-peak intervals overlap or share a peak.
- Filter insignificant event-level moves below `max(3%, 2 × realized volatility)`.
- Add FAST, MEDIUM and SLOW pump regimes plus `--regime` filtering.
- Explicitly document and test that snapshot targets are measured from snapshot close.
- Add real/synthetic dataset provenance, symbols, periods and split statistics to reports.
