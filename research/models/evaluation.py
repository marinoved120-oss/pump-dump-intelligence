from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    threshold: float | None
    average_precision: float
    precision: float
    recall: float
    f1: float
    brier: float
    roc_auc: float | None
    false_alerts_per_day: float
    positives: int
    rows: int

    def as_dict(self) -> dict[str, float | int | None]:
        return self.__dict__.copy()


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    grid_size: int = 101,
) -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.0, 1.0, grid_size):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, predictions, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _span_days(timestamps: pd.Series | np.ndarray) -> float:
    values = pd.to_datetime(timestamps, utc=True, errors="coerce")
    values = values[~pd.isna(values)]
    if len(values) < 2:
        return 1 / 1440
    return max((values.max() - values.min()).total_seconds() / 86400, 1 / 1440)


def _score_arrays(
    actual: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    timestamps: pd.Series | np.ndarray,
    threshold: float | None,
) -> EvaluationResult:
    actual = np.asarray(actual, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = np.asarray(predictions, dtype=int)
    if len(actual) == 0:
        return EvaluationResult(
            threshold=threshold,
            average_precision=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            brier=0.0,
            roc_auc=None,
            false_alerts_per_day=0.0,
            positives=0,
            rows=0,
        )
    average_precision = (
        float(average_precision_score(actual, probabilities)) if actual.sum() else 0.0
    )
    roc_auc = (
        float(roc_auc_score(actual, probabilities))
        if len(np.unique(actual)) > 1
        else None
    )
    false_positives = int(((predictions == 1) & (actual == 0)).sum())
    return EvaluationResult(
        threshold=threshold,
        average_precision=average_precision,
        precision=float(precision_score(actual, predictions, zero_division=0)),
        recall=float(recall_score(actual, predictions, zero_division=0)),
        f1=float(f1_score(actual, predictions, zero_division=0)),
        brier=float(brier_score_loss(actual, probabilities)),
        roc_auc=roc_auc,
        false_alerts_per_day=false_positives / _span_days(timestamps),
        positives=int(actual.sum()),
        rows=int(len(actual)),
    )


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    timestamps: pd.Series,
) -> EvaluationResult:
    actual = np.asarray(y_true, dtype=int)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    return _score_arrays(actual, probabilities, predictions, timestamps, threshold)


def evaluate_prediction_frame(frame: pd.DataFrame) -> EvaluationResult:
    """Evaluate a frame whose rows may use fold-specific thresholds."""
    required = {"actual", "probability", "prediction", "timestamp"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {sorted(missing)}")
    return _score_arrays(
        frame["actual"].to_numpy(),
        frame["probability"].to_numpy(),
        frame["prediction"].to_numpy(),
        frame["timestamp"],
        threshold=None,
    )


def event_prediction_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse snapshot predictions into one row per independent event."""
    required = {"event_id", "timestamp", "actual", "prediction"}
    missing = required.difference(predictions.columns)
    if missing or predictions.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "symbol",
                "pump_regime",
                "actual_event",
                "predicted_event",
                "alerts",
                "first_alert_time",
                "first_positive_time",
                "lead_to_first_positive_minutes",
                "first_alert_minutes_before_peak",
            ]
        )

    rows: list[dict[str, Any]] = []
    ordered = predictions.sort_values("timestamp").copy()
    ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], utc=True)
    for event_id, group in ordered.groupby("event_id", sort=False):
        alerts = group[group["prediction"] == 1]
        positives = group[group["actual"] == 1]
        first_alert_time = alerts["timestamp"].min() if not alerts.empty else pd.NaT
        first_positive_time = positives["timestamp"].min() if not positives.empty else pd.NaT
        lead = np.nan
        if pd.notna(first_alert_time) and pd.notna(first_positive_time):
            lead = (first_positive_time - first_alert_time).total_seconds() / 60
        before_peak = np.nan
        if not alerts.empty and "minutes_before_peak" in alerts:
            first_alert_row = alerts.sort_values("timestamp").iloc[0]
            value = pd.to_numeric(
                pd.Series([first_alert_row.get("minutes_before_peak")]), errors="coerce"
            ).iloc[0]
            if pd.notna(value):
                before_peak = float(value)
        rows.append(
            {
                "event_id": str(event_id),
                "symbol": str(group["symbol"].iloc[0]) if "symbol" in group else "UNKNOWN",
                "pump_regime": (
                    str(group["pump_regime"].iloc[0])
                    if "pump_regime" in group
                    else "UNKNOWN"
                ),
                "actual_event": int((group["actual"] == 1).any()),
                "predicted_event": int((group["prediction"] == 1).any()),
                "alerts": int((group["prediction"] == 1).sum()),
                "first_alert_time": first_alert_time,
                "first_positive_time": first_positive_time,
                "lead_to_first_positive_minutes": lead,
                "first_alert_minutes_before_peak": before_peak,
            }
        )
    return pd.DataFrame(rows)


def _event_metric_values(events: pd.DataFrame) -> dict[str, float | int]:
    if events.empty:
        return {
            "event_precision": 0.0,
            "event_recall": 0.0,
            "event_f1": 0.0,
            "actual_positive_events": 0,
            "predicted_events": 0,
            "true_positive_events": 0,
            "false_positive_events": 0,
        }
    actual = events["actual_event"].to_numpy(dtype=int)
    predicted = events["predicted_event"].to_numpy(dtype=int)
    return {
        "event_precision": float(precision_score(actual, predicted, zero_division=0)),
        "event_recall": float(recall_score(actual, predicted, zero_division=0)),
        "event_f1": float(f1_score(actual, predicted, zero_division=0)),
        "actual_positive_events": int(actual.sum()),
        "predicted_events": int(predicted.sum()),
        "true_positive_events": int(((actual == 1) & (predicted == 1)).sum()),
        "false_positive_events": int(((actual == 0) & (predicted == 1)).sum()),
    }


def grouped_bootstrap_event_metrics(
    events: pd.DataFrame,
    repeats: int = 2000,
    random_state: int = 42,
) -> dict[str, float | int | None]:
    """Vectorized grouped bootstrap by resampling whole independent events."""
    if events.empty or repeats <= 0:
        return {"bootstrap_repeats": 0}
    actual = events["actual_event"].to_numpy(dtype=np.int8)
    predicted = events["predicted_event"].to_numpy(dtype=np.int8)
    event_count = len(events)
    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, event_count, size=(repeats, event_count))
    sampled_actual = actual[indices]
    sampled_predicted = predicted[indices]
    true_positive = ((sampled_actual == 1) & (sampled_predicted == 1)).sum(axis=1)
    false_positive = ((sampled_actual == 0) & (sampled_predicted == 1)).sum(axis=1)
    false_negative = ((sampled_actual == 1) & (sampled_predicted == 0)).sum(axis=1)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = np.divide(
        true_positive,
        precision_denominator,
        out=np.full(repeats, np.nan, dtype=float),
        where=precision_denominator != 0,
    )
    recall = np.divide(
        true_positive,
        recall_denominator,
        out=np.full(repeats, np.nan, dtype=float),
        where=recall_denominator != 0,
    )
    f1_denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        f1_denominator,
        out=np.full(repeats, np.nan, dtype=float),
        where=(f1_denominator != 0) & np.isfinite(f1_denominator),
    )
    result: dict[str, float | int | None] = {"bootstrap_repeats": repeats}
    for name, observations in {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
    }.items():
        finite = observations[np.isfinite(observations)]
        result[f"{name}_ci_low"] = (
            float(np.percentile(finite, 2.5)) if len(finite) else None
        )
        result[f"{name}_ci_high"] = (
            float(np.percentile(finite, 97.5)) if len(finite) else None
        )
    return result


def evaluate_event_predictions(
    predictions: pd.DataFrame,
    bootstrap_repeats: int = 2000,
    random_state: int = 42,
) -> tuple[dict[str, float | int | None], pd.DataFrame]:
    event_table = event_prediction_table(predictions)
    metrics: dict[str, float | int | None] = dict(_event_metric_values(event_table))
    span_days = _span_days(predictions["timestamp"]) if not predictions.empty else 1 / 1440
    metrics["false_events_per_day"] = (
        float(metrics["false_positive_events"]) / span_days
    )
    predicted = event_table[event_table["predicted_event"] == 1]
    metrics["alerts_per_predicted_event"] = (
        float(predicted["alerts"].sum() / len(predicted)) if len(predicted) else 0.0
    )
    true_positive = event_table[
        (event_table["actual_event"] == 1) & (event_table["predicted_event"] == 1)
    ]
    leads = pd.to_numeric(
        true_positive["lead_to_first_positive_minutes"], errors="coerce"
    ).dropna()
    before_peak = pd.to_numeric(
        true_positive["first_alert_minutes_before_peak"], errors="coerce"
    ).dropna()
    metrics["median_lead_to_first_positive_minutes"] = (
        float(leads.median()) if not leads.empty else None
    )
    metrics["median_first_alert_minutes_before_peak"] = (
        float(before_peak.median()) if not before_peak.empty else None
    )
    metrics.update(
        grouped_bootstrap_event_metrics(event_table, bootstrap_repeats, random_state)
    )
    return metrics, event_table


def feature_availability(
    splits: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        row: dict[str, Any] = {"feature": feature}
        for split_name, frame in splits.items():
            if feature not in frame or frame.empty:
                row[f"{split_name}_available"] = 0.0
                row[f"{split_name}_non_null"] = 0
            else:
                non_null = int(pd.to_numeric(frame[feature], errors="coerce").notna().sum())
                row[f"{split_name}_available"] = non_null / len(frame)
                row[f"{split_name}_non_null"] = non_null
        rows.append(row)
    return pd.DataFrame(rows)
