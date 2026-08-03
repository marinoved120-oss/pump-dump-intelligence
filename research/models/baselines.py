from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from research.config import TrainingConfig
from research.models.evaluation import (
    choose_threshold,
    evaluate_event_predictions,
    evaluate_predictions,
)


@dataclass
class ModelOutput:
    name: str
    model: Any
    threshold: float
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    event_predictions: pd.DataFrame
    feature_columns: list[str]


def chronological_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True)
    train_end = max(1, int(len(ordered) * train_fraction))
    validation_end = max(
        train_end + 1,
        int(len(ordered) * (train_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, len(ordered) - 1)
    return (
        ordered.iloc[:train_end],
        ordered.iloc[train_end:validation_end],
        ordered.iloc[validation_end:],
    )


def _group_order(frame: pd.DataFrame, group_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("timestamp").reset_index(drop=True).copy()
    group_values = ordered[group_column].astype("string")
    fallback = "ROW-" + ordered["timestamp"].astype(str)
    ordered["__split_group"] = group_values.fillna(fallback)
    groups = (
        ordered.groupby("__split_group", as_index=False)["timestamp"]
        .min()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return ordered, groups


def chronological_group_split(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    group_column: str = "event_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split that never places one event in multiple partitions."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True).copy()
    if group_column not in ordered or ordered[group_column].isna().all():
        return chronological_split(ordered, train_fraction, validation_fraction)

    ordered, group_order = _group_order(ordered, group_column)
    count = len(group_order)
    empty = ordered.iloc[0:0].drop(columns="__split_group")
    if count == 0:
        return empty, empty.copy(), empty.copy()
    if count == 1:
        only = set(group_order.iloc[:1]["__split_group"])
        train = ordered[ordered["__split_group"].isin(only)].drop(columns="__split_group")
        return train, empty, empty.copy()
    if count == 2:
        train_groups = set(group_order.iloc[:1]["__split_group"])
        validation_groups = set(group_order.iloc[1:]["__split_group"])
        train = ordered[ordered["__split_group"].isin(train_groups)].drop(
            columns="__split_group"
        )
        validation = ordered[ordered["__split_group"].isin(validation_groups)].drop(
            columns="__split_group"
        )
        return train, validation, empty

    train_end = max(1, int(count * train_fraction))
    validation_end = max(
        train_end + 1,
        int(count * (train_fraction + validation_fraction)),
    )
    validation_end = min(validation_end, count - 1)
    train_groups = set(group_order.iloc[:train_end]["__split_group"])
    validation_groups = set(group_order.iloc[train_end:validation_end]["__split_group"])
    test_groups = set(group_order.iloc[validation_end:]["__split_group"])

    train = ordered[ordered["__split_group"].isin(train_groups)].drop(
        columns="__split_group"
    )
    validation = ordered[ordered["__split_group"].isin(validation_groups)].drop(
        columns="__split_group"
    )
    test = ordered[ordered["__split_group"].isin(test_groups)].drop(
        columns="__split_group"
    )
    return train, validation, test


def chronological_group_train_validation_split(
    frame: pd.DataFrame,
    validation_fraction: float = 0.20,
    group_column: str = "event_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation split for external-symbol evaluation."""
    ordered = frame.sort_values("timestamp").reset_index(drop=True).copy()
    if group_column not in ordered or ordered[group_column].isna().all():
        cut = max(1, min(len(ordered) - 1, int(len(ordered) * (1 - validation_fraction))))
        return ordered.iloc[:cut], ordered.iloc[cut:]
    ordered, groups = _group_order(ordered, group_column)
    count = len(groups)
    if count < 2:
        return ordered.drop(columns="__split_group"), ordered.iloc[0:0].drop(
            columns="__split_group"
        )
    validation_count = max(1, int(round(count * validation_fraction)))
    validation_count = min(validation_count, count - 1)
    train_groups = set(groups.iloc[:-validation_count]["__split_group"])
    validation_groups = set(groups.iloc[-validation_count:]["__split_group"])
    train = ordered[ordered["__split_group"].isin(train_groups)].drop(
        columns="__split_group"
    )
    validation = ordered[ordered["__split_group"].isin(validation_groups)].drop(
        columns="__split_group"
    )
    return train, validation


def usable_feature_columns(
    train: pd.DataFrame,
    feature_columns: Iterable[str],
) -> tuple[list[str], list[str]]:
    """Remove features that contain no observed numeric value in training."""
    usable: list[str] = []
    dropped: list[str] = []
    for feature in feature_columns:
        if feature not in train:
            dropped.append(feature)
            continue
        observed = pd.to_numeric(train[feature], errors="coerce").notna().any()
        (usable if observed else dropped).append(feature)
    return usable, dropped


def _probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    decision = np.asarray(model.decision_function(features))
    return 1 / (1 + np.exp(-decision))


def build_estimators(feature_columns: list[str], config: TrainingConfig) -> dict[str, Any]:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                feature_columns,
            )
        ],
        remainder="drop",
    )

    logistic = Pipeline(
        [
            ("preprocess", preprocessing),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=10,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    hist_gradient = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.07,
                    max_iter=250,
                    max_leaf_nodes=31,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    class_weight="balanced",
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        lightgbm = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=350,
                        learning_rate=0.04,
                        num_leaves=31,
                        class_weight="balanced",
                        random_state=config.random_state,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        lightgbm = hist_gradient

    return {
        "logistic_regression": logistic,
        "random_forest": random_forest,
        "lightgbm_or_hist_gradient": lightgbm,
    }


def rule_probability(frame: pd.DataFrame) -> np.ndarray:
    def safe(name: str) -> pd.Series:
        if name not in frame:
            return pd.Series(0.0, index=frame.index)
        return pd.to_numeric(frame[name], errors="coerce").fillna(0.0)

    exhaustion = (
        (-safe("cvd_slope_15m") / (safe("volume_quote") + 1)).clip(-1, 1)
        + (-safe("buy_efficiency_change") * 100).clip(-1, 1)
        + safe("upper_wick_ratio").clip(0, 1)
        + safe("failed_breakouts_15m").clip(0, 3) / 3
    ) / 4
    leverage = (
        safe("oi_change_15m").clip(-0.1, 0.1) * 5
        + safe("funding_robust_z").clip(-3, 3) / 6
        - safe("return_15m").clip(-0.2, 0.2)
    )
    raw = -1.2 + 2.4 * exhaustion + 1.2 * leverage + 1.5 * safe("pump_context")
    return 1 / (1 + np.exp(-raw.to_numpy()))


def _prediction_frame(
    test: pd.DataFrame,
    target: str,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "timestamp": test["timestamp"].to_numpy(),
        "actual": test[target].to_numpy(),
        "probability": probabilities,
        "prediction": (probabilities >= threshold).astype(int),
    }
    for name in (
        "sample_id",
        "event_id",
        "symbol",
        "pump_regime",
        "minutes_before_peak",
        "minutes_from_pump_start",
    ):
        if name in test:
            columns[name] = test[name].to_numpy()
    return pd.DataFrame(columns)


def _evaluate_output(
    name: str,
    model: Any,
    threshold: float,
    test: pd.DataFrame,
    target: str,
    probabilities: np.ndarray,
    feature_columns: list[str],
    config: TrainingConfig,
) -> ModelOutput:
    predictions = _prediction_frame(test, target, probabilities, threshold)
    metrics = evaluate_predictions(
        test[target], probabilities, threshold, test["timestamp"]
    ).as_dict()
    event_metrics, event_predictions = evaluate_event_predictions(
        predictions,
        bootstrap_repeats=config.bootstrap_repeats,
        random_state=config.random_state,
    )
    metrics.update(event_metrics)
    return ModelOutput(
        name=name,
        model=model,
        threshold=threshold,
        metrics=metrics,
        predictions=predictions,
        event_predictions=event_predictions,
        feature_columns=feature_columns,
    )


def _fit_selected_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: TrainingConfig,
    model_names: Iterable[str] | None = None,
    include_rules: bool = True,
) -> list[ModelOutput]:
    if train.empty or validation.empty or test.empty:
        raise ValueError("Not enough data for train/validation/test evaluation")
    if train[target].nunique() < 2:
        raise ValueError("Training period contains only one target class")
    if not feature_columns:
        raise ValueError("No observed model features remain in the training period")

    outputs: list[ModelOutput] = []
    selected = set(model_names or build_estimators(feature_columns, config).keys())

    if include_rules and (not model_names or "adaptive_rules" in selected):
        validation_rule = rule_probability(validation)
        threshold = choose_threshold(
            validation[target].to_numpy(),
            validation_rule,
            config.decision_threshold_grid,
        )
        test_rule = rule_probability(test)
        outputs.append(
            _evaluate_output(
                "adaptive_rules",
                None,
                threshold,
                test,
                target,
                test_rule,
                feature_columns,
                config,
            )
        )

    estimators = build_estimators(feature_columns, config)
    unknown = selected.difference(set(estimators) | {"adaptive_rules"})
    if unknown:
        raise ValueError(f"Unknown model names: {', '.join(sorted(unknown))}")
    for name, estimator in estimators.items():
        if name not in selected:
            continue
        estimator.fit(train[feature_columns], train[target])
        validation_probability = _probability(estimator, validation[feature_columns])
        threshold = choose_threshold(
            validation[target].to_numpy(),
            validation_probability,
            config.decision_threshold_grid,
        )
        test_probability = _probability(estimator, test[feature_columns])
        outputs.append(
            _evaluate_output(
                name,
                estimator,
                threshold,
                test,
                target,
                test_probability,
                feature_columns,
                config,
            )
        )
    return outputs


def train_models(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: TrainingConfig,
    group_column: str | None = None,
) -> list[ModelOutput]:
    usable = frame.dropna(subset=["timestamp", target]).copy()
    if group_column:
        train, validation, test = chronological_group_split(
            usable,
            config.train_fraction,
            config.validation_fraction,
            group_column,
        )
    else:
        train, validation, test = chronological_split(
            usable,
            config.train_fraction,
            config.validation_fraction,
        )
    selected_features, _ = usable_feature_columns(train, feature_columns)
    return _fit_selected_models(
        train,
        validation,
        test,
        selected_features,
        target,
        config,
    )


def train_models_external_holdout(
    training_pool: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    target: str,
    config: TrainingConfig,
    group_column: str = "event_id",
    model_names: Iterable[str] | None = None,
) -> tuple[list[ModelOutput], list[str], list[str]]:
    """Train on other symbols and evaluate on one completely unseen symbol."""
    training_pool = training_pool.dropna(subset=["timestamp", target]).copy()
    holdout = holdout.dropna(subset=["timestamp", target]).copy()
    train, validation = chronological_group_train_validation_split(
        training_pool,
        validation_fraction=config.loso_validation_fraction,
        group_column=group_column,
    )
    selected_features, dropped = usable_feature_columns(train, feature_columns)
    outputs = _fit_selected_models(
        train,
        validation,
        holdout,
        selected_features,
        target,
        config,
        model_names=model_names,
        include_rules=True,
    )
    return outputs, selected_features, dropped
