"""
Leakage-safe nested subject-level evaluation for three-class gait classification.

Classes
-------
0: Healthy leg
1: Affected side
2: Non-affected side

The script treats each participant as an indivisible group. The outer loop is
Leave-One-Subject-Out (LOSO) and estimates final generalisation. The inner loop
uses StratifiedGroupKFold to select the sensor combination, fold-fitted feature
count, and prespecified classifier using mean macro F1.

Only outer-LOSO predictions are used for final performance figures. Inner-CV
figures are written to a supplementary directory and labelled as selection
diagnostics.

Example
-------
python ml_multi_classification_nested_journal.py --data features_dataset.csv
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ID_COLUMN = "ID"
LABEL_COLUMN = "Label"
CLASS_LABELS = [0, 1, 2]
CLASS_NAMES = {
    0: "Healthy leg",
    1: "Affected side",
    2: "Non-affected side",
}
RANDOM_STATE = 42
INNER_SPLITS = 5
BOOTSTRAP_REPLICATES = 5000
FEATURE_COUNTS: Tuple[Optional[int], ...] = (5, 10, 15, None)
PRIMARY_METRIC = "macro_f1"

GYRO_PREFIXES = ("gyrox", "gyroy", "gyroz")
ACC_PREFIXES = ("accx", "accy", "accz")
EMG_PREFIXES = ("GMinter", "RFinter", "BFinter", "MGinter", "TAinter", "PLinter")
SENSOR_PREFIXES: Dict[str, Tuple[str, ...]] = {
    "Gyroscope": GYRO_PREFIXES,
    "Accelerometer": ACC_PREFIXES,
    "EMG": EMG_PREFIXES,
    "Gyroscope + Accelerometer": GYRO_PREFIXES + ACC_PREFIXES,
    "Gyroscope + EMG": GYRO_PREFIXES + EMG_PREFIXES,
    "Accelerometer + EMG": ACC_PREFIXES + EMG_PREFIXES,
    "All sensors": GYRO_PREFIXES + ACC_PREFIXES + EMG_PREFIXES,
}
SENSOR_ORDER = list(SENSOR_PREFIXES)
MODEL_ORDER = [
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "k-NN",
    "Gaussian Naive Bayes",
    "Random Forest",
    "Extra Trees",
    "XGBoost",
]

COLOURS = {
    "blue": "#2F6690",
    "light_blue": "#D9EAF4",
    "dark": "#243447",
    "muted": "#627D98",
    "grid": "#D9E2EC",
    "healthy": "#4C78A8",
    "affected": "#E45756",
    "non_affected": "#72B7B2",
}


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "savefig.facecolor": "white",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "x",
            "axes.axisbelow": True,
            "grid.color": COLOURS["grid"],
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def parse_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.strip() # type: ignore
    cleaned = cleaned.str.replace("\u00a0", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
    cleaned = cleaned.str.replace(",", ".", regex=False)
    cleaned = cleaned.replace(
        {"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan, "<NA>": np.nan}
    )
    return pd.to_numeric(cleaned, errors="coerce")


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path, sep=None, engine="python")
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df.columns = [str(column).strip().replace("PLLinter_", "PLinter_") for column in df]
    missing = {ID_COLUMN, LABEL_COLUMN}.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df[LABEL_COLUMN] = parse_numeric(df[LABEL_COLUMN])
    df = df.dropna(subset=[ID_COLUMN, LABEL_COLUMN]).copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    unknown = sorted(set(df[LABEL_COLUMN]).difference(CLASS_LABELS))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}; expected {CLASS_LABELS}")
    df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()
    for column in df.columns:
        if column not in {ID_COLUMN, LABEL_COLUMN}:
            df[column] = parse_numeric(df[column])
    return df.reset_index(drop=True)


def infer_subject(raw_id: object) -> str:
    text = str(raw_id).strip()
    text = re.sub(
        r"(?i)(non[_\- ]?affected|affected|healthy|left|right|leg|side)", "", text
    )
    text = re.sub(r"[_\-\s]+", "_", text).strip("_")
    match = re.search(r"(?i)([a-z]+)[_\- ]*0*([0-9]+)", text)
    if match:
        return f"{match.group(1).upper()}{int(match.group(2)):02d}"
    match = re.search(r"([0-9]+)", text)
    if match:
        return f"S{int(match.group(1)):02d}"
    if not text:
        raise ValueError(f"Could not infer subject from ID {raw_id!r}")
    return text.upper()


def validate_subject_structure(df: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    rows = []
    for subject in sorted(np.unique(groups)):
        labels = df.loc[groups == subject, LABEL_COLUMN].astype(int).tolist() # type: ignore
        counts = Counter(labels)
        if counts == Counter({0: 2}):
            phenotype = "healthy"
        elif counts == Counter({1: 1, 2: 1}):
            phenotype = "stroke"
        else:
            phenotype = "unexpected"
        rows.append(
            {
                "subject": subject,
                "n_rows": len(labels),
                "labels": "|".join(map(str, labels)),
                "phenotype": phenotype,
            }
        )
    summary = pd.DataFrame(rows)
    bad = summary[summary["phenotype"] == "unexpected"]
    if not bad.empty:
        raise ValueError(
            "Subject grouping produced unexpected label patterns:\n"
            + bad.to_string(index=False)
        )
    if summary["phenotype"].nunique() != 2:
        raise ValueError("Both healthy and stroke subjects are required.")
    return summary


def get_sensor_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for sensor_name, prefixes in SENSOR_PREFIXES.items():
        columns = [
            column
            for column in df.columns
            if column not in {ID_COLUMN, LABEL_COLUMN}
            and any(str(column).startswith(prefix + "_") for prefix in prefixes)
        ]
        result[sensor_name] = columns
    return result


class AdaptiveANOVASelector(BaseEstimator, TransformerMixin):
    """Fold-fitted ANOVA selector that safely caps k at available features."""

    def __init__(self, k: Optional[int] = None):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("Feature selector received no columns.")
        variances = np.nanvar(X, axis=0)
        self.nonconstant_mask_ = np.isfinite(variances) & (variances > 0)
        if not self.nonconstant_mask_.any():
            raise ValueError("All training features are constant.")
        X_valid = X[:, self.nonconstant_mask_]
        if self.k is None or self.k >= X_valid.shape[1]:
            local_support = np.ones(X_valid.shape[1], dtype=bool)
            self.scores_ = np.full(X_valid.shape[1], np.nan)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores, _ = f_classif(X_valid, y)
            scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=np.inf)
            chosen = np.argsort(scores, kind="mergesort")[-int(self.k) :]
            local_support = np.zeros(X_valid.shape[1], dtype=bool)
            local_support[chosen] = True
            self.scores_ = scores
        self.support_ = np.zeros(X.shape[1], dtype=bool)
        self.support_[np.flatnonzero(self.nonconstant_mask_)[local_support]] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.support_]

    def get_support(self) -> np.ndarray:
        return self.support_.copy()


def build_models() -> Dict[str, BaseEstimator]:
    models: Dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE
        ),
        "Linear SVM": SVC(kernel="linear", C=1.0, class_weight="balanced"),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        "k-NN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    except ImportError:
        pass
    return models


def build_pipeline(model: BaseEstimator, feature_count: Optional[int]) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("selector", AdaptiveANOVASelector(feature_count)),
            ("model", clone(model)),
        ]
    )


def metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)
    return {
        "accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "macro_precision": float(
            precision_score(y_true_array, y_pred_array, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true_array, y_pred_array, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true_array, y_pred_array, labels=CLASS_LABELS, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true_array, y_pred_array, labels=CLASS_LABELS, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true_array, y_pred_array)),
    }


def validate_inner_splits(
    splits: Iterable[Tuple[np.ndarray, np.ndarray]], y: pd.Series
) -> List[Tuple[np.ndarray, np.ndarray]]:
    checked = []
    expected = set(CLASS_LABELS)
    for train_index, valid_index in splits:
        if set(y.iloc[train_index].astype(int)) != expected:
            raise ValueError("An inner training fold does not contain all classes.")
        if set(y.iloc[valid_index].astype(int)) != expected:
            raise ValueError("An inner validation fold does not contain all classes.")
        checked.append((train_index, valid_index))
    return checked


def feature_label(feature_count: Optional[int]) -> str:
    return "All features" if feature_count is None else f"Top {feature_count} ANOVA features"


def model_complexity_rank(name: str) -> int:
    preferred = [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "Linear SVM",
        "k-NN",
        "RBF SVM",
        "Random Forest",
        "Extra Trees",
        "XGBoost",
    ]
    return preferred.index(name) if name in preferred else len(preferred)


def nested_evaluation(
    df: pd.DataFrame,
    groups: np.ndarray,
    sensor_columns: Dict[str, List[str]],
    models: Dict[str, BaseEstimator],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_columns = sorted({column for columns in sensor_columns.values() for column in columns})
    if not all_columns:
        raise ValueError("No sensor feature columns matched the configured prefixes.")
    X = df[all_columns].replace([np.inf, -np.inf], np.nan)
    y = df[LABEL_COLUMN].astype(int)
    predictions: List[dict] = []
    selections: List[dict] = []
    candidate_rows: List[dict] = []
    fold_score_rows: List[dict] = []
    model_prediction_rows: List[dict] = []

    outer = LeaveOneGroupOut()
    for outer_fold, (train_index, test_index) in enumerate(
        outer.split(X, y, groups), start=1
    ):
        held_out = str(np.unique(groups[test_index])[0])
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        train_groups = groups[train_index]
        inner = StratifiedGroupKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )
        inner_splits = validate_inner_splits(
            inner.split(X_train, y_train, train_groups), y_train
        )
        fold_candidates: List[dict] = []

        for sensor_rank, sensor_name in enumerate(SENSOR_ORDER):
            columns = [column for column in sensor_columns[sensor_name] if column in X]
            if not columns:
                continue
            counts = [count for count in FEATURE_COUNTS if count is None or count < len(columns)]
            for feature_count in counts:
                for model_name, estimator in models.items():
                    per_fold_scores: List[Dict[str, float]] = []
                    failed = False
                    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits, start=1):
                        pipeline = build_pipeline(estimator, feature_count)
                        try:
                            pipeline.fit(X_train.iloc[inner_train][columns], y_train.iloc[inner_train])
                            predicted = pipeline.predict(X_train.iloc[inner_valid][columns])
                        except Exception as error:
                            warnings.warn(
                                f"Skipped candidate in outer fold {outer_fold}: "
                                f"{sensor_name} | {feature_label(feature_count)} | {model_name}: {error}"
                            )
                            failed = True
                            break
                        score = metrics(y_train.iloc[inner_valid], predicted) # type: ignore
                        per_fold_scores.append(score)
                        fold_score_rows.append(
                            {
                                "outer_fold": outer_fold,
                                "held_out_subject": held_out,
                                "inner_fold": inner_fold,
                                "sensor_combination": sensor_name,
                                "feature_set": feature_label(feature_count),
                                "feature_count": "all" if feature_count is None else feature_count,
                                "model": model_name,
                                **score,
                            }
                        )
                    if failed:
                        continue
                    candidate = {
                        "outer_fold": outer_fold,
                        "held_out_subject": held_out,
                        "sensor_combination": sensor_name,
                        "sensor_rank": sensor_rank,
                        "feature_set": feature_label(feature_count),
                        "feature_count": "all" if feature_count is None else feature_count,
                        "feature_rank": len(columns) if feature_count is None else int(feature_count),
                        "model": model_name,
                        "model_rank": model_complexity_rank(model_name),
                    }
                    for metric_name in per_fold_scores[0]:
                        values = [score[metric_name] for score in per_fold_scores]
                        candidate[f"mean_{metric_name}"] = float(np.mean(values))
                        candidate[f"sd_{metric_name}"] = float(np.std(values, ddof=1))
                    fold_candidates.append(candidate)
                    candidate_rows.append(candidate.copy())

        if not fold_candidates:
            raise RuntimeError(f"No valid candidates in outer fold {outer_fold}.")
        ranked = pd.DataFrame(fold_candidates).sort_values(
            [
                "mean_macro_f1",
                "mean_balanced_accuracy",
                "mean_accuracy",
                "feature_rank",
                "sensor_rank",
                "model_rank",
            ],
            ascending=[False, False, False, True, True, True],
            kind="mergesort",
        )
        winner = ranked.iloc[0].to_dict()
        sensor_name = str(winner["sensor_combination"])
        model_name = str(winner["model"])
        raw_count = winner["feature_count"]
        feature_count = None if str(raw_count) == "all" else int(raw_count)
        columns = sensor_columns[sensor_name]
        final_pipeline = build_pipeline(models[model_name], feature_count)
        final_pipeline.fit(X_train[columns], y_train)
        outer_pred = final_pipeline.predict(X_test[columns]).astype(int) # type: ignore
        support = final_pipeline.named_steps["selector"].get_support()
        selected_features = list(np.asarray(columns)[support])
        selection = {
            **winner,
            "selected_features": "|".join(selected_features),
        }
        selections.append(selection)
        for local_index, sample_index in enumerate(test_index):
            true_label = int(y_test.iloc[local_index])
            predicted_label = int(outer_pred[local_index])
            predictions.append(
                {
                    "outer_fold": outer_fold,
                    "sample_index": int(sample_index),
                    "subject": held_out,
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": predicted_label,
                    "predicted_class": CLASS_NAMES[predicted_label],
                    "correct": int(true_label == predicted_label),
                    "selected_model": model_name,
                    "selected_sensor_combination": sensor_name,
                    "selected_feature_set": feature_label(feature_count),
                }
            )

        # Unbiased classifier comparison: within this outer training set,
        # independently select the best sensor/feature configuration for each
        # classifier using inner CV only. Each classifier then predicts the
        # same untouched outer subject. These predictions support a fair
        # outer-LOSO comparison between classifiers.
        for comparison_model_name, comparison_estimator in models.items():
            model_ranked = ranked[ranked["model"] == comparison_model_name]
            if model_ranked.empty:
                continue
            model_winner = model_ranked.iloc[0].to_dict()
            comparison_sensor = str(model_winner["sensor_combination"])
            comparison_columns = sensor_columns[comparison_sensor]
            comparison_raw_count = model_winner["feature_count"]
            comparison_feature_count = (
                None
                if str(comparison_raw_count) == "all"
                else int(comparison_raw_count)
            )
            comparison_pipeline = build_pipeline(
                comparison_estimator, comparison_feature_count
            )
            comparison_pipeline.fit(
                X_train[comparison_columns], y_train
            )
            comparison_predictions = comparison_pipeline.predict(
                X_test[comparison_columns] 
            ).astype(int) # type: ignore
            for local_index, sample_index in enumerate(test_index):
                true_label = int(y_test.iloc[local_index])
                predicted_label = int(comparison_predictions[local_index])
                model_prediction_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "sample_index": int(sample_index),
                        "subject": held_out,
                        "true_label": true_label,
                        "true_class": CLASS_NAMES[true_label],
                        "predicted_label": predicted_label,
                        "predicted_class": CLASS_NAMES[predicted_label],
                        "correct": int(true_label == predicted_label),
                        "model": comparison_model_name,
                        "selected_sensor_combination": comparison_sensor,
                        "selected_feature_set": feature_label(
                            comparison_feature_count
                        ),
                        "inner_macro_f1": float(
                            model_winner["mean_macro_f1"]
                        ),
                    }
                )
        print(
            f"Outer {outer_fold:02d} | held out {held_out} | "
            f"{sensor_name} | {feature_label(feature_count)} | {model_name} | "
            f"inner macro F1={winner['mean_macro_f1']:.3f}"
        )

    return (
        pd.DataFrame(predictions),
        pd.DataFrame(selections),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(fold_score_rows),
        pd.DataFrame(model_prediction_rows),
    )


def subject_stratified_bootstrap(
    predictions: pd.DataFrame,
    subject_summary: pd.DataFrame,
    n_replicates: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    subject_tables = {
        str(subject): group.copy()
        for subject, group in predictions.groupby("subject", sort=False)
    }
    strata = {
        phenotype: group["subject"].astype(str).tolist()
        for phenotype, group in subject_summary.groupby("phenotype", sort=False)
    }
    overall_rows: List[dict] = []
    class_rows: List[dict] = []
    for replicate in range(n_replicates):
        sampled_tables = []
        for subjects in strata.values():
            sampled_subjects = rng.choice(subjects, size=len(subjects), replace=True)
            sampled_tables.extend(subject_tables[str(subject)] for subject in sampled_subjects)
        sample = pd.concat(sampled_tables, ignore_index=True)
        y_true = sample["true_label"].to_numpy(dtype=int)
        y_pred = sample["predicted_label"].to_numpy(dtype=int)
        overall_rows.append({"replicate": replicate, **metrics(y_true, y_pred)})
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_LABELS, zero_division=0
        )
        for index, label in enumerate(CLASS_LABELS):
            class_rows.append(
                {
                    "replicate": replicate,
                    "label": label,
                    "class": CLASS_NAMES[label],
                    "precision": float(precision[index]), # type: ignore
                    "recall": float(recall[index]), # type: ignore
                    "f1": float(f1[index]), # type: ignore
                    "support": int(support[index]), # type: ignore
                }
            )
    return pd.DataFrame(overall_rows), pd.DataFrame(class_rows)


def confidence_interval(values: pd.Series) -> Tuple[float, float]:
    return float(values.quantile(0.025)), float(values.quantile(0.975))


def model_comparison_with_intervals(
    model_predictions: pd.DataFrame,
    subject_summary: pd.DataFrame,
    n_replicates: int,
) -> pd.DataFrame:
    """Calculate unbiased outer-LOSO metrics and subject-level intervals per model."""
    rows: List[dict] = []
    for model_name, predictions in model_predictions.groupby("model", sort=False):
        estimates = metrics( predictions["true_label"], predictions["predicted_label"]) # type: ignore
        bootstrap, _ = subject_stratified_bootstrap(
            predictions, subject_summary, n_replicates
        )
        for metric_name, estimate in estimates.items():
            lower, upper = confidence_interval(bootstrap[metric_name])
            rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "estimate": estimate,
                    "lower": lower,
                    "upper": upper,
                    "n_subjects": int(predictions["subject"].nunique()),
                    "n_predictions": int(len(predictions)),
                }
            )
    return pd.DataFrame(rows)


def plot_outer_model_comparison(table: pd.DataFrame, path: Path) -> None:
    """Four-panel forest plot comparing classifiers on identical outer subjects."""
    metric_panels = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_f1", "Macro F1"),
        ("mcc", "MCC"),
    ]
    available_models = set(table["model"].astype(str))
    ordered_models = [name for name in MODEL_ORDER if name in available_models]
    ordered_models.extend(sorted(available_models.difference(ordered_models))) # type: ignore
    model_colours = dict(
        zip(ordered_models, plt.cm.Blues(np.linspace(0.45, 0.9, len(ordered_models)))) # type: ignore
    )
    y = np.arange(len(ordered_models))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, max(8.0, 0.62 * len(ordered_models) + 5.0)),
        sharey=True,
    )
    for ax, (metric_name, metric_label) in zip(axes.flat, metric_panels):
        subset = (
            table[table["metric"] == metric_name]
            .set_index("model")
            .reindex(ordered_models)
        )
        estimates = subset["estimate"].to_numpy(dtype=float)
        lower = subset["lower"].to_numpy(dtype=float)
        upper = subset["upper"].to_numpy(dtype=float)
        for index, model_name in enumerate(ordered_models):
            ax.errorbar(
                estimates[index],
                index,
                xerr=np.array(
                    [
                        [estimates[index] - lower[index]],
                        [upper[index] - estimates[index]],
                    ]
                ),
                fmt="o",
                color=model_colours[model_name],
                ecolor=model_colours[model_name],
                elinewidth=2,
                capsize=3,
                markersize=7,
            )
            ax.text(
                min(upper[index] + 0.018, 1.04),
                index,
                f"{estimates[index]:.3f}",
                va="center",
                fontsize=8.5,
                color=COLOURS["dark"],
            )
        ax.set_yticks(y, ordered_models)
        ax.set_title(metric_label)
        ax.set_xlabel("Outer-LOSO score (95% CI)")
        if metric_name == "mcc":
            minimum = min(-0.10, float(np.nanmin(lower)) - 0.05)
            ax.set_xlim(max(-1.0, minimum), 1.08)
            ax.axvline(0, color=COLOURS["muted"], linewidth=1, linestyle="--")
        else:
            ax.set_xlim(-0.03, 1.08)
        ax.grid(axis="x")
    fig.suptitle(
        "Unbiased classifier comparison on held-out subjects\n"
        "Sensor combination and feature count selected independently within each outer training set",
        fontsize=14,
        fontweight="semibold",
    )
    save_figure(fig, path)


def save_figure(fig: plt.Figure, path: Path) -> None: # type: ignore
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_overall_metrics(
    estimates: Dict[str, float], bootstrap: pd.DataFrame, path: Path
) -> pd.DataFrame:
    labels = {
        "accuracy": "Accuracy",
        "balanced_accuracy": "Balanced accuracy",
        "macro_precision": "Macro precision",
        "macro_recall": "Macro recall",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "mcc": "MCC",
    }
    rows = []
    for key, label in labels.items():
        lower, upper = confidence_interval(bootstrap[key])
        rows.append(
            {"metric": key, "label": label, "estimate": estimates[key], "lower": lower, "upper": upper}
        )
    table = pd.DataFrame(rows).sort_values("estimate")
    y = np.arange(len(table))
    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    ax.errorbar(
        table["estimate"],
        y,
        xerr=np.vstack([table["estimate"] - table["lower"], table["upper"] - table["estimate"]]),
        fmt="o",
        color=COLOURS["blue"],
        ecolor=COLOURS["muted"],
        capsize=4,
        markersize=7,
    )
    for index, row in table.reset_index(drop=True).iterrows():
        ax.text(
            min(float(row["upper"]) + 0.015, 1.03),
            index,
            f"{row['estimate']:.3f} [{row['lower']:.3f}, {row['upper']:.3f}]",
            va="center",
            fontsize=8.7,
            color=COLOURS["dark"],
        )
    ax.set_yticks(y, table["label"])
    ax.set_xlim(min(-0.05, float(table["lower"].min()) - 0.05), 1.08)
    ax.set_xlabel("Score and 95% subject bootstrap interval")
    ax.set_title("Outer LOSO performance on held-out subjects")
    save_figure(fig, path)
    return table


def per_class_estimates(predictions: pd.DataFrame) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=CLASS_LABELS,
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label": CLASS_LABELS,
            "class": [CLASS_NAMES[label] for label in CLASS_LABELS],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def plot_per_class_metrics(
    estimates: pd.DataFrame, bootstrap: pd.DataFrame, path: Path
) -> pd.DataFrame:
    interval_rows = []
    for _, row in estimates.iterrows():
        subset = bootstrap[bootstrap["label"] == row["label"]]
        for metric_name in ("precision", "recall", "f1"):
            lower, upper = confidence_interval(subset[metric_name])
            interval_rows.append(
                {
                    "label": int(row["label"]),
                    "class": row["class"],
                    "metric": metric_name,
                    "estimate": float(row[metric_name]),
                    "lower": lower,
                    "upper": upper,
                    "support": int(row["support"]),
                }
            )
    table = pd.DataFrame(interval_rows)
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    x = np.arange(len(CLASS_LABELS))
    offsets = {"precision": -0.22, "recall": 0.0, "f1": 0.22}
    colours = {"precision": "#4C78A8", "recall": "#F2A541", "f1": "#59A14F"}
    for metric_name in ("precision", "recall", "f1"):
        subset = table[table["metric"] == metric_name].sort_values("label")
        positions = x + offsets[metric_name]
        ax.errorbar(
            positions,
            subset["estimate"],
            yerr=np.vstack([subset["estimate"] - subset["lower"], subset["upper"] - subset["estimate"]]),
            fmt="o",
            color=colours[metric_name],
            capsize=3,
            markersize=7,
            label=metric_name.capitalize(),
        )
    tick_labels = [
        f"{row['class']}\n(n={int(row['support'])})" for _, row in estimates.sort_values("label").iterrows()
    ]
    ax.set_xticks(x, tick_labels)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Score and 95% subject bootstrap interval")
    ax.set_title("Per-class outer LOSO performance")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.grid(axis="y")
    save_figure(fig, path)
    return table


def plot_confusion(predictions: pd.DataFrame, path: Path) -> pd.DataFrame:
    cm = confusion_matrix(
        predictions["true_label"], predictions["predicted_label"], labels=CLASS_LABELS
    )
    row_total = cm.sum(axis=1, keepdims=True)
    percentages = np.divide(cm, row_total, out=np.zeros_like(cm, dtype=float), where=row_total != 0)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(percentages, cmap="Blues", vmin=0, vmax=1)
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            colour = "white" if percentages[row, column] >= 0.55 else COLOURS["dark"]
            ax.text(
                column,
                row,
                f"{cm[row, column]}\n{percentages[row, column] * 100:.1f}%",
                ha="center",
                va="center",
                color=colour,
                fontweight="semibold",
            )
    names = [CLASS_NAMES[label] for label in CLASS_LABELS]
    ax.set_xticks(range(3), names, rotation=20, ha="right")
    ax.set_yticks(range(3), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Outer LOSO confusion matrix\nCounts and row percentages")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Row proportion")
    ax.grid(False)
    save_figure(fig, path)
    rows = []
    for row, true_label in enumerate(CLASS_LABELS):
        for column, predicted_label in enumerate(CLASS_LABELS):
            rows.append(
                {
                    "true_label": true_label,
                    "true_class": CLASS_NAMES[true_label],
                    "predicted_label": predicted_label,
                    "predicted_class": CLASS_NAMES[predicted_label],
                    "count": int(cm[row, column]),
                    "row_percentage": float(percentages[row, column]),
                }
            )
    return pd.DataFrame(rows)


def plot_subject_outcomes(predictions: pd.DataFrame, path: Path) -> pd.DataFrame:
    summary = (
        predictions.groupby("subject", as_index=False)
        .agg(accuracy=("correct", "mean"), correct=("correct", "sum"), n_predictions=("correct", "size"))
        .sort_values(["accuracy", "subject"])
    )
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    y = np.arange(len(summary))
    colours = np.where(summary["accuracy"] == 1.0, "#59A14F", np.where(summary["accuracy"] == 0.0, "#E45756", "#F2A541"))
    ax.scatter(summary["accuracy"], y, c=colours, s=55, edgecolor="white", linewidth=0.7)
    ax.hlines(y, 0, summary["accuracy"], color=COLOURS["grid"], linewidth=1.5)
    ax.set_yticks(y, summary["subject"])
    ax.set_xlim(-0.03, 1.05)
    ax.set_xlabel("Held-out subject accuracy")
    ax.set_ylabel("Subject")
    ax.set_title("Outer LOSO outcome for each held-out subject")
    save_figure(fig, path)
    return summary


def plot_outer_prediction_grid(predictions: pd.DataFrame, path: Path) -> None:
    """Show every held-out prediction as a correct/incorrect subject-level tile."""
    ordered = predictions.sort_values(["subject", "sample_index"]).copy()
    subjects = list(dict.fromkeys(ordered["subject"].astype(str)))
    grouped = {
        subject: group.reset_index(drop=True)
        for subject, group in ordered.groupby("subject", sort=False)
    }
    maximum = max(len(group) for group in grouped.values())
    matrix = np.full((len(subjects), maximum), np.nan)
    annotations = np.full((len(subjects), maximum), "", dtype=object)
    short_names = {
        "Healthy leg": "Healthy",
        "Affected side": "Affected",
        "Non-affected side": "Non-affected",
    }
    for row, subject in enumerate(subjects):
        group = grouped[subject]
        for column, result in group.iterrows():
            matrix[row, column] = int(result["correct"])
            true_name = short_names.get(str(result["true_class"]), str(result["true_class"]))
            predicted_name = short_names.get(
                str(result["predicted_class"]), str(result["predicted_class"])
            )
            annotations[row, column] = (
                f"{true_name}\n→ {predicted_name}"
            )
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(
        figsize=(8.8, max(6.5, 0.42 * len(subjects) + 2.2))
    )
    masked = np.ma.masked_invalid(matrix)
    ax.imshow(
        masked,
        cmap=ListedColormap(["#E45756", "#59A14F"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if np.isfinite(matrix[row, column]):
                ax.text(
                    column,
                    row,
                    annotations[row, column],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="semibold",
                )
    ax.set_yticks(np.arange(len(subjects)), subjects)
    ax.set_xticks(np.arange(maximum), [f"Observation {i + 1}" for i in range(maximum)])
    ax.set_xlabel("Held-out observation")
    ax.set_ylabel("Held-out subject")
    ax.set_title("Outer-LOSO prediction outcomes by subject\nGreen = correct; red = incorrect")
    ax.grid(False)
    save_figure(fig, path)


def plot_selected_configuration_by_subject(
    selections: pd.DataFrame, predictions: pd.DataFrame, path: Path
) -> None:
    """Display the inner-CV-selected model, sensors and feature set per outer fold."""
    outcomes = (
        predictions.groupby(["outer_fold", "subject"], as_index=False)
        .agg(outer_accuracy=("correct", "mean"))
        .rename(columns={"subject": "held_out_subject"})
    )
    table = selections.merge(
        outcomes, on=["outer_fold", "held_out_subject"], how="left"
    ).sort_values("outer_fold")
    columns = [
        ("model", "Selected model"),
        ("sensor_combination", "Selected sensors"),
        ("feature_set", "Selected feature set"),
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, max(7.5, 0.38 * len(table) + 2.4)),
        sharey=True,
    )
    y = np.arange(len(table))
    for ax, (column, title) in zip(axes, columns):
        categories = list(dict.fromkeys(table[column].astype(str)))
        category_position = {value: index for index, value in enumerate(categories)}
        x = table[column].astype(str).map(category_position).to_numpy()
        colours = plt.cm.RdYlGn(table["outer_accuracy"].fillna(0.5).to_numpy()) # type: ignore
        ax.scatter(x, y, c=colours, s=55, edgecolor="white", linewidth=0.6)
        ax.set_xticks(np.arange(len(categories)), categories, rotation=35, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35)
    axes[0].set_yticks(y, table["held_out_subject"].astype(str))
    axes[0].set_ylabel("Held-out subject")
    fig.suptitle(
        "Configuration selected independently within each outer training set\n"
        "Point colour indicates held-out subject accuracy",
        fontsize=14,
        fontweight="semibold",
    )
    save_figure(fig, path)


def plot_selection_frequency(selections: pd.DataFrame, path: Path) -> None:
    columns = [
        ("model", "Selected model"),
        ("sensor_combination", "Selected sensor combination"),
        ("feature_set", "Selected feature set"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for ax, (column, title) in zip(axes, columns):
        counts = selections[column].astype(str).value_counts().sort_values()
        ax.barh(counts.index, counts.values, color=COLOURS["blue"])
        for index, value in enumerate(counts.values):
            ax.text(value + 0.15, index, str(int(value)), va="center", fontsize=9)
        ax.set_title(title)
        ax.set_xlabel("Number of outer folds")
        ax.set_xlim(0, max(counts.max() * 1.18, 1))
    fig.suptitle("Configuration-selection frequency across outer folds", fontsize=14, fontweight="semibold")
    save_figure(fig, path)


def plot_model_sensor_selection_matrix(selections: pd.DataFrame, path: Path) -> None:
    """Count how often each model–sensor pairing wins an outer-fold inner CV."""
    matrix = pd.crosstab(
        selections["model"].astype(str),
        selections["sensor_combination"].astype(str),
    )
    rows = [name for name in MODEL_ORDER if name in matrix.index]
    columns = [name for name in SENSOR_ORDER if name in matrix.columns]
    matrix = matrix.reindex(index=rows, columns=columns, fill_value=0)
    fig, ax = plt.subplots(figsize=(11.8, 6.3))
    maximum = max(1, int(matrix.to_numpy().max()))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues", vmin=0, vmax=maximum, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix.iloc[row, column])
            colour = "white" if value >= maximum * 0.55 else COLOURS["dark"]
            ax.text(column, row, str(value), ha="center", va="center", color=colour)
    ax.set_xticks(np.arange(len(columns)), columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xlabel("Selected sensor combination")
    ax.set_ylabel("Selected model")
    ax.set_title("Model–sensor selection frequency across outer folds")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="Outer folds selected")
    save_figure(fig, path)


def plot_feature_stability(selections: pd.DataFrame, path: Path) -> pd.DataFrame:
    counts = Counter()
    for value in selections["selected_features"].fillna(""):
        counts.update(feature for feature in str(value).split("|") if feature)
    table = pd.DataFrame(counts.items(), columns=["feature", "outer_folds_selected"])
    table = table.sort_values(["outer_folds_selected", "feature"], ascending=[False, True])
    shown = table.head(20).sort_values("outer_folds_selected")
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.barh(shown["feature"], shown["outer_folds_selected"], color=COLOURS["blue"])
    ax.set_xlabel("Number of winning outer-fold configurations containing feature")
    ax.set_title("Feature-selection stability across outer folds\nTop 20 fold-fitted ANOVA features")
    save_figure(fig, path)
    return table


def plot_inner_macro_f1_heatmaps(candidate_scores: pd.DataFrame, directory: Path) -> None:
    averaged = (
        candidate_scores.groupby(["feature_set", "model", "sensor_combination"], as_index=False)["mean_macro_f1"]
        .mean()
    )
    for feature_set, subset in averaged.groupby("feature_set", sort=False):
        pivot = subset.pivot(index="model", columns="sensor_combination", values="mean_macro_f1")
        rows = [name for name in MODEL_ORDER if name in pivot.index]
        columns = [name for name in SENSOR_ORDER if name in pivot.columns]
        pivot = pivot.reindex(index=rows, columns=columns)
        fig, ax = plt.subplots(figsize=(11.5, 6.2))
        image = ax.imshow(pivot.to_numpy(), cmap="Blues", vmin=max(0, np.nanmin(pivot.to_numpy()) - 0.03), vmax=min(1, np.nanmax(pivot.to_numpy()) + 0.03), aspect="auto")
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = pivot.iloc[row, column]
                if pd.notna(value):
                    ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
        ax.set_yticks(range(len(rows)), rows)
        ax.set_title(f"Mean inner CV macro F1 by model and sensor\n{feature_set} — selection diagnostic only")
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="Mean inner CV macro F1")
        safe = re.sub(r"[^a-z0-9]+", "_", feature_set.lower()).strip("_") # type: ignore
        save_figure(fig, directory / f"supp_inner_macro_f1_{safe}.png")


def averaged_inner_macro_f1(candidate_scores: pd.DataFrame) -> pd.DataFrame:
    """Average each candidate's inner-CV macro F1 across outer training sets."""
    return (candidate_scores.groupby(["sensor_combination", "feature_set", "model"], as_index=False)["mean_macro_f1"].mean().rename(columns={"mean_macro_f1": "macro_f1"})) # type: ignore


def plot_best_model_inner_macro_f1(candidate_scores: pd.DataFrame, path: Path) -> None:
    """Show the best inner-CV model for every sensor and feature-set pairing."""
    diagnostic = averaged_inner_macro_f1(candidate_scores)
    best = diagnostic.loc[
        diagnostic.groupby(["sensor_combination", "feature_set"])["macro_f1"].idxmax()
    ].copy()
    feature_sets = list(dict.fromkeys(candidate_scores["feature_set"].astype(str)))
    model_names = [name for name in MODEL_ORDER if name in set(best["model"])]
    model_colours = dict(
        zip(model_names, plt.cm.tab10(np.linspace(0, 1, max(1, len(model_names))))) # type: ignore
    )
    fig, axes = plt.subplots(
        1, len(feature_sets), figsize=(4.2 * len(feature_sets), 6.6), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    sensors = [name for name in SENSOR_ORDER if name in set(best["sensor_combination"])]
    y = np.arange(len(sensors))
    for ax, feature_set in zip(axes, feature_sets):
        subset = best[best["feature_set"] == feature_set].set_index("sensor_combination")
        for index, sensor in enumerate(sensors):
            if sensor not in subset.index:
                continue
            row = subset.loc[sensor]
            ax.scatter(
                float(row["macro_f1"]),
                index,
                s=65,
                color=model_colours[str(row["model"])],
                edgecolor="white",
                linewidth=0.6,
            )
            ax.text(float(row["macro_f1"]) + 0.008, index, f"{float(row['macro_f1']):.3f}", va="center", fontsize=8)
        ax.set_title(feature_set)
        ax.set_xlabel("Mean inner-CV macro F1")
        ax.set_xlim(0, 1.04)
    axes[0].set_yticks(y, sensors)
    axes[0].set_ylabel("Sensor combination")
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colour, label=model) # type: ignore
        for model, colour in model_colours.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle(
        "Best model for each sensor and feature-set combination\n"
        "Inner-CV macro F1 — selection diagnostic only",
        fontsize=14,
        fontweight="semibold",
    )
    fig.subplots_adjust(bottom=0.16)
    save_figure(fig, path)


def plot_model_ranking_inner_macro_f1(candidate_scores: pd.DataFrame, path: Path) -> None:
    """Rank models by mean inner-CV macro F1 across sensor and feature conditions."""
    diagnostic = averaged_inner_macro_f1(candidate_scores)
    ranking = diagnostic.groupby("model", as_index=False)["macro_f1"].mean().sort_values("macro_f1") # type: ignore
    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    y = np.arange(len(ranking))
    ax.hlines(y, 0, ranking["macro_f1"], color=COLOURS["grid"], linewidth=2)
    ax.scatter(ranking["macro_f1"], y, color=COLOURS["blue"], s=65)
    for index, value in enumerate(ranking["macro_f1"]):
        ax.text(float(value) + 0.008, index, f"{float(value):.3f}", va="center", fontsize=9)
    ax.set_yticks(y, ranking["model"])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Mean inner-CV macro F1")
    ax.set_title("Average model ranking\nInner-CV selection diagnostic only")
    save_figure(fig, path)


def plot_feature_set_inner_macro_f1(candidate_scores: pd.DataFrame, path: Path) -> None:
    """Compare feature sets after retaining the best model for each sensor pairing."""
    diagnostic = averaged_inner_macro_f1(candidate_scores)
    best = diagnostic.loc[
        diagnostic.groupby(["sensor_combination", "feature_set"])["macro_f1"].idxmax()
    ]
    comparison = best.groupby("feature_set", as_index=False)["macro_f1"].mean().sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8.5, 4.9))
    y = np.arange(len(comparison))
    ax.hlines(y, 0, comparison["macro_f1"], color=COLOURS["grid"], linewidth=2)
    ax.scatter(comparison["macro_f1"], y, color=COLOURS["blue"], s=70)
    for index, value in enumerate(comparison["macro_f1"]):
        ax.text(float(value) + 0.008, index, f"{float(value):.3f}", va="center", fontsize=9)
    ax.set_yticks(y, comparison["feature_set"])
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("Mean best-model inner-CV macro F1 across sensors")
    ax.set_title("Feature-set comparison\nInner-CV selection diagnostic only")
    save_figure(fig, path)


def plot_sensor_feature_inner_macro_f1(candidate_scores: pd.DataFrame, path: Path) -> None:
    """Matrix of the best-model inner-CV macro F1 for each sensor/feature pair."""
    diagnostic = averaged_inner_macro_f1(candidate_scores)
    best = diagnostic.groupby(["sensor_combination", "feature_set"], as_index=False)["macro_f1"].max()
    matrix = best.pivot(index="sensor_combination", columns="feature_set", values="macro_f1")
    rows = [name for name in SENSOR_ORDER if name in matrix.index]
    feature_sets = list(dict.fromkeys(candidate_scores["feature_set"].astype(str)))
    matrix = matrix.reindex(index=rows, columns=feature_sets)
    fig, ax = plt.subplots(figsize=(9.8, 6.3))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(
        values,
        cmap="Blues",
        vmin=max(0, float(np.nanmin(values)) - 0.03),
        vmax=min(1, float(np.nanmax(values)) + 0.03),
        aspect="auto",
    )
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{float(value):.3f}", ha="center", va="center", fontsize=8.5)
    ax.set_xticks(np.arange(len(feature_sets)), feature_sets, rotation=25, ha="right") # type: ignore
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Sensor combination")
    ax.set_title("Best-model macro F1 by sensor and feature set\nInner-CV selection diagnostic only")
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="Mean inner-CV macro F1")
    save_figure(fig, path)


def write_manifest(output: Path) -> None:
    text = """JOURNAL FIGURE SET

Primary figures
1. primary_01_outer_loso_metrics_with_ci.png
2. primary_02_outer_loso_per_class_with_ci.png
3. primary_03_outer_loso_confusion_matrix.png
4. primary_04_outer_loso_model_comparison.png

Optional main-text or supplementary transparency figure
5. supplementary_01_outer_subject_outcomes.png

Selection diagnostics for supplementary material
6. supplementary_02_selection_frequency.png
7. supplementary_03_feature_stability.png
8. supplementary_04_outer_prediction_grid.png
9. supplementary_05_selected_configuration_by_subject.png
10. supplementary_06_model_sensor_selection_matrix.png
11. supplementary_07_best_model_inner_cv_macro_f1.png
12. supplementary_08_model_ranking_inner_cv_macro_f1.png
13. supplementary_09_feature_set_comparison_inner_cv_macro_f1.png
14. supplementary_10_sensor_feature_matrix_inner_cv_macro_f1.png
15. supp_inner_macro_f1_*.png

Interpretation rule
Only the primary outer-LOSO figures estimate final performance on unseen
subjects. Inner-CV heatmaps and selection-frequency figures are diagnostic and
must not be reported as independent test performance.
"""
    (output / "FIGURE_MANIFEST.txt").write_text(text, encoding="utf-8")


def run(data_path: Path, output: Path, bootstrap_replicates: int) -> None:
    configure_plot_style()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "journal_figures"
    primary = figures / "primary"
    supplementary = figures / "supplementary"
    tables = output / "tables"
    for directory in (primary, supplementary, tables):
        directory.mkdir(parents=True, exist_ok=True)

    df = read_dataset(data_path)
    groups = df[ID_COLUMN].map(infer_subject).to_numpy(dtype=str)
    subject_summary = validate_subject_structure(df, groups)
    sensor_columns = get_sensor_columns(df)
    missing_sensors = [name for name, columns in sensor_columns.items() if not columns]
    if missing_sensors:
        warnings.warn(f"No matching columns for: {missing_sensors}")
    models = build_models()
    (
        predictions,
        selections,
        candidates,
        fold_scores,
        model_predictions,
    ) = nested_evaluation(df, groups, sensor_columns, models)

    final_metrics = metrics(predictions["true_label"], predictions["predicted_label"]) # type: ignore
    bootstrap_overall, bootstrap_classes = subject_stratified_bootstrap(
        predictions, subject_summary, bootstrap_replicates
    )
    overall_table = plot_overall_metrics(
        final_metrics,
        bootstrap_overall,
        primary / "primary_01_outer_loso_metrics_with_ci.png",
    )
    class_estimates = per_class_estimates(predictions)
    class_table = plot_per_class_metrics(
        class_estimates,
        bootstrap_classes,
        primary / "primary_02_outer_loso_per_class_with_ci.png",
    )
    confusion_table = plot_confusion(
        predictions, primary / "primary_03_outer_loso_confusion_matrix.png"
    )
    model_comparison_table = model_comparison_with_intervals(
        model_predictions, subject_summary, bootstrap_replicates
    )
    plot_outer_model_comparison(
        model_comparison_table,
        primary / "primary_04_outer_loso_model_comparison.png",
    )
    subject_table = plot_subject_outcomes(
        predictions, supplementary / "supplementary_01_outer_subject_outcomes.png"
    )
    plot_selection_frequency(
        selections, supplementary / "supplementary_02_selection_frequency.png"
    )
    stability_table = plot_feature_stability(
        selections, supplementary / "supplementary_03_feature_stability.png"
    )
    plot_outer_prediction_grid(
        predictions, supplementary / "supplementary_04_outer_prediction_grid.png"
    )
    plot_selected_configuration_by_subject(
        selections,
        predictions,
        supplementary / "supplementary_05_selected_configuration_by_subject.png",
    )
    plot_model_sensor_selection_matrix(
        selections,
        supplementary / "supplementary_06_model_sensor_selection_matrix.png",
    )
    plot_best_model_inner_macro_f1(
        candidates,
        supplementary / "supplementary_07_best_model_inner_cv_macro_f1.png",
    )
    plot_model_ranking_inner_macro_f1(
        candidates,
        supplementary / "supplementary_08_model_ranking_inner_cv_macro_f1.png",
    )
    plot_feature_set_inner_macro_f1(
        candidates,
        supplementary / "supplementary_09_feature_set_comparison_inner_cv_macro_f1.png",
    )
    plot_sensor_feature_inner_macro_f1(
        candidates,
        supplementary / "supplementary_10_sensor_feature_matrix_inner_cv_macro_f1.png",
    )
    plot_inner_macro_f1_heatmaps(candidates, supplementary)

    predictions.to_csv(tables / "outer_loso_predictions.csv", index=False)
    selections.to_csv(tables / "outer_fold_selected_configurations.csv", index=False)
    candidates.to_csv(tables / "inner_cv_candidate_mean_scores.csv", index=False)
    fold_scores.to_csv(tables / "inner_cv_fold_scores.csv", index=False)
    model_predictions.to_csv(
        tables / "outer_loso_predictions_by_model.csv", index=False
    )
    subject_summary.to_csv(tables / "subject_structure.csv", index=False)
    overall_table.to_csv(tables / "outer_loso_metrics_with_ci.csv", index=False)
    class_table.to_csv(tables / "outer_loso_per_class_metrics_with_ci.csv", index=False)
    confusion_table.to_csv(tables / "outer_loso_confusion_matrix.csv", index=False)
    model_comparison_table.to_csv(
        tables / "outer_loso_model_comparison_with_ci.csv", index=False
    )
    subject_table.to_csv(tables / "outer_loso_subject_outcomes.csv", index=False)
    stability_table.to_csv(tables / "selected_feature_stability.csv", index=False)
    pd.DataFrame([{"validation": "outer LOSO with inner 5-fold StratifiedGroupKFold", "selection_metric": PRIMARY_METRIC, "n_subjects": subject_summary.shape[0], "bootstrap_replicates": bootstrap_replicates, **final_metrics}]).to_csv(
        tables / "final_outer_loso_performance.csv", index=False
    )
    (tables / "final_classification_report.txt").write_text(
        classification_report(predictions["true_label"], predictions["predicted_label"], labels=CLASS_LABELS, target_names=[CLASS_NAMES[label] for label in CLASS_LABELS], zero_division=0,), encoding="utf-8", # type: ignore
    )
    metadata = {
        "data_path": str(data_path),
        "n_rows": len(df),
        "n_subjects": int(subject_summary.shape[0]),
        "n_healthy_subjects": int((subject_summary["phenotype"] == "healthy").sum()),
        "n_stroke_subjects": int((subject_summary["phenotype"] == "stroke").sum()),
        "outer_validation": "Leave-One-Subject-Out",
        "inner_validation": f"{INNER_SPLITS}-fold StratifiedGroupKFold",
        "selection_metric": PRIMARY_METRIC,
        "feature_selection": "ANOVA F-score fitted independently in each training fold",
        "bootstrap": "Subject-stratified percentile bootstrap",
        "bootstrap_replicates": bootstrap_replicates,
        "models": list(models),
        "sensor_combinations": {name: len(columns) for name, columns in sensor_columns.items()},
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_manifest(output)
    print("\nFinal outer-LOSO metrics")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.4f}")
    print(f"\nSaved results to {output}")


def rebuild_plots(output: Path, bootstrap_replicates: int) -> None:
    """Recreate every journal figure from previously saved result tables."""
    configure_plot_style()
    figures = output / "journal_figures"
    primary = figures / "primary"
    supplementary = figures / "supplementary"
    tables = output / "tables"
    for directory in (primary, supplementary):
        directory.mkdir(parents=True, exist_ok=True)

    required = {
        "predictions": tables / "outer_loso_predictions.csv",
        "selections": tables / "outer_fold_selected_configurations.csv",
        "candidates": tables / "inner_cv_candidate_mean_scores.csv",
        "subjects": tables / "subject_structure.csv",
        "model_predictions": tables / "outer_loso_predictions_by_model.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Plot-only mode requires these saved tables:\n  - " + "\n  - ".join(missing)
        )

    predictions = pd.read_csv(required["predictions"])
    selections = pd.read_csv(required["selections"])
    candidates = pd.read_csv(required["candidates"])
    subject_summary = pd.read_csv(required["subjects"])
    model_predictions = pd.read_csv(required["model_predictions"])

    final_metrics = metrics(predictions["true_label"], predictions["predicted_label"] # type: ignore
    )
    
    bootstrap_overall, bootstrap_classes = subject_stratified_bootstrap(predictions, subject_summary, bootstrap_replicates)
    plot_overall_metrics(final_metrics, bootstrap_overall, primary / "primary_01_outer_loso_metrics_with_ci.png")
    plot_per_class_metrics(per_class_estimates(predictions), bootstrap_classes, primary / "primary_02_outer_loso_per_class_with_ci.png")
    plot_confusion(predictions, primary / "primary_03_outer_loso_confusion_matrix.png")
    model_comparison_table = model_comparison_with_intervals(model_predictions, subject_summary, bootstrap_replicates)
    plot_outer_model_comparison(model_comparison_table, primary / "primary_04_outer_loso_model_comparison.png")
    plot_subject_outcomes(predictions, supplementary / "supplementary_01_outer_subject_outcomes.png")
    plot_selection_frequency(selections, supplementary / "supplementary_02_selection_frequency.png")
    plot_feature_stability(selections, supplementary / "supplementary_03_feature_stability.png")
    plot_outer_prediction_grid(predictions, supplementary / "supplementary_04_outer_prediction_grid.png")
    plot_selected_configuration_by_subject(selections, predictions, supplementary / "supplementary_05_selected_configuration_by_subject.png")
    plot_model_sensor_selection_matrix(selections, supplementary / "supplementary_06_model_sensor_selection_matrix.png")
    plot_best_model_inner_macro_f1(candidates, supplementary / "supplementary_07_best_model_inner_cv_macro_f1.png")
    plot_model_ranking_inner_macro_f1(candidates, supplementary / "supplementary_08_model_ranking_inner_cv_macro_f1.png")
    plot_feature_set_inner_macro_f1(candidates,supplementary / "supplementary_09_feature_set_comparison_inner_cv_macro_f1.png")
    plot_sensor_feature_inner_macro_f1(candidates, supplementary / "supplementary_10_sensor_feature_matrix_inner_cv_macro_f1.png")
    plot_inner_macro_f1_heatmaps(candidates, supplementary)
    write_manifest(output)
    print(f"Rebuilt journal figures from saved tables in {figures}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("features_dataset.csv"))
    parser.add_argument( "--output", type=Path, default=Path("outputs/ml_results_nested_journal"))
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--plots-only", action="store_true", help="Rebuild all figures from existing tables without refitting models.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.plots_only:
        rebuild_plots(arguments.output, arguments.bootstrap_replicates)
    else:
        run(arguments.data, arguments.output, arguments.bootstrap_replicates)