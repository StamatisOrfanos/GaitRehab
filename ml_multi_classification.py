"""
Machine learning comparison script for 3-class gait classification.

Classes:
    0 = Healthy leg
    1 = Affected side
    2 = Non-affected side

Input:
    data.csv

Optional input from feature-selection step:
    outputs/eda_plots/feature_selection/selected_features_for_classification.csv

This script evaluates each sensor combination in three modes:
    1. All features
    2. Top 5 selected features from RFE
    3. Top 10 selected features from RFE

Validation:
    - Leave-One-Subject-Out if subject groups can be inferred from ID
    - Otherwise row-level Leave-One-Out, with warning

Outputs:
    outputs/ml_results_loocv_top5_top10_all/
        final_model_comparison.csv
        best_model_per_sensor_combination_and_feature_set.csv
        loo_predictions.csv
        classification_reports/
        confusion_matrices/
        plots/
        selected_feature_lists/

Sensor combinations:
    1. Gyroscope
    2. Accelerometer
    3. EMG
    4. Gyroscope + Accelerometer
    5. Gyroscope + EMG
    6. Accelerometer + EMG
    7. All sensors
"""

from __future__ import annotations

import re
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data.csv")

SELECTED_FEATURES_PATH = Path(
    "outputs/eda_plots/feature_selection/selected_features_for_classification.csv"
)

OUTPUT_DIR = Path("outputs/ml_results_loocv_top5_top10_all")

ID_COLUMN = "ID"
LABEL_COLUMN = "Label"

RANDOM_STATE = 42

SELECTED_FEATURE_COUNTS = [5, 10]

CLASS_NAMES = {
    0: "Healthy leg",
    1: "Affected side",
    2: "Non-affected side",
}

CLASS_LABELS = [0, 1, 2]
CLASS_LABEL_NAMES = [CLASS_NAMES[label] for label in CLASS_LABELS]

GYRO_PREFIXES = ["gyrox", "gyroy", "gyroz"]
ACC_PREFIXES = ["accx", "accy", "accz"]
EMG_PREFIXES = ["GMinter", "RFinter", "BFinter", "MGinter", "TAinter", "PLinter"]

SENSOR_PREFIXES = {
    "Gyroscope": GYRO_PREFIXES,
    "Accelerometer": ACC_PREFIXES,
    "EMG": EMG_PREFIXES,
    "Gyroscope + Accelerometer": GYRO_PREFIXES + ACC_PREFIXES,
    "Gyroscope + EMG": GYRO_PREFIXES + EMG_PREFIXES,
    "Accelerometer + EMG": ACC_PREFIXES + EMG_PREFIXES,
    "All sensors": GYRO_PREFIXES + ACC_PREFIXES + EMG_PREFIXES,
}

SOURCE_ORDER = [
    "Gyroscope",
    "Accelerometer",
    "EMG",
    "Gyroscope + Accelerometer",
    "Gyroscope + EMG",
    "Accelerometer + EMG",
    "All sensors",
]

FEATURE_SET_ORDER = [
    "All features",
    "Top 5 selected features",
    "Top 10 selected features",
]


# =============================================================================
# Plot styling
# =============================================================================

plt.rcParams.update(
    {
        "figure.figsize": (11, 7),
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


# =============================================================================
# Utility functions
# =============================================================================

def ensure_directories() -> None:
    directories = [
        OUTPUT_DIR,
        OUTPUT_DIR / "plots",
        OUTPUT_DIR / "confusion_matrices",
        OUTPUT_DIR / "classification_reports",
        OUTPUT_DIR / "selected_feature_lists",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    name = name.lower()
    name = name.replace("+", "plus")
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put this script in the same folder as data.csv "
            f"or change DATA_PATH."
        )

    df = pd.read_csv(path, sep=None, engine="python")

    # Remove fully empty columns and Excel-export blank columns.
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Clean column names.
    df.columns = [str(col).strip() for col in df.columns]

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected label column '{LABEL_COLUMN}', but found:\n{df.columns.tolist()}" # type: ignore
        )

    if ID_COLUMN not in df.columns:
        warnings.warn(
            f"Expected ID column '{ID_COLUMN}' was not found. "
            f"Leave-One-Subject-Out will not be possible."
        )

    # Make labels numeric.
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    # Drop rows without labels.
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # Convert feature columns to numeric.
    metadata_columns = {ID_COLUMN, LABEL_COLUMN}

    for col in df.columns:
        if col not in metadata_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def get_feature_columns(df: pd.DataFrame, prefixes: List[str]) -> List[str]:
    feature_columns = []

    for col in df.columns:
        if col in {ID_COLUMN, LABEL_COLUMN}:
            continue

        for prefix in prefixes:
            if str(col).startswith(prefix + "_"):
                feature_columns.append(col)
                break

    return feature_columns


def get_sensor_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        source_name: get_feature_columns(df, prefixes)
        for source_name, prefixes in SENSOR_PREFIXES.items()
    }


def clean_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    existing_features = [col for col in feature_columns if col in df.columns]

    if len(existing_features) == 0:
        return pd.DataFrame(index=df.index), df[LABEL_COLUMN].copy()

    X = df[existing_features].copy()
    y = df[LABEL_COLUMN].copy()

    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop features that are entirely missing.
    X = X.dropna(axis=1, how="all")

    # Drop constant columns.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    return X, y


# =============================================================================
# Subject grouping
# =============================================================================

def infer_subject_group_from_id(raw_id: object) -> str:
    """
    Attempts to infer subject-level grouping from sample ID.

    This is important because left/right legs from the same subject should not
    be split across train and test.

    Examples handled:
        H01_left        -> H01
        H01_right       -> H01
        HT01_L          -> HT01
        PT03_affected   -> PT03
        P12_nonaffected -> P12
        Stroke_05_A     -> STROKE05

    If your ID format is different, modify this function.
    """

    text = str(raw_id).strip()

    # Remove common side/leg suffixes.
    text = re.sub(
        r"(?i)(left|right|affected|nonaffected|non_affected|non-affected|healthy|leg|side)",
        "",
        text,
    )

    # Remove repeated separators.
    text = re.sub(r"[_\-\s]+", "_", text).strip("_")

    # Prefer a leading text prefix plus subject number.
    match = re.search(r"(?i)([a-z]+)[_\- ]*0*([0-9]+)", text)

    if match:
        prefix = match.group(1).upper()
        number = int(match.group(2))
        return f"{prefix}{number:02d}"

    # If only a number exists, return the number.
    match = re.search(r"([0-9]+)", text)

    if match:
        return f"S{int(match.group(1)):02d}"

    return text


def infer_groups(df: pd.DataFrame) -> Optional[np.ndarray]:
    if ID_COLUMN not in df.columns:
        return None

    groups = df[ID_COLUMN].apply(infer_subject_group_from_id).astype(str).values

    n_samples = len(groups)
    n_groups = len(np.unique(groups)) # type: ignore

    print()
    print("=" * 80)
    print("Subject/group inference")
    print("=" * 80)
    print(f"Samples: {n_samples}")
    print(f"Inferred groups: {n_groups}")

    group_counts = pd.Series(groups).value_counts().sort_index()

    print("Group size distribution:")
    print(group_counts.value_counts().sort_index().to_string())

    if n_groups == n_samples:
        print()
        print("[WARNING]")
        print(
            "Every row appears to have a unique group. "
            "This usually means the ID column is a sample ID, not a subject ID."
        )
        print(
            "The script will fall back to row-level Leave-One-Out. "
            "For paired-leg gait data, this is weaker than Leave-One-Subject-Out."
        )
        return None

    return groups # type: ignore


# =============================================================================
# Selected feature loading
# =============================================================================

def load_selected_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        print()
        print("[WARNING]")
        print(f"Selected-features file was not found: {path}")
        print("The script will evaluate only the all-features mode.")
        return pd.DataFrame(columns=["source", "n_features", "feature"])

    selected_df = pd.read_csv(path)

    required_columns = {"source", "n_features", "feature"}
    missing = required_columns - set(selected_df.columns)

    if missing:
        raise ValueError(
            f"Selected-features file is missing columns: {missing}. "
            f"Expected columns: {required_columns}"
        )

    selected_df["source"] = selected_df["source"].astype(str)
    selected_df["feature"] = selected_df["feature"].astype(str)
    selected_df["n_features"] = pd.to_numeric(
        selected_df["n_features"],
        errors="coerce",
    ).astype("Int64")

    return selected_df


def get_top_selected_features_for_source(
    selected_df: pd.DataFrame,
    source_name: str,
    available_features: List[str],
    top_n: int,
) -> List[str]:
    """
    Reads the selected features for one sensor source.

    Preferred:
        source == source_name and n_features == top_n

    Fallback:
        if exact top_n is missing, use the smallest available n_features > top_n.
        if that is missing, use the largest available n_features < top_n.
    """

    if selected_df.empty:
        return []

    source_df = selected_df[selected_df["source"] == source_name].copy()

    if source_df.empty:
        return []

    available_feature_set = set(available_features)

    exact = source_df[source_df["n_features"] == top_n].copy()

    if not exact.empty:
        selected_features = exact["feature"].tolist()
    else:
        available_counts = sorted(
            [
                int(value)
                for value in source_df["n_features"].dropna().unique().tolist()
            ]
        )

        larger_or_equal = [value for value in available_counts if value >= top_n]
        smaller = [value for value in available_counts if value < top_n]

        if larger_or_equal:
            chosen_count = larger_or_equal[0]
        elif smaller:
            chosen_count = smaller[-1]
        else:
            return []

        print(
            f"[WARNING] No exact top-{top_n} selected feature set for {source_name}. "
            f"Using n_features={chosen_count} and taking the first {top_n} features."
        )

        selected_features = source_df[
            source_df["n_features"] == chosen_count
        ]["feature"].tolist()

    selected_features = [
        feature for feature in selected_features
        if feature in available_feature_set
    ]

    selected_features = selected_features[:top_n]

    return selected_features


def build_feature_sets_for_source(
    source_name: str,
    all_features: List[str],
    selected_df: pd.DataFrame,
) -> Dict[str, List[str]]:
    feature_sets = {
        "All features": all_features,
    }

    for selected_count in SELECTED_FEATURE_COUNTS:
        feature_set_name = f"Top {selected_count} selected features"

        selected_features = get_top_selected_features_for_source(
            selected_df=selected_df,
            source_name=source_name,
            available_features=all_features,
            top_n=selected_count,
        )

        if selected_features:
            feature_sets[feature_set_name] = selected_features

            output_path = (
                OUTPUT_DIR /
                "selected_feature_lists" /
                f"selected_features_{safe_filename(source_name)}_top_{selected_count}.txt"
            )

            with open(output_path, "w", encoding="utf-8") as file:
                for feature in selected_features:
                    file.write(f"{feature}\n")

        else:
            print(
                f"[WARNING] No selected top-{selected_count} features found "
                f"for {source_name}. Skipping {feature_set_name}."
            )

    return feature_sets


# =============================================================================
# Model definitions
# =============================================================================

def build_models() -> Dict[str, Pipeline]:
    """
    Common ML models for tabular classification.

    Scaling is included for all models for consistency.
    Tree-based models do not require scaling, but keeping one uniform pipeline
    makes the code simpler.
    """

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        multi_class="auto",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Linear SVM": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        class_weight="balanced",
                        probability=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "RBF SVM": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=False,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "k-NN": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=5,
                        weights="distance",
                    ),
                ),
            ]
        ),
        "Gaussian Naive Bayes": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingClassifier(
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=300,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="multi:softmax",
                        num_class=3,
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        print("[INFO] XGBoost found and included.")

    except ImportError:
        print("[INFO] XGBoost not installed. Skipping XGBoost.")

    return models


# =============================================================================
# Leave-one-out evaluation
# =============================================================================

def make_loo_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
):
    if groups is not None:
        print("[INFO] Using Leave-One-Subject-Out validation.")
        splitter = LeaveOneGroupOut()
        return splitter.split(X, y, groups=groups), "Leave-One-Subject-Out"

    print("[INFO] Using row-level Leave-One-Out validation.")
    splitter = LeaveOneOut()
    return splitter.split(X, y), "Leave-One-Out"


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
    } # type: ignore


def evaluate_model_with_loo(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray, str]:
    split_iterator, validation_method = make_loo_splits(X, y, groups)

    prediction_rows = []
    all_true = []
    all_pred = []

    for fold_idx, (train_idx, test_idx) in enumerate(split_iterator, start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)

        for local_idx, sample_index in enumerate(test_idx):
            true_label = int(y_test.iloc[local_idx])
            predicted_label = int(y_pred[local_idx])

            group_value = None

            if groups is not None:
                group_value = str(groups[sample_index])

            prediction_rows.append(
                {
                    "fold": fold_idx,
                    "sample_index": int(sample_index),
                    "group": group_value,
                    "true_label": true_label,
                    "true_class": CLASS_NAMES.get(true_label, str(true_label)),
                    "predicted_label": predicted_label,
                    "predicted_class": CLASS_NAMES.get(predicted_label, str(predicted_label)),
                    "correct": int(true_label == predicted_label),
                }
            )

        all_true.extend(y_test.values)
        all_pred.extend(y_pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_metrics = evaluate_predictions(all_true, all_pred)
    predictions_df = pd.DataFrame(prediction_rows)

    return overall_metrics, predictions_df, all_pred, validation_method


# =============================================================================
# Plot functions
# =============================================================================

def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_name: str,
    feature_set_name: str,
    model_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(
        f"Confusion matrix\n"
        f"{source_name} — {feature_set_name} — {model_name}"
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(np.arange(len(CLASS_LABEL_NAMES)))
    ax.set_yticks(np.arange(len(CLASS_LABEL_NAMES)))
    ax.set_xticklabels(CLASS_LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_LABEL_NAMES)

    max_value = cm.max() if cm.size > 0 else 1

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text_color = "white" if value > max_value / 2 else "black"

            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontweight="bold",
                fontsize=13,
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Number of samples")

    ax.grid(False)

    filename = (
        f"confusion_matrix_{safe_filename(source_name)}_"
        f"{safe_filename(feature_set_name)}_"
        f"{safe_filename(model_name)}.png"
    )

    output_path = OUTPUT_DIR / "confusion_matrices" / filename

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_metric_heatmap(
    results_df: pd.DataFrame,
    metric: str,
    feature_set_name: str,
    filename: str,
    title: str,
) -> None:
    subset = results_df[results_df["feature_set"] == feature_set_name].copy()

    if subset.empty:
        return

    pivot = subset.pivot(
        index="sensor_combination",
        columns="model",
        values=metric,
    )

    pivot = pivot.reindex(SOURCE_ORDER)

    fig, ax = plt.subplots(figsize=(15, 7))

    image = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Sensor combination")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]

            if pd.isna(value):
                label = "NA"
            else:
                label = f"{value:.2f}"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if not pd.isna(value) and value < 0.65 else "black",
                fontsize=9,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    ax.grid(False)

    output_path = OUTPUT_DIR / "plots" / filename

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_best_model_barplot(
    best_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> None:
    plot_df = best_df.copy()
    plot_df["label"] = plot_df["sensor_combination"] + " | " + plot_df["feature_set"]
    plot_df = plot_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(13, 11))

    bars = ax.barh(
        plot_df["label"],
        plot_df[metric],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, model_name, value in zip(
        bars,
        plot_df["model"],
        plot_df[metric],
    ):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} | {model_name}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Best model per sensor combination and feature set")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Sensor combination | Feature set")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    output_path = OUTPUT_DIR / "plots" / f"best_model_per_combination_and_feature_set_{metric}.png"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_model_ranking_barplot(
    results_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> None:
    model_ranking = (
        results_df.groupby("model")[metric]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    bars = ax.barh(
        model_ranking["model"],
        model_ranking[metric],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, value in zip(bars, model_ranking[metric]):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Average model performance across all experiments")
    ax.set_xlabel(f"Mean {metric.replace('_', ' ').title()}")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    output_path = OUTPUT_DIR / "plots" / f"average_model_ranking_{metric}.png"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_feature_set_comparison_barplot(
    results_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> None:
    best_per_sensor_feature_set = (
        results_df.sort_values(
            by=["macro_f1", "accuracy", "balanced_accuracy"],
            ascending=False,
        )
        .groupby(["sensor_combination", "feature_set"], as_index=False)
        .first()
    )

    best_per_sensor_feature_set["label"] = (
        best_per_sensor_feature_set["sensor_combination"]
        + " | "
        + best_per_sensor_feature_set["feature_set"]
    )

    plot_df = best_per_sensor_feature_set.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(13, 11))

    bars = ax.barh(
        plot_df["label"],
        plot_df[metric],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, model_name, value in zip(
        bars,
        plot_df["model"],
        plot_df[metric],
    ):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f} | {model_name}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Best performance: all features vs top 5 vs top 10")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Experiment")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    output_path = OUTPUT_DIR / "plots" / f"all_vs_top5_vs_top10_best_{metric}.png"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_sensor_feature_set_matrix_plot(
    best_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> None:
    """
    Shows the best model score for each sensor combination and feature-set mode.
    """

    pivot = best_df.pivot(
        index="sensor_combination",
        columns="feature_set",
        values=metric,
    )

    pivot = pivot.reindex(SOURCE_ORDER)
    existing_columns = [col for col in FEATURE_SET_ORDER if col in pivot.columns]
    pivot = pivot[existing_columns]

    fig, ax = plt.subplots(figsize=(10, 7))

    image = ax.imshow(pivot.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")

    ax.set_title(f"Best {metric.replace('_', ' ').title()} by sensor and feature set")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Sensor combination")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))

    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticklabels(pivot.index)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.values[i, j]

            if pd.isna(value):
                label = "NA"
            else:
                label = f"{value:.3f}"

            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if not pd.isna(value) and value < 0.65 else "black",
                fontsize=10,
                fontweight="bold",
            )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric.replace("_", " ").title())

    ax.grid(False)

    output_path = OUTPUT_DIR / "plots" / f"best_{metric}_sensor_by_feature_set_matrix.png"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# =============================================================================
# Report saving
# =============================================================================

def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_name: str,
    feature_set_name: str,
    model_name: str,
) -> None:
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_LABEL_NAMES,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_LABEL_NAMES,
        zero_division=0,
    )

    base_filename = (
        f"classification_report_{safe_filename(source_name)}_"
        f"{safe_filename(feature_set_name)}_"
        f"{safe_filename(model_name)}"
    )

    json_path = OUTPUT_DIR / "classification_reports" / f"{base_filename}.json"
    txt_path = OUTPUT_DIR / "classification_reports" / f"{base_filename}.txt"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report_dict, file, indent=4)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(report_text) # type: ignore


# =============================================================================
# Main ML pipeline
# =============================================================================

def main() -> None:
    ensure_directories()

    print("=" * 80)
    print("Loading dataset")
    print("=" * 80)

    df = read_dataset(DATA_PATH)

    print(f"Dataset shape after basic cleaning: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    print()
    print("=" * 80)
    print("Label distribution")
    print("=" * 80)

    label_counts = df[LABEL_COLUMN].value_counts().sort_index()

    for label, count in label_counts.items():
        print(f"{label} - {CLASS_NAMES.get(int(label), 'Unknown')}: {count}") # type: ignore

    groups = infer_groups(df)

    sensor_columns = get_sensor_columns(df)

    print()
    print("=" * 80)
    print("Feature counts by source")
    print("=" * 80)

    for source in SOURCE_ORDER:
        print(f"{source}: {len(sensor_columns[source])} features")

    selected_df = load_selected_features(SELECTED_FEATURES_PATH)

    print()
    print("=" * 80)
    print("Building models")
    print("=" * 80)

    models = build_models()

    all_result_rows = []
    all_prediction_tables = []

    prediction_store = {}

    for source_name in SOURCE_ORDER:
        print()
        print("=" * 80)
        print(f"Evaluating sensor combination: {source_name}")
        print("=" * 80)

        all_features_for_source = sensor_columns[source_name]

        if len(all_features_for_source) == 0:
            print(f"[SKIP] No features found for {source_name}")
            continue

        feature_sets = build_feature_sets_for_source(
            source_name=source_name,
            all_features=all_features_for_source,
            selected_df=selected_df,
        )

        for feature_set_name, feature_columns in feature_sets.items():
            print()
            print("-" * 80)
            print(f"Feature set: {feature_set_name}")
            print("-" * 80)

            X, y = clean_feature_matrix(df, feature_columns)

            if X.shape[1] == 0:
                print(f"[SKIP] No usable features for {source_name} | {feature_set_name}")
                continue

            print(f"Usable samples: {X.shape[0]}")
            print(f"Usable features: {X.shape[1]}")

            for model_name, model in models.items():
                print(f"  - Model: {model_name}")

                try:
                    metrics, predictions_df, y_pred, validation_method = evaluate_model_with_loo(
                        model=model,
                        X=X,
                        y=y,
                        groups=groups,
                    )

                    result_row = {
                        "sensor_combination": source_name,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "validation_method": validation_method,
                        "n_samples": X.shape[0],
                        "n_features": X.shape[1],
                        **metrics,
                    }

                    all_result_rows.append(result_row)

                    predictions_df.insert(0, "sensor_combination", source_name)
                    predictions_df.insert(1, "feature_set", feature_set_name)
                    predictions_df.insert(2, "model", model_name)
                    predictions_df.insert(3, "validation_method", validation_method)

                    all_prediction_tables.append(predictions_df)

                    prediction_store[(source_name, feature_set_name, model_name)] = {
                        "y_true": y.values,
                        "y_pred": y_pred,
                    }

                    save_classification_report(
                        y_true=y.values, # type: ignore
                        y_pred=y_pred,
                        source_name=source_name,
                        feature_set_name=feature_set_name,
                        model_name=model_name,
                    )

                    print(
                        f"    accuracy={metrics['accuracy']:.3f}, "
                        f"macro_f1={metrics['macro_f1']:.3f}, "
                        f"balanced_accuracy={metrics['balanced_accuracy']:.3f}"
                    )

                except Exception as exc:
                    print(
                        f"    [ERROR] {source_name} | {feature_set_name} | "
                        f"{model_name}: {exc}"
                    )

    results_df = pd.DataFrame(all_result_rows)

    if results_df.empty:
        raise RuntimeError("No model results were generated. Check feature columns and labels.")

    predictions_all_df = pd.concat(all_prediction_tables, ignore_index=True)

    results_df = results_df.sort_values(
        by=["macro_f1", "accuracy", "balanced_accuracy"],
        ascending=False,
    )

    final_results_path = OUTPUT_DIR / "final_model_comparison.csv"
    results_df.to_csv(final_results_path, index=False)

    predictions_path = OUTPUT_DIR / "loo_predictions.csv"
    predictions_all_df.to_csv(predictions_path, index=False)

    print()
    print("=" * 80)
    print("Selecting best model per sensor combination and feature set")
    print("=" * 80)

    best_rows = []

    grouped = results_df.groupby(["sensor_combination", "feature_set"], sort=False)

    for (source_name, feature_set_name), subset in grouped:
        best_row = subset.sort_values(
            by=["macro_f1", "accuracy", "balanced_accuracy"],
            ascending=False,
        ).iloc[0]

        best_rows.append(best_row)

        key = (
            best_row["sensor_combination"],
            best_row["feature_set"],
            best_row["model"],
        )

        y_true = prediction_store[key]["y_true"]
        y_pred = prediction_store[key]["y_pred"]

        save_confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            source_name=best_row["sensor_combination"],
            feature_set_name=best_row["feature_set"],
            model_name=best_row["model"],
        )

        print(
            f"{best_row['sensor_combination']} | {best_row['feature_set']}: "
            f"{best_row['model']} | "
            f"accuracy={best_row['accuracy']:.3f}, "
            f"macro_f1={best_row['macro_f1']:.3f}"
        )

    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values(
        by=["macro_f1", "accuracy", "balanced_accuracy"],
        ascending=False,
    )

    best_path = OUTPUT_DIR / "best_model_per_sensor_combination_and_feature_set.csv"
    best_df.to_csv(best_path, index=False)

    print()
    print("=" * 80)
    print("Creating summary plots")
    print("=" * 80)

    for feature_set_name in FEATURE_SET_ORDER:
        if feature_set_name not in results_df["feature_set"].unique():
            continue

        safe_feature_set = safe_filename(feature_set_name)

        save_metric_heatmap(
            results_df=results_df,
            metric="accuracy",
            feature_set_name=feature_set_name,
            filename=f"heatmap_accuracy_{safe_feature_set}.png",
            title=f"Accuracy by model and sensor combination — {feature_set_name}",
        )

        save_metric_heatmap(
            results_df=results_df,
            metric="macro_f1",
            feature_set_name=feature_set_name,
            filename=f"heatmap_macro_f1_{safe_feature_set}.png",
            title=f"Macro F1-score by model and sensor combination — {feature_set_name}",
        )

        save_metric_heatmap(
            results_df=results_df,
            metric="balanced_accuracy",
            feature_set_name=feature_set_name,
            filename=f"heatmap_balanced_accuracy_{safe_feature_set}.png",
            title=f"Balanced accuracy by model and sensor combination — {feature_set_name}",
        )

    save_best_model_barplot(
        best_df=best_df,
        metric="macro_f1",
    )

    save_model_ranking_barplot(
        results_df=results_df,
        metric="macro_f1",
    )

    save_feature_set_comparison_barplot(
        results_df=results_df,
        metric="macro_f1",
    )

    save_sensor_feature_set_matrix_plot(
        best_df=best_df,
        metric="macro_f1",
    )

    print()
    print("=" * 80)
    print("Machine learning comparison completed")
    print("=" * 80)
    print(f"[SAVED] Final results: {final_results_path}")
    print(f"[SAVED] Best models: {best_path}")
    print(f"[SAVED] Leave-one-out predictions: {predictions_path}")
    print(f"[SAVED] Plots: {OUTPUT_DIR / 'plots'}")
    print(f"[SAVED] Confusion matrices: {OUTPUT_DIR / 'confusion_matrices'}")
    print(f"[SAVED] Classification reports: {OUTPUT_DIR / 'classification_reports'}")
    print(f"[SAVED] Selected feature lists: {OUTPUT_DIR / 'selected_feature_lists'}")


if __name__ == "__main__":
    main()