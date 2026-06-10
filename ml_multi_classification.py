"""
Machine learning comparison script for 3-class gait classification.

Classes:
    0 = Healthy leg
    1 = Affected side
    2 = Non-affected side

Input:
    data.csv

Outputs:
    outputs/ml_results/
        final_model_comparison.csv
        best_model_per_sensor_combination.csv
        per_fold_results.csv
        classification_reports/
        confusion_matrices/
        plots/

Sensor combinations:
    1. Only gyroscope
    2. Only accelerometer
    3. Only EMG
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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
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
OUTPUT_DIR = Path("outputs/ml_results")

ID_COLUMN = "ID"
LABEL_COLUMN = "Label"

RANDOM_STATE = 42
N_SPLITS = 5

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

    # sep=None makes the script robust to comma, semicolon, or tab-separated files.
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
            f"Subject-aware validation will not be possible."
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


def infer_subject_group_from_id(raw_id: object) -> str:
    """
    Attempts to infer subject-level grouping from sample ID.

    Examples that this handles reasonably:
        H01_left       -> H01
        H01_right      -> H01
        HT01_L         -> HT01
        PT03_affected  -> PT03
        P12_nonaffected -> P12
        Stroke_05_A    -> Stroke_05

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
    # This may not be enough for subject grouping if rows are simple 1..60 sample IDs.
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

    # If every row has a unique group, subject grouping is probably not useful.
    if n_groups == n_samples:
        print()
        print("[WARNING]")
        print(
            "Every row appears to have a unique group. "
            "This usually means the ID column is a sample ID, not a subject ID."
        )
        print(
            "The script will fall back to StratifiedKFold unless you modify "
            "infer_subject_group_from_id()."
        )
        return None

    return groups # type: ignore


def clean_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    X = df[feature_columns].copy()
    y = df[LABEL_COLUMN].copy()

    # Replace infinite values.
    X = X.replace([np.inf, -np.inf], np.nan)

    # Drop features that are entirely missing.
    X = X.dropna(axis=1, how="all")

    # Drop constant columns.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    return X, y


# =============================================================================
# Model definitions
# =============================================================================

def build_models() -> Dict[str, Pipeline]:
    """
    Common ML models for tabular classification.

    Scaling is included for all models for simplicity and consistency.
    Tree-based models do not require scaling, but scaling does not harm their logic.
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

    # Optional XGBoost support.
    # If xgboost is not installed, the script continues normally.
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
# Cross-validation
# =============================================================================

def make_cv_splitter(
    y: pd.Series,
    groups: Optional[np.ndarray],
):
    if groups is not None:
        print("[INFO] Using StratifiedGroupKFold.")
        return StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    print("[INFO] Using StratifiedKFold.")
    return StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def get_cv_splits(
    splitter,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
):
    if groups is not None:
        return splitter.split(X, y, groups=groups)

    return splitter.split(X, y)


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


def evaluate_model_with_cv(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[np.ndarray],
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    splitter = make_cv_splitter(y, groups)

    all_true = []
    all_pred = []
    per_fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(
        get_cv_splits(splitter, X, y, groups),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_pred = fold_model.predict(X_test)

        fold_metrics = evaluate_predictions(y_test.values, y_pred)

        fold_row = {
            "fold": fold_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            **fold_metrics,
        }
        per_fold_rows.append(fold_row)

        all_true.extend(y_test.values)
        all_pred.extend(y_pred)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    overall_metrics = evaluate_predictions(all_true, all_pred)
    per_fold_df = pd.DataFrame(per_fold_rows)

    return overall_metrics, per_fold_df, all_pred


# =============================================================================
# Plot functions
# =============================================================================

def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_name: str,
    model_name: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    image = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"Confusion matrix\n{source_name} — {model_name}")
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
        f"{safe_filename(model_name)}.png"
    )
    output_path = OUTPUT_DIR / "confusion_matrices" / filename

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_metric_heatmap(
    results_df: pd.DataFrame,
    metric: str,
    filename: str,
    title: str,
) -> None:
    pivot = results_df.pivot(
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
    plot_df = plot_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(
        plot_df["sensor_combination"],
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
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Best model per sensor combination")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Sensor combination")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    output_path = OUTPUT_DIR / "plots" / f"best_model_per_combination_{metric}.png"

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

    ax.set_title("Average model performance across all sensor combinations")
    ax.set_xlabel(f"Mean {metric.replace('_', ' ').title()}")
    ax.set_ylabel("Model")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    output_path = OUTPUT_DIR / "plots" / f"average_model_ranking_{metric}.png"

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_sensor_ranking_barplot(
    results_df: pd.DataFrame,
    metric: str = "macro_f1",
) -> None:
    sensor_ranking = (
        results_df.groupby("sensor_combination")[metric]
        .mean()
        .reindex(SOURCE_ORDER)
        .sort_values(ascending=True)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(
        sensor_ranking["sensor_combination"],
        sensor_ranking[metric],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, value in zip(bars, sensor_ranking[metric]):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Average sensor-combination performance across all models")
    ax.set_xlabel(f"Mean {metric.replace('_', ' ').title()}")
    ax.set_ylabel("Sensor combination")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    output_path = OUTPUT_DIR / "plots" / f"average_sensor_ranking_{metric}.png"

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

    print()
    print("=" * 80)
    print("Building models")
    print("=" * 80)

    models = build_models()

    all_result_rows = []
    all_fold_rows = []

    # Store predictions so we can create confusion matrices for the best models.
    prediction_store = {}

    for source_name in SOURCE_ORDER:
        print()
        print("=" * 80)
        print(f"Evaluating sensor combination: {source_name}")
        print("=" * 80)

        feature_columns = sensor_columns[source_name]

        if len(feature_columns) == 0:
            print(f"[SKIP] No features found for {source_name}")
            continue

        X, y = clean_feature_matrix(df, feature_columns)

        print(f"Usable samples: {X.shape[0]}")
        print(f"Usable features: {X.shape[1]}")

        for model_name, model in models.items():
            print(f"  - Model: {model_name}")

            try:
                metrics, per_fold_df, y_pred = evaluate_model_with_cv(
                    model=model,
                    X=X,
                    y=y,
                    groups=groups,
                )

                result_row = {
                    "sensor_combination": source_name,
                    "model": model_name,
                    "n_samples": X.shape[0],
                    "n_features": X.shape[1],
                    **metrics,
                }

                all_result_rows.append(result_row)

                per_fold_df.insert(0, "sensor_combination", source_name)
                per_fold_df.insert(1, "model", model_name)
                all_fold_rows.append(per_fold_df)

                prediction_store[(source_name, model_name)] = {
                    "y_true": y.values,
                    "y_pred": y_pred,
                }

                save_classification_report(
                    y_true=y.values, # type: ignore
                    y_pred=y_pred,
                    source_name=source_name,
                    model_name=model_name,
                )

                print(
                    f"    accuracy={metrics['accuracy']:.3f}, "
                    f"macro_f1={metrics['macro_f1']:.3f}, "
                    f"balanced_accuracy={metrics['balanced_accuracy']:.3f}"
                )

            except Exception as exc:
                print(f"    [ERROR] {source_name} | {model_name}: {exc}")

    results_df = pd.DataFrame(all_result_rows)

    if results_df.empty:
        raise RuntimeError("No model results were generated. Check feature columns and labels.")

    fold_results_df = pd.concat(all_fold_rows, ignore_index=True)

    # Sort final table by macro F1 and accuracy.
    results_df = results_df.sort_values(
        by=["macro_f1", "accuracy", "balanced_accuracy"],
        ascending=False,
    )

    final_results_path = OUTPUT_DIR / "final_model_comparison.csv"
    results_df.to_csv(final_results_path, index=False)

    per_fold_path = OUTPUT_DIR / "per_fold_results.csv"
    fold_results_df.to_csv(per_fold_path, index=False)

    print()
    print("=" * 80)
    print("Selecting best model per sensor combination")
    print("=" * 80)

    best_rows = []

    for source_name in SOURCE_ORDER:
        subset = results_df[results_df["sensor_combination"] == source_name]

        if subset.empty:
            continue

        best_row = subset.sort_values(
            by=["macro_f1", "accuracy", "balanced_accuracy"],
            ascending=False,
        ).iloc[0]

        best_rows.append(best_row)

        key = (best_row["sensor_combination"], best_row["model"])
        y_true = prediction_store[key]["y_true"]
        y_pred = prediction_store[key]["y_pred"]

        save_confusion_matrix_plot(
            y_true=y_true,
            y_pred=y_pred,
            source_name=best_row["sensor_combination"],
            model_name=best_row["model"],
        )

        print(
            f"{best_row['sensor_combination']}: "
            f"{best_row['model']} | "
            f"accuracy={best_row['accuracy']:.3f}, "
            f"macro_f1={best_row['macro_f1']:.3f}"
        )

    best_df = pd.DataFrame(best_rows)
    best_df = best_df.sort_values(
        by=["macro_f1", "accuracy", "balanced_accuracy"],
        ascending=False,
    )

    best_path = OUTPUT_DIR / "best_model_per_sensor_combination.csv"
    best_df.to_csv(best_path, index=False)

    print()
    print("=" * 80)
    print("Creating summary plots")
    print("=" * 80)

    save_metric_heatmap(
        results_df=results_df,
        metric="accuracy",
        filename="heatmap_accuracy_by_model_and_sensor.png",
        title="Accuracy by model and sensor combination",
    )

    save_metric_heatmap(
        results_df=results_df,
        metric="macro_f1",
        filename="heatmap_macro_f1_by_model_and_sensor.png",
        title="Macro F1-score by model and sensor combination",
    )

    save_metric_heatmap(
        results_df=results_df,
        metric="balanced_accuracy",
        filename="heatmap_balanced_accuracy_by_model_and_sensor.png",
        title="Balanced accuracy by model and sensor combination",
    )

    save_best_model_barplot(
        best_df=best_df,
        metric="macro_f1",
    )

    save_model_ranking_barplot(
        results_df=results_df,
        metric="macro_f1",
    )

    save_sensor_ranking_barplot(
        results_df=results_df,
        metric="macro_f1",
    )

    print()
    print("=" * 80)
    print("Machine learning comparison completed")
    print("=" * 80)
    print(f"[SAVED] Final results: {final_results_path}")
    print(f"[SAVED] Best models: {best_path}")
    print(f"[SAVED] Per-fold results: {per_fold_path}")
    print(f"[SAVED] Plots: {OUTPUT_DIR / 'plots'}")
    print(f"[SAVED] Confusion matrices: {OUTPUT_DIR / 'confusion_matrices'}")
    print(f"[SAVED] Classification reports: {OUTPUT_DIR / 'classification_reports'}")


if __name__ == "__main__":
    main()