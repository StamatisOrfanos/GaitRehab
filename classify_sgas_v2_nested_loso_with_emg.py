"""
Nested subject-level classification of frozen SGAS-v2 severity labels.
This script implements the classification design for:
    0. Low/reference asymmetry (healthy)
    1. Moderate sensor-derived asymmetry (stroke)
    2. High sensor-derived asymmetry (stroke)

The labels are read from the frozen SGAS-v2 output and are never recomputed or
changed here. Cycle-aligned IMU predictors are extracted directly from the raw
participant folders. Previously extracted EMG features are read from the
verified participant-level mapping created by
create_sensor_derived_severity_dataset_with_emg.py.

Leakage barrier
---------------
The SGAS-v2 labels were constructed from bilateral shank gyroscope-z gait-cycle
measurements. Therefore, the primary classifier deliberately excludes:

* every gyroscope-z sample and feature;
* all four SGAS-v2 components and their transformed contributions;
* the continuous SGAS-v2 score;
* stroke ranks, class-confidence values, borderline flags and thresholds;
* participant cohort as a predictor.

Eligible predictors are derived from:

* bilateral shank accelerometer x/y/z signals; and
* bilateral shank gyroscope x/y signals; and
* six-muscle EMG feature summaries expressed as order-invariant bilateral
  means and absolute bilateral differences.

The frozen gyroscope-z gait-cycle start/end timestamps are used only to align
eligible signals. Predictor construction does not use gyroscope-z values,
stride duration, cycle amplitude, cycle count or polarity. Per-cycle magnitude
features are aggregated into side-invariant bilateral means and differences.

Validation
----------
* Outer leave-one-subject-out (LOSO): unbiased final participant predictions.
* Inner stratified grouped cross-validation: sensor-set, model,
  hyperparameter and feature-count selection using outer-training subjects only.
* Imputation, scaling and feature selection are all inside the pipeline.
* Final metrics are calculated from pooled outer predictions, not averaged from
  one-participant LOSO folds.

Analyses
--------
1. Primary direct three-class model without waveform-derived predictors.
2. Outer-LOSO comparison of eleven common classifier families, each with
   independently nested sensor-set, feature-count and hyperparameter selection.
3. Sensitivity analysis excluding the two predeclared borderline participants.
4. Sensitivity analysis adding eligible cycle-normalized waveform predictors.
5. Exploratory hierarchical model: Healthy versus Stroke, followed by
   Moderate versus High within Stroke.
6. Continuous SGAS-v2 score regression with nested LOSO, MAE and Spearman
   correlation.
7. Matched-participant IMU-only baseline for fair modality comparison.
8. EMG-only classification on the same matched participants.
9. Cycle-aligned IMU plus EMG classification on the matched participants.
10. Optional sensitivity analysis additionally incorporating eligible legacy
    non-gyroscope-z IMU summaries from the mapped feature table.

No command-line arguments are used. Edit the constants in the configuration
section before running:

    python classify_sgas_v2_nested_loso_with_emg.py

Expected directory structure
----------------------------
    Data/
        Healthy/Patient_1/*.csv
        Stroke/Patient_1/*.csv

    third_article/outputs/sensor_derived_severity_with_emg/
        sensor_derived_gait_asymmetry_severity.csv
        frozen_score_definition.json
        gait_cycle_metrics.csv
        participant_level_mapped_external_features.csv

Main output directory
---------------------
    third_article/outputs/sgas_v2_nested_classification_with_emg/
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import dct
from scipy.signal import butter, sosfiltfilt
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import LinearSVR, SVC
from sklearn.tree import DecisionTreeClassifier


# =============================================================================
# Configuration: edit these constants before running
# =============================================================================

DATA_DIR = Path("Data")
SEVERITY_OUTPUT_DIR = Path(
    "third_article/outputs/sensor_derived_severity_with_emg"
)
LABELS_PATH = Path(
    SEVERITY_OUTPUT_DIR / "sensor_derived_gait_asymmetry_severity.csv"
)
SCORE_DEFINITION_PATH = Path(
    SEVERITY_OUTPUT_DIR / "frozen_score_definition.json"
)
CYCLE_METRICS_PATH = Path(
    SEVERITY_OUTPUT_DIR / "gait_cycle_metrics.csv"
)
MAPPED_EXTERNAL_FEATURES_PATH = Path(
    SEVERITY_OUTPUT_DIR / "participant_level_mapped_external_features.csv"
)
OUTPUT_DIR = Path(
    "third_article/outputs/sgas_v2_nested_classification_with_emg"
)

# Canonical hash of the exact participant labels approved before modelling.
# This protects the class assignments independently of CSV formatting or row order.
EXPECTED_FROZEN_LABEL_ASSIGNMENT_SHA256 = (
    "ff581408ab73ece6714b4fca2e57d52460774f03acf22327a76eae2445b1318a"
)

# The output directory must be empty unless this is deliberately enabled.
OVERWRITE = False

# Each analysis can be enabled or disabled independently.
RUN_PRIMARY_ANALYSIS = True
RUN_BORDERLINE_EXCLUSION_SENSITIVITY = True
RUN_WAVEFORM_SENSITIVITY = True
RUN_HIERARCHICAL_SENSITIVITY = True
RUN_CONTINUOUS_SCORE_REGRESSION = True
RUN_MATCHED_IMU_BASELINE = True
RUN_EMG_ONLY_ANALYSIS = True
RUN_IMU_EMG_ANALYSIS = True
# This optional analysis adds whole-recording legacy IMU summaries. It is off
# by default because the primary multimodal comparison should use the stronger
# cycle-aligned raw-IMU predictors plus EMG.
RUN_LEGACY_IMU_FEATURE_SENSITIVITY = False

EXPECTED_MAPPED_FEATURE_PARTICIPANTS = 30
EXPECTED_EMG_COMPLETE_CASE_PARTICIPANTS = 28

CLASS_IDS = (0, 1, 2)
CLASS_SHORT_NAMES = {
    0: "Low/reference",
    1: "Moderate",
    2: "High",
}

RAW_IMU_SENSOR_SETS = (
    "accelerometer",
    "gyroscope_xy",
    "combined",
)
EMG_ONLY_SENSOR_SETS = ("emg",)
IMU_EMG_SENSOR_SETS = ("imu_emg",)
IMU_EMG_LEGACY_SENSOR_SETS = ("imu_emg_legacy",)

# Broad conventional model benchmark. Every family is evaluated with held-out
# outer-LOSO predictions; inner CV is still responsible for selecting its
# sensor set, feature count and compact hyperparameter configuration.
CLASSIFICATION_MODEL_NAMES = (
    "Logistic Regression",
    "Linear SVM",
    "RBF SVM",
    "k-Nearest Neighbours",
    "Gaussian Naive Bayes",
    "Shrinkage LDA",
    "Decision Tree",
    "Random Forest",
    "Extra Trees",
    "Gradient Boosting",
    "AdaBoost",
)

# Journal-ready plot palette. Blue tones distinguish primary data series;
# neutral grey is reserved for secondary series, reference lines and errors.
BLUE_DARK = "#1F4E79"
BLUE_MEDIUM = "#4F81BD"
BLUE_LIGHT = "#9DC3E6"
GREY_DARK = "#595959"
GREY_MEDIUM = "#A6A6A6"
CLASS_PLOT_COLORS = {
    0: BLUE_LIGHT,
    1: BLUE_MEDIUM,
    2: BLUE_DARK,
}

REQUIRED_RAW_FILES = (
    "LeftShank-Accelerometer.csv",
    "LeftShank-Gyroscope.csv",
    "RightShank-Accelerometer.csv",
    "RightShank-Gyroscope.csv",
)

LABEL_ONLY_COLUMNS = {
    "temporal_asymmetry",
    "amplitude_asymmetry",
    "variability_asymmetry",
    "waveform_dissimilarity",
    "temporal_asymmetry_healthy_robust_z",
    "temporal_asymmetry_severity_contribution",
    "amplitude_asymmetry_healthy_robust_z",
    "amplitude_asymmetry_severity_contribution",
    "variability_asymmetry_healthy_robust_z",
    "variability_asymmetry_severity_contribution",
    "waveform_dissimilarity_healthy_robust_z",
    "waveform_dissimilarity_severity_contribution",
    "sensor_derived_severity_score",
    "healthy_leave_one_out_reference_score",
    "severity_class_id",
    "severity_class",
    "above_healthy_reference",
    "stroke_severity_rank_percentile",
    "moderate_high_assignment_confidence",
    "stroke_split_borderline",
    "bootstrap_probability_high",
    "score_definition_version",
    "score_definition_sha256",
}


@dataclass(frozen=True)
class SignalConfig:
    sampling_rate_hz: float = 100.0
    maximum_interpolation_gap_seconds: float = 0.05
    minimum_common_overlap_seconds: float = 8.0
    lowpass_cutoff_hz: float = 20.0
    lowpass_order: int = 2
    minimum_filter_segment_seconds: float = 4.0
    minimum_cycle_samples: int = 20
    cycle_waveform_points: int = 101


@dataclass(frozen=True)
class ValidationConfig:
    inner_splits: int = 5
    random_state: int = 20261004
    bootstrap_iterations: int = 2000
    n_jobs: int = -1
    primary_metric: str = "macro_f1"
    feature_counts: Tuple[Any, ...] = (5, 10)


SIGNAL_CONFIG = SignalConfig()
VALIDATION_CONFIG = ValidationConfig()


# =============================================================================
# General utilities
# =============================================================================


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def natural_sort_key(value: str) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Set OVERWRITE = True only when replacement is intended."
        )
    path.mkdir(parents=True, exist_ok=True)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def as_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower() # type: ignore
    return normalized.isin({"1", "true", "yes", "y"})


def symmetric_difference(left: float, right: float, epsilon: float = 1e-12) -> float:
    denominator = abs(left) + abs(right)
    if not np.isfinite(denominator) or denominator <= epsilon:
        return np.nan
    return float(abs(left - right) / denominator)


# =============================================================================
# Frozen label loading and validation
# =============================================================================


def validate_score_definition(path: Path) -> Dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen score definition not found: {path}")

    definition = json.loads(path.read_text(encoding="utf-8"))
    required = {"score_definition_version", "class_rule", "sha256"}
    missing = required.difference(definition)
    if missing:
        raise ValueError(
            "Frozen score definition is missing: " + ", ".join(sorted(missing))
        )

    expected_hash = str(definition["sha256"])
    unhashed = dict(definition)
    unhashed.pop("sha256", None)
    canonical = json.dumps(
        unhashed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    actual_hash = hashlib.sha256(canonical).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            "The frozen score-definition hash does not match its contents. "
            "Do not continue until the correct frozen file is restored."
        )
    return definition


def load_frozen_labels(
    labels_path: Path,
    score_definition: Mapping[str, object],
) -> pd.DataFrame:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Frozen severity dataset not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    required = {
        "participant_key",
        "cohort",
        "patient_folder",
        "severity_class_id",
        "severity_class",
        "sensor_derived_severity_score",
        "stroke_split_borderline",
        "score_definition_version",
        "score_definition_sha256",
    }
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(
            "Frozen severity dataset is missing: " + ", ".join(sorted(missing))
        )
    if labels["participant_key"].duplicated().any():
        duplicates = labels.loc[
            labels["participant_key"].duplicated(False), "participant_key"
        ].tolist()
        raise ValueError(f"Duplicate frozen participant keys: {duplicates}")

    labels["severity_class_id"] = pd.to_numeric(
        labels["severity_class_id"], errors="raise"
    ).astype(int)
    observed_classes = set(labels["severity_class_id"].unique())
    if observed_classes != set(CLASS_IDS):
        raise ValueError(
            f"Expected frozen classes {set(CLASS_IDS)}, found {observed_classes}."
        )

    expected_version = str(score_definition["score_definition_version"])
    expected_hash = str(score_definition["sha256"])
    if not labels["score_definition_version"].eq(expected_version).all():
        raise ValueError("Severity rows do not all use the frozen score version.")
    if not labels["score_definition_sha256"].eq(expected_hash).all():
        raise ValueError("Severity rows do not all match the frozen score hash.")

    labels["stroke_split_borderline"] = as_boolean(
        labels["stroke_split_borderline"]
    )
    assignment_hash = frozen_label_assignment_hash(labels)
    if assignment_hash != EXPECTED_FROZEN_LABEL_ASSIGNMENT_SHA256:
        raise ValueError(
            "The participant labels no longer match the labels frozen before "
            "model testing. Expected assignment hash "
            f"{EXPECTED_FROZEN_LABEL_ASSIGNMENT_SHA256}, found {assignment_hash}."
        )
    labels = labels.sort_values(
        ["severity_class_id", "participant_key"],
        key=lambda column: (
            column.map(natural_sort_key)
            if column.name == "participant_key"
            else column
        ),
    ).reset_index(drop=True)
    return labels


def frozen_label_assignment_hash(labels: pd.DataFrame) -> str:
    columns = [
        "participant_key",
        "cohort",
        "patient_folder",
        "severity_class_id",
        "severity_class",
        "sensor_derived_severity_score",
        "stroke_split_borderline",
        "score_definition_version",
        "score_definition_sha256",
    ]
    canonical_rows = (
        labels[columns]
        .sort_values("participant_key")
        .to_dict(orient="records")
    )
    canonical = json.dumps(
        canonical_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def save_frozen_label_snapshot(
    labels: pd.DataFrame,
    score_definition: Mapping[str, object],
    output_dir: Path,
) -> str:
    columns = [
        "participant_key",
        "cohort",
        "patient_folder",
        "severity_class_id",
        "severity_class",
        "sensor_derived_severity_score",
        "stroke_split_borderline",
        "moderate_high_assignment_confidence",
        "bootstrap_probability_high",
        "score_definition_version",
        "score_definition_sha256",
    ]
    columns = [column for column in columns if column in labels.columns]
    snapshot = labels[columns].copy()
    snapshot_path = output_dir / "frozen_labels_used.csv"
    snapshot.to_csv(snapshot_path, index=False)
    snapshot_hash = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    manifest = {
        "frozen_label_file": str(labels_path_for_manifest()),
        "frozen_label_snapshot": str(snapshot_path),
        "frozen_label_snapshot_sha256": snapshot_hash,
        "score_definition_version": score_definition["score_definition_version"],
        "score_definition_sha256": score_definition["sha256"],
        "frozen_label_assignment_sha256": frozen_label_assignment_hash(labels),
        "participant_count": int(len(snapshot)),
        "class_counts": {
            str(key): int(value)
            for key, value in snapshot["severity_class_id"].value_counts().sort_index().items()
        },
        "labels_are_read_only": True,
    }
    (output_dir / "frozen_label_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default),
        encoding="utf-8",
    )
    return snapshot_hash


def labels_path_for_manifest() -> Path:
    return LABELS_PATH


def load_mapped_external_features(
    path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load classifier-safe mapped EMG and optional legacy IMU predictors.

    This file is created by the EMG-enabled dataset script. It must contain
    only the participant key and explicitly prefixed predictor columns. Source
    ID and source Label fields are rejected because they are audit metadata,
    not model inputs.
    """
    if not path.is_file():
        raise FileNotFoundError(
            "Mapped external predictor dataset not found: "
            f"{path}. Run create_sensor_derived_severity_dataset_with_emg.py "
            "before this classifier."
        )
    features = pd.read_csv(path)
    if "participant_key" not in features.columns:
        raise ValueError(
            "Mapped external predictor dataset has no participant_key column."
        )
    if features["participant_key"].duplicated().any():
        duplicates = features.loc[
            features["participant_key"].duplicated(False),
            "participant_key",
        ].tolist()
        raise ValueError(f"Duplicate mapped participant keys: {duplicates}")
    if len(features) != EXPECTED_MAPPED_FEATURE_PARTICIPANTS:
        raise ValueError(
            "Mapped external predictor dataset must contain exactly "
            f"{EXPECTED_MAPPED_FEATURE_PARTICIPANTS} participants; found "
            f"{len(features)}."
        )

    predictor_columns = [
        column for column in features.columns if column != "participant_key"
    ]
    if not predictor_columns:
        raise ValueError("Mapped external predictor dataset has no predictors.")
    forbidden_metadata = [
        column
        for column in predictor_columns
        if column.lower().startswith(
            ("id", "label", "feature_source", "audit_only")
        )
    ]
    if forbidden_metadata:
        raise ValueError(
            "Audit metadata entered the mapped predictor table: "
            + ", ".join(forbidden_metadata)
        )
    unexpected = [
        column
        for column in predictor_columns
        if not column.startswith(("emg__", "legacy_imu__"))
    ]
    if unexpected:
        raise ValueError(
            "Mapped predictor columns have unexpected prefixes: "
            + ", ".join(unexpected)
        )

    numeric = features[predictor_columns].apply(pd.to_numeric, errors="coerce")
    nonfinite = ~np.isfinite(numeric.to_numpy(dtype=float))
    if nonfinite.any():
        rows, columns = np.where(nonfinite)
        examples = [
            f"participant {features.iloc[int(row)]['participant_key']}, "
            f"column {predictor_columns[int(column)]}"
            for row, column in zip(rows[:10], columns[:10])
        ]
        raise ValueError(
            "Mapped predictors contain missing or non-finite values: "
            + "; ".join(examples)
        )
    features.loc[:, predictor_columns] = numeric
    assert_eligible_feature_names(predictor_columns)

    manifest_rows: List[Dict[str, object]] = []
    for column in predictor_columns:
        if column.startswith("emg__"):
            modality = "emg"
            primary_eligible = True
        else:
            modality = "legacy_imu"
            primary_eligible = False
        manifest_rows.append(
            {
                "feature_name": column,
                "sensor_set": modality,
                "waveform_derived": False,
                "uses_gyroscope_z": False,
                "cycle_aligned": False,
                "side_invariant": True,
                "eligible_for_primary_classifier": primary_eligible,
                "feature_source": "mapped_precomputed_feature_table",
            }
        )
    return features, pd.DataFrame(manifest_rows)


def create_matched_emg_cohort(
    labels: pd.DataFrame,
    external_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create and audit the label/EMG participant intersection."""
    label_keys = set(labels["participant_key"].astype(str))
    feature_keys = set(external_features["participant_key"].astype(str))
    matched_keys = label_keys.intersection(feature_keys)
    if len(matched_keys) != EXPECTED_EMG_COMPLETE_CASE_PARTICIPANTS:
        raise ValueError(
            "The frozen-label/EMG intersection changed. Expected "
            f"{EXPECTED_EMG_COMPLETE_CASE_PARTICIPANTS} participants, found "
            f"{len(matched_keys)}. Review the mapping QC output before modelling."
        )

    matched_labels = labels.loc[
        labels["participant_key"].isin(matched_keys)
    ].copy()
    matched_labels = matched_labels.sort_values(
        ["severity_class_id", "participant_key"],
        key=lambda column: (
            column.map(natural_sort_key)
            if column.name == "participant_key"
            else column
        ),
    ).reset_index(drop=True)
    if set(matched_labels["severity_class_id"].unique()) != set(CLASS_IDS):
        raise ValueError(
            "The matched EMG cohort does not contain all three severity classes."
        )

    rows: List[Dict[str, object]] = []
    for key in sorted(label_keys.union(feature_keys), key=natural_sort_key):
        in_labels = key in label_keys
        in_features = key in feature_keys
        if in_labels and in_features:
            status = "matched"
            note = "Included in matched IMU/EMG comparisons."
        elif in_labels:
            status = "frozen_label_only"
            note = "Valid SGAS-v2 label but no mapped EMG feature record."
        else:
            status = "mapped_feature_only"
            note = "Mapped EMG record but no valid frozen SGAS-v2 label."
        rows.append(
            {
                "participant_key": key,
                "present_in_frozen_labels": int(in_labels),
                "present_in_mapped_features": int(in_features),
                "mapping_status": status,
                "included_in_matched_comparison": int(
                    in_labels and in_features
                ),
                "notes": note,
            }
        )
    return matched_labels, pd.DataFrame(rows)


def subset_predictors_to_labels(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    keys = set(labels["participant_key"].astype(str))
    subset = features.loc[features["participant_key"].isin(keys)].copy()
    if set(subset["participant_key"].astype(str)) != keys:
        missing = sorted(
            keys.difference(set(subset["participant_key"].astype(str))),
            key=natural_sort_key,
        )
        raise ValueError(f"Missing predictor rows for matched participants: {missing}")
    return subset.reset_index(drop=True)


def select_external_modality(
    external_features: pd.DataFrame,
    include_legacy_imu: bool,
) -> pd.DataFrame:
    prefixes = (
        ("emg__", "legacy_imu__")
        if include_legacy_imu
        else ("emg__",)
    )
    columns = [
        column
        for column in external_features.columns
        if column == "participant_key" or column.startswith(prefixes)
    ]
    selected = external_features[columns].copy()
    if len(columns) <= 1:
        raise ValueError("No mapped predictors were selected for this modality.")
    return selected


def combine_predictor_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
) -> pd.DataFrame:
    """Combine two predictor tables without duplicating metadata columns."""
    metadata = [
        column
        for column in ("participant_key", "cohort", "patient_folder")
        if column in first.columns
    ]
    first_predictors = [
        column for column in first.columns if column not in set(metadata)
    ]
    second_predictors = [
        column for column in second.columns if column != "participant_key"
    ]
    overlap = set(first_predictors).intersection(second_predictors)
    if overlap:
        raise ValueError(
            "Predictor tables contain duplicate feature names: "
            + ", ".join(sorted(overlap))
        )
    combined = first[metadata + first_predictors].merge(
        second[["participant_key"] + second_predictors],
        on="participant_key",
        how="inner",
        validate="one_to_one",
    )
    if len(combined) != len(first):
        raise ValueError(
            "Combining predictor modalities unexpectedly changed the "
            "participant count."
        )
    assert_eligible_feature_names(first_predictors + second_predictors)
    return combined


def load_frozen_cycle_intervals(
    path: Path,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Load only the frozen cycle boundaries needed for signal segmentation.

    Gyroscope-z amplitudes, stride durations, cycle counts and all other
    label-generating measurements are deliberately discarded. The remaining
    timestamps are used only to slice eligible accelerometer and gyroscope-x/y
    samples into previously validated gait cycles.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Frozen gait-cycle metrics not found: {path}")
    cycle_metrics = pd.read_csv(path)
    required = {"participant_key", "side", "start_epoch_ms", "end_epoch_ms"}
    missing = required.difference(cycle_metrics.columns)
    if missing:
        raise ValueError(
            "Frozen gait-cycle metrics are missing: " + ", ".join(sorted(missing))
        )

    intervals = cycle_metrics[
        ["participant_key", "side", "start_epoch_ms", "end_epoch_ms"]
    ].copy()
    intervals["participant_key"] = intervals["participant_key"].astype(str)
    intervals["side"] = intervals["side"].astype(str).str.strip().str.lower()
    intervals["start_epoch_ms"] = pd.to_numeric(
        intervals["start_epoch_ms"], errors="raise"
    )
    intervals["end_epoch_ms"] = pd.to_numeric(
        intervals["end_epoch_ms"], errors="raise"
    )
    if not intervals["side"].isin({"left", "right"}).all():
        invalid = sorted(set(intervals.loc[~intervals["side"].isin({"left", "right"}), "side"]))
        raise ValueError(f"Unexpected cycle side labels: {invalid}")
    if (intervals["end_epoch_ms"] <= intervals["start_epoch_ms"]).any():
        raise ValueError("Frozen gait-cycle metrics contain non-positive intervals.")

    expected_keys = set(labels["participant_key"].astype(str))
    observed_keys = set(intervals["participant_key"])
    missing_keys = sorted(expected_keys - observed_keys, key=natural_sort_key)
    if missing_keys:
        raise ValueError(
            "Frozen labels have no matching cycle intervals: " + ", ".join(missing_keys)
        )
    intervals = intervals.loc[intervals["participant_key"].isin(expected_keys)].copy()

    side_counts = intervals.groupby(["participant_key", "side"]).size().unstack(fill_value=0)
    for side in ("left", "right"):
        if side not in side_counts.columns or (side_counts[side] == 0).any():
            failed = side_counts.index[
                side_counts.get(side, pd.Series(0, index=side_counts.index)) == 0
            ].tolist()
            raise ValueError(f"Participants without frozen {side} cycles: {failed}")

    return intervals.sort_values(
        ["participant_key", "side", "start_epoch_ms", "end_epoch_ms"]
    ).reset_index(drop=True)


# =============================================================================
# Raw IMU loading and synchronization
# =============================================================================


def find_column(columns: Sequence[str], candidates: Iterable[str]) -> str:
    normalized = {re.sub(r"\s+", "", column.lower()): column for column in columns}
    for candidate in candidates:
        key = re.sub(r"\s+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not find any of {list(candidates)} in columns: {list(columns)}"
    )


def read_imu_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    epoch_column = find_column(frame.columns, ("epoc (ms)", "epoch (ms)", "epoch_ms", "epoc_ms")) # type: ignore
    x_column = find_column(frame.columns, ("x-axis (deg/s)", "x-axis (g)", "x")) # type: ignore
    y_column = find_column(frame.columns, ("y-axis (deg/s)", "y-axis (g)", "y")) # type: ignore
    z_column = find_column(frame.columns, ("z-axis (deg/s)", "z-axis (g)", "z")) # type: ignore

    clean = pd.DataFrame(
        {
            "epoch_ms": pd.to_numeric(frame[epoch_column], errors="coerce"),
            "x": pd.to_numeric(frame[x_column], errors="coerce"),
            "y": pd.to_numeric(frame[y_column], errors="coerce"),
            "z": pd.to_numeric(frame[z_column], errors="coerce"),
        }
    ).dropna()
    clean = clean.sort_values("epoch_ms").drop_duplicates("epoch_ms", keep="first")
    if len(clean) < 10:
        raise ValueError(f"Too few valid rows in {path}")
    return clean.reset_index(drop=True)


def interpolate_stream(
    stream: pd.DataFrame,
    grid_ms: np.ndarray,
    maximum_nearest_gap_ms: float,
) -> Tuple[np.ndarray, np.ndarray]:
    source_times = stream["epoch_ms"].to_numpy(dtype=float)
    values = stream[["x", "y", "z"]].to_numpy(dtype=float)
    interpolated = np.column_stack(
        [np.interp(grid_ms, source_times, values[:, axis]) for axis in range(3)]
    )

    right = np.searchsorted(source_times, grid_ms, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    left = np.clip(right - 1, 0, len(source_times) - 1)
    nearest_distance = np.minimum(
        np.abs(grid_ms - source_times[left]),
        np.abs(source_times[right] - grid_ms),
    )
    valid = nearest_distance <= maximum_nearest_gap_ms
    interpolated[~valid, :] = np.nan
    return interpolated, valid


def synchronise_participant(
    participant_dir: Path,
    config: SignalConfig,
) -> pd.DataFrame:
    missing = [
        filename for filename in REQUIRED_RAW_FILES if not (participant_dir / filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing files for {participant_dir}: {', '.join(missing)}"
        )

    streams = {
        "left_acc": read_imu_csv(participant_dir / "LeftShank-Accelerometer.csv"),
        "left_gyro": read_imu_csv(participant_dir / "LeftShank-Gyroscope.csv"),
        "right_acc": read_imu_csv(participant_dir / "RightShank-Accelerometer.csv"),
        "right_gyro": read_imu_csv(participant_dir / "RightShank-Gyroscope.csv"),
    }

    common_start = max(float(frame["epoch_ms"].iloc[0]) for frame in streams.values())
    common_end = min(float(frame["epoch_ms"].iloc[-1]) for frame in streams.values())
    step_ms = 1000.0 / config.sampling_rate_hz
    common_start = math.ceil(common_start / step_ms) * step_ms
    common_end = math.floor(common_end / step_ms) * step_ms
    if common_end - common_start < config.minimum_common_overlap_seconds * 1000.0:
        raise ValueError("The four sensor streams have insufficient common overlap.")

    grid_ms = np.arange(common_start, common_end + step_ms / 2.0, step_ms)
    synchronized = pd.DataFrame(
        {
            "epoch_ms": grid_ms.astype(np.int64),
            "elapsed_common_s": (grid_ms - grid_ms[0]) / 1000.0,
        }
    )
    maximum_gap_ms = config.maximum_interpolation_gap_seconds * 1000.0
    for stream_name, stream in streams.items():
        values, valid = interpolate_stream(stream, grid_ms, maximum_gap_ms)
        synchronized[f"{stream_name}_x"] = values[:, 0]
        synchronized[f"{stream_name}_y"] = values[:, 1]
        synchronized[f"{stream_name}_z"] = values[:, 2]
        synchronized[f"{stream_name}_valid"] = valid.astype(bool)
    return synchronized


def contiguous_true_ranges(mask: np.ndarray, minimum_samples: int) -> List[Tuple[int, int]]:
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(end))
        for start, end in zip(changes[0::2], changes[1::2])
        if end - start >= minimum_samples
    ]


def filter_signal_segments(
    values: np.ndarray,
    valid: np.ndarray,
    config: SignalConfig,
) -> np.ndarray:
    nyquist = config.sampling_rate_hz / 2.0
    if not 0.0 < config.lowpass_cutoff_hz < nyquist:
        raise ValueError("Low-pass cutoff must be between zero and Nyquist.")
    sos = butter(
        config.lowpass_order,
        config.lowpass_cutoff_hz / nyquist,
        btype="low",
        output="sos",
    )
    result = np.full(len(values), np.nan, dtype=float)
    minimum_samples = max(
        int(round(config.minimum_filter_segment_seconds * config.sampling_rate_hz)),
        3 * (2 * config.lowpass_order + 1),
    )
    finite_valid = valid.astype(bool) & np.isfinite(values)
    for start, end in contiguous_true_ranges(finite_valid, minimum_samples):
        result[start:end] = sosfiltfilt(sos, values[start:end])
    return result


# =============================================================================
# Cycle-aligned eligible feature extraction
# =============================================================================


def resample_cycle(values: np.ndarray, points: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < 2 or not np.isfinite(values).all():
        return np.full(points, np.nan, dtype=float)
    source = np.linspace(0.0, 1.0, values.size)
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, source, values)


def cycle_scalar_features(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 3 or not np.isfinite(values).all():
        return {
            name: np.nan
            for name in (
                "mean",
                "std",
                "median",
                "iqr",
                "range",
                "rms",
                "mean_absolute",
                "peak_absolute",
            )
        }
    q25, q75 = np.percentile(values, [25.0, 75.0])
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "median": float(np.median(values)),
        "iqr": float(q75 - q25),
        "range": float(np.ptp(values)),
        "rms": float(np.sqrt(np.mean(np.square(values)))),
        "mean_absolute": float(np.mean(np.abs(values))),
        "peak_absolute": float(np.max(np.abs(values))),
    }


def slice_signal_into_cycles(
    signal: np.ndarray,
    epoch_ms: np.ndarray,
    intervals: pd.DataFrame,
    minimum_samples: int,
) -> List[np.ndarray]:
    cycles: List[np.ndarray] = []
    for row in intervals.itertuples(index=False):
        start = int(np.searchsorted(epoch_ms, float(row.start_epoch_ms), side="left"))
        end = int(np.searchsorted(epoch_ms, float(row.end_epoch_ms), side="right"))
        segment = np.asarray(signal[start:end], dtype=float)
        if segment.size < minimum_samples or not np.isfinite(segment).all():
            continue
        cycles.append(segment)
    return cycles


def aggregate_cycle_features(cycles: Sequence[np.ndarray]) -> Dict[str, Dict[str, float]]:
    if not cycles:
        raise ValueError("No valid eligible-signal cycles remained after alignment.")
    rows = pd.DataFrame([cycle_scalar_features(cycle) for cycle in cycles])
    aggregated: Dict[str, Dict[str, float]] = {}
    for column in rows.columns:
        values = rows[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            aggregated[column] = {"median": np.nan, "iqr": np.nan}
        else:
            q25, q75 = np.percentile(finite, [25.0, 75.0])
            aggregated[column] = {
                "median": float(np.median(finite)),
                "iqr": float(q75 - q25),
            }
    return aggregated


def median_shape_template(
    cycles: Sequence[np.ndarray],
    points: int,
) -> np.ndarray:
    templates: List[np.ndarray] = []
    for cycle in cycles:
        resampled = resample_cycle(cycle, points)
        standard_deviation = float(np.std(resampled))
        if not np.isfinite(resampled).all() or standard_deviation <= 1e-12:
            continue
        templates.append((resampled - np.mean(resampled)) / standard_deviation)
    if not templates:
        return np.full(points, np.nan, dtype=float)
    return np.median(np.vstack(templates), axis=0)


def add_side_invariant_cycle_features(
    target: Dict[str, float],
    modality: str,
    left_cycles: Sequence[np.ndarray],
    right_cycles: Sequence[np.ndarray],
    config: SignalConfig,
) -> None:
    left = aggregate_cycle_features(left_cycles)
    right = aggregate_cycle_features(right_cycles)
    for scalar_name in left:
        for aggregation in ("median", "iqr"):
            left_value = float(left[scalar_name][aggregation])
            right_value = float(right[scalar_name][aggregation])
            prefix = f"{modality}__cycle_{scalar_name}_{aggregation}"
            target[f"{prefix}__bilateral_mean"] = float(
                np.mean([left_value, right_value])
            )
            target[f"{prefix}__bilateral_absolute_difference"] = float(
                abs(left_value - right_value)
            )
            target[f"{prefix}__bilateral_symmetric_difference"] = (
                symmetric_difference(left_value, right_value)
            )

    left_template = median_shape_template(
        left_cycles,
        config.cycle_waveform_points,
    )
    right_template = median_shape_template(
        right_cycles,
        config.cycle_waveform_points,
    )
    if np.isfinite(left_template).all() and np.isfinite(right_template).all():
        difference = left_template - right_template
        correlation = float(np.corrcoef(left_template, right_template)[0, 1])
        target[f"{modality}__wave_template_mean_absolute_difference"] = float(
            np.mean(np.abs(difference))
        )
        target[f"{modality}__wave_template_rms_difference"] = float(
            np.sqrt(np.mean(np.square(difference)))
        )
        target[f"{modality}__wave_template_correlation_dissimilarity"] = float(
            1.0 - abs(correlation)
        )

        average_template = 0.5 * (left_template + right_template)
        difference_template = np.abs(difference)
        average_coefficients = dct(average_template, norm="ortho")
        difference_coefficients = dct(difference_template, norm="ortho")
        for coefficient in range(1, 6):
            target[f"{modality}__wave_average_dct_{coefficient}"] = float(
                abs(average_coefficients[coefficient])
            )
            target[f"{modality}__wave_difference_dct_{coefficient}"] = float(
                abs(difference_coefficients[coefficient])
            )
    else:
        target[f"{modality}__wave_template_mean_absolute_difference"] = np.nan
        target[f"{modality}__wave_template_rms_difference"] = np.nan
        target[f"{modality}__wave_template_correlation_dissimilarity"] = np.nan
        for coefficient in range(1, 6):
            target[f"{modality}__wave_average_dct_{coefficient}"] = np.nan
            target[f"{modality}__wave_difference_dct_{coefficient}"] = np.nan


def extract_eligible_features(
    synchronized: pd.DataFrame,
    participant_cycles: pd.DataFrame,
    config: SignalConfig,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Extract cycle-aligned, side-invariant predictors.

    Frozen gyroscope-z cycle timestamps are used only as segmentation markers.
    No gyroscope-z value, cycle duration, amplitude, count or polarity enters
    the predictor matrix.
    """

    features: Dict[str, float] = {}
    qc: Dict[str, float] = {
        "common_overlap_seconds": float(
            synchronized["elapsed_common_s"].iloc[-1]
            - synchronized["elapsed_common_s"].iloc[0]
        )
    }
    epoch_ms = synchronized["epoch_ms"].to_numpy(dtype=float)

    filtered_axes: Dict[Tuple[str, str, str], np.ndarray] = {}
    source_specs = [
        ("acc", "left", "x", "left_acc_x", "left_acc_valid"),
        ("acc", "left", "y", "left_acc_y", "left_acc_valid"),
        ("acc", "left", "z", "left_acc_z", "left_acc_valid"),
        ("acc", "right", "x", "right_acc_x", "right_acc_valid"),
        ("acc", "right", "y", "right_acc_y", "right_acc_valid"),
        ("acc", "right", "z", "right_acc_z", "right_acc_valid"),
        ("gyro_xy", "left", "x", "left_gyro_x", "left_gyro_valid"),
        ("gyro_xy", "left", "y", "left_gyro_y", "left_gyro_valid"),
        ("gyro_xy", "right", "x", "right_gyro_x", "right_gyro_valid"),
        ("gyro_xy", "right", "y", "right_gyro_y", "right_gyro_valid"),
    ]
    for modality, side, axis, value_column, valid_column in source_specs:
        raw = synchronized[value_column].to_numpy(dtype=float)
        valid = synchronized[valid_column].to_numpy(dtype=bool) & np.isfinite(raw)
        filtered_axes[(modality, side, axis)] = filter_signal_segments(
            raw,
            valid,
            config,
        )
        qc[f"{modality}_{side}_{axis}_valid_fraction"] = float(np.mean(valid))

    modality_axes = {
        "acc": ("x", "y", "z"),
        "gyro_xy": ("x", "y"),
    }
    for modality, axes in modality_axes.items():
        side_cycles: Dict[str, List[np.ndarray]] = {}
        for side in ("left", "right"):
            axis_values = [filtered_axes[(modality, side, axis)] for axis in axes]
            magnitude = np.sqrt(
                np.sum(np.square(np.column_stack(axis_values)), axis=1)
            )
            invalid = ~np.logical_and.reduce(
                [np.isfinite(values) for values in axis_values]
            )
            magnitude[invalid] = np.nan
            intervals = participant_cycles.loc[
                participant_cycles["side"].eq(side),
                ["start_epoch_ms", "end_epoch_ms"],
            ]
            cycles = slice_signal_into_cycles(
                magnitude,
                epoch_ms,
                intervals,
                config.minimum_cycle_samples,
            )
            if not cycles:
                raise ValueError(
                    f"No usable {modality} {side} cycles after alignment."
                )
            side_cycles[side] = cycles
            qc[f"{modality}_{side}_frozen_interval_count"] = int(len(intervals))
            qc[f"{modality}_{side}_usable_cycle_count"] = int(len(cycles))

        add_side_invariant_cycle_features(
            features,
            modality,
            side_cycles["left"],
            side_cycles["right"],
            config,
        )

    assert_eligible_feature_names(features.keys())
    return features, qc


def assert_eligible_feature_names(feature_names: Iterable[str]) -> None:
    names = list(feature_names)
    forbidden_tokens = (
        "gyro_z",
        "gyroz",
        "gyroscope_z",
        "severity_score",
        "severity_class",
        "stroke_split",
        "bootstrap_probability",
        "assignment_confidence",
        "healthy_robust_z",
        "severity_contribution",
        "feature_source_id",
        "source_row_label",
    )
    offending = [
        name for name in names if any(token in name.lower() for token in forbidden_tokens)
    ]
    if offending:
        raise AssertionError(
            "Leakage barrier rejected predictor columns: " + ", ".join(offending)
        )
    direct_label_names = LABEL_ONLY_COLUMNS.intersection(names)
    if direct_label_names:
        raise AssertionError(
            "Frozen-label columns entered the predictor matrix: "
            + ", ".join(sorted(direct_label_names))
        )
    allowed_prefixes = ("acc__", "gyro_xy__", "emg__", "legacy_imu__")
    unexpected = [
        name for name in names if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise AssertionError(
            "Predictor columns have unapproved prefixes: "
            + ", ".join(unexpected)
        )


def build_predictor_dataset(
    labels: pd.DataFrame,
    data_dir: Path,
    cycle_intervals: pd.DataFrame,
    config: SignalConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows: List[Dict[str, object]] = []
    qc_rows: List[Dict[str, object]] = []

    for number, label_row in enumerate(labels.itertuples(index=False), start=1):
        participant_key = str(label_row.participant_key)
        cohort = str(label_row.cohort)
        patient_folder = str(label_row.patient_folder)
        participant_dir = data_dir / cohort / patient_folder
        print(f"[features {number:02d}/{len(labels):02d}] {participant_key}")
        synchronized = synchronise_participant(participant_dir, config)
        participant_cycles = cycle_intervals.loc[
            cycle_intervals["participant_key"].eq(participant_key)
        ].copy()
        participant_features, participant_qc = extract_eligible_features(
            synchronized,
            participant_cycles,
            config,
        )
        feature_rows.append(
            {
                "participant_key": participant_key,
                "cohort": cohort,
                "patient_folder": patient_folder,
                **participant_features,
            }
        )
        qc_rows.append(
            {
                "participant_key": participant_key,
                "cohort": cohort,
                "patient_folder": patient_folder,
                **participant_qc,
            }
        )

    features = pd.DataFrame(feature_rows)
    qc = pd.DataFrame(qc_rows)
    if features["participant_key"].duplicated().any():
        raise RuntimeError("Predictor extraction produced duplicate participants.")
    numeric_feature_columns = [
        column
        for column in features.columns
        if column not in {"participant_key", "cohort", "patient_folder"}
    ]
    assert_eligible_feature_names(numeric_feature_columns)

    manifest_rows = []
    for column in numeric_feature_columns:
        if column.startswith("acc__"):
            sensor_set = "accelerometer"
        elif column.startswith("gyro_xy__"):
            sensor_set = "gyroscope_xy"
        else:
            raise AssertionError(f"Unexpected predictor prefix: {column}")
        manifest_rows.append(
            {
                "feature_name": column,
                "sensor_set": sensor_set,
                "waveform_derived": "__wave_" in column,
                "uses_gyroscope_z": False,
                "cycle_aligned": True,
                "side_invariant": True,
                "eligible_for_primary_classifier": "__wave_" not in column,
                "feature_source": "raw_cycle_aligned_imu",
            }
        )
    manifest = pd.DataFrame(manifest_rows)
    return features, qc, manifest


# =============================================================================
# Nested model selection
# =============================================================================


def available_feature_counts(
    requested: Sequence[Any],
    number_of_features: int,
) -> List[Any]:
    counts: List[Any] = []
    for value in requested:
        if value == "all":
            counts.append("all")
        elif int(value) <= number_of_features:
            counts.append(int(value))
    if not counts:
        raise ValueError(
            f"No requested feature count is valid for {number_of_features} predictors."
        )
    return list(dict.fromkeys(counts))


def candidate_columns(
    all_feature_columns: Sequence[str],
    sensor_set: str,
    include_waveform: bool,
) -> List[str]:
    if sensor_set == "accelerometer":
        selected = [column for column in all_feature_columns if column.startswith("acc__")]
    elif sensor_set == "gyroscope_xy":
        selected = [
            column for column in all_feature_columns if column.startswith("gyro_xy__")
        ]
    elif sensor_set == "combined":
        selected = [
            column
            for column in all_feature_columns
            if column.startswith(("acc__", "gyro_xy__"))
        ]
    elif sensor_set == "emg":
        selected = [
            column for column in all_feature_columns if column.startswith("emg__")
        ]
    elif sensor_set == "imu_emg":
        selected = [
            column
            for column in all_feature_columns
            if column.startswith(("acc__", "gyro_xy__", "emg__"))
        ]
    elif sensor_set == "imu_emg_legacy":
        selected = [
            column
            for column in all_feature_columns
            if column.startswith(
                ("acc__", "gyro_xy__", "emg__", "legacy_imu__")
            )
        ]
    else:
        raise ValueError(f"Unknown sensor set: {sensor_set}")
    if not include_waveform:
        selected = [column for column in selected if "__wave_" not in column]
    assert_eligible_feature_names(selected)
    if not selected:
        raise ValueError(
            f"No eligible features for sensor_set={sensor_set}, "
            f"include_waveform={include_waveform}."
        )
    return selected


def build_model_candidates(
    number_of_features: int,
    config: ValidationConfig,
) -> List[Tuple[str, Pipeline, Dict[str, Sequence[Any]]]]:
    feature_counts = available_feature_counts(config.feature_counts, number_of_features)
    def pipeline(classifier: Any) -> Pipeline:
        return Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=f_classif)),
                ("classifier", classifier),
            ]
        )

    candidates = [
        (
            "Logistic Regression",
            pipeline(
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=config.random_state,
                )
            ),
            {
                "selector__k": feature_counts,
                "classifier__C": (0.1, 1.0),
            },
        ),
        (
            "Linear SVM",
            pipeline(
                SVC(
                    kernel="linear",
                    probability=True,
                    class_weight="balanced",
                    random_state=config.random_state,
                )
            ),
            {
                "selector__k": feature_counts,
                "classifier__C": (0.1, 1.0),
            },
        ),
        (
            "RBF SVM",
            pipeline(
                SVC(
                    kernel="rbf",
                    probability=True,
                    class_weight="balanced",
                    random_state=config.random_state,
                )
            ),
            {
                "selector__k": feature_counts,
                "classifier__C": (0.1, 1.0),
                "classifier__gamma": ("scale", 0.1),
            },
        ),
        (
            "k-Nearest Neighbours",
            pipeline(KNeighborsClassifier()),
            {
                "selector__k": feature_counts,
                "classifier__n_neighbors": (3, 5),
                "classifier__weights": ("uniform", "distance"),
            },
        ),
        (
            "Gaussian Naive Bayes",
            pipeline(GaussianNB()),
            {
                "selector__k": feature_counts,
                "classifier__var_smoothing": (1e-9, 1e-7),
            },
        ),
        (
            "Shrinkage LDA",
            pipeline(LinearDiscriminantAnalysis(solver="lsqr")),
            {
                "selector__k": feature_counts,
                "classifier__shrinkage": ("auto", 0.5),
            },
        ),
        (
            "Decision Tree",
            pipeline(
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=config.random_state,
                )
            ),
            {
                "selector__k": feature_counts,
                "classifier__max_depth": (2, None),
                "classifier__min_samples_leaf": (1, 3),
            },
        ),
        (
            "Random Forest",
            pipeline(
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=config.random_state,
                    n_jobs=1,
                )
            ),
            {
                "selector__k": feature_counts,
                "classifier__max_depth": (3, None),
                "classifier__min_samples_leaf": (1, 2),
            },
        ),
        (
            "Extra Trees",
            pipeline(
                ExtraTreesClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=config.random_state,
                    n_jobs=1,
                )
            ),
            {
                "selector__k": feature_counts,
                "classifier__max_depth": (3, None),
                "classifier__min_samples_leaf": (1, 2),
            },
        ),
        (
            "Gradient Boosting",
            pipeline(GradientBoostingClassifier(random_state=config.random_state)),
            {
                "selector__k": feature_counts,
                "classifier__n_estimators": (50, 100),
                "classifier__learning_rate": (0.05, 0.1),
            },
        ),
        (
            "AdaBoost",
            pipeline(AdaBoostClassifier(random_state=config.random_state)),
            {
                "selector__k": feature_counts,
                "classifier__n_estimators": (50, 100),
                "classifier__learning_rate": (0.1, 0.5),
            },
        ),
    ]
    candidate_names = tuple(name for name, _, _ in candidates)
    if candidate_names != CLASSIFICATION_MODEL_NAMES:
        raise AssertionError("Classification model registry is inconsistent.")
    return candidates


def determine_inner_splits(y: np.ndarray, requested: int) -> int:
    class_counts = Counter(int(value) for value in y)
    if len(class_counts) < 2:
        raise ValueError("Inner model selection requires at least two classes.")
    n_splits = min(requested, min(class_counts.values()))
    if n_splits < 2:
        raise ValueError(
            f"Insufficient class counts for grouped validation: {dict(class_counts)}"
        )
    return int(n_splits)


def nested_select_model(
    training_frame: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    all_feature_columns: Sequence[str],
    include_waveform: bool,
    outer_participant: str,
    analysis_name: str,
    stage: str,
    config: ValidationConfig,
    candidate_sensor_sets: Sequence[str] = RAW_IMU_SENSOR_SETS,
) -> Tuple[
    Pipeline,
    List[str],
    Dict[str, object],
    List[Dict[str, object]],
    List[str],
    Dict[str, Dict[str, object]],
]:
    n_splits = determine_inner_splits(y, config.inner_splits)
    inner_cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state,
    )
    scoring = {
        "macro_f1": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    }

    best: Dict[str, object] | None = None
    best_by_model: Dict[str, Dict[str, object]] = {}
    candidate_summaries: List[Dict[str, object]] = []
    for sensor_set in candidate_sensor_sets:
        columns = candidate_columns(
            all_feature_columns,
            sensor_set,
            include_waveform,
        )
        X = training_frame[columns]
        for model_name, pipeline, parameter_grid in build_model_candidates(
            len(columns),
            config,
        ):
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring=scoring,
                refit=config.primary_metric,
                cv=inner_cv,
                n_jobs=config.n_jobs,
                error_score="raise",
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Features .* are constant",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered",
                    category=RuntimeWarning,
                )
                grid.fit(X, y, groups=groups)

            best_index = int(grid.best_index_)
            macro_f1 = float(grid.cv_results_["mean_test_macro_f1"][best_index])
            balanced = float(
                grid.cv_results_["mean_test_balanced_accuracy"][best_index]
            )
            summary = {
                "analysis": analysis_name,
                "outer_participant": outer_participant,
                "stage": stage,
                "sensor_set": sensor_set,
                "model": model_name,
                "inner_splits": n_splits,
                "include_waveform_predictors": include_waveform,
                "candidate_feature_count": len(columns),
                "best_inner_macro_f1": macro_f1,
                "best_inner_balanced_accuracy": balanced,
                "best_parameters": json.dumps(
                    grid.best_params_,
                    sort_keys=True,
                    default=json_default,
                ),
            }
            candidate_summaries.append(summary)
            comparison = (macro_f1, balanced)
            model_best = best_by_model.get(model_name)
            if model_best is None or comparison > model_best["comparison"]:  # type: ignore
                selector = grid.best_estimator_.named_steps["selector"]
                support = selector.get_support()
                selected_model_features = [
                    column for column, keep in zip(columns, support) if bool(keep)
                ]
                best_by_model[model_name] = {
                    "comparison": comparison,
                    "estimator": grid.best_estimator_,
                    "columns": list(columns),
                    "summary": dict(summary),
                    "selected_features": selected_model_features,
                }
            if best is None or comparison > best["comparison"]: # type: ignore
                best = {
                    "comparison": comparison,
                    "estimator": grid.best_estimator_,
                    "columns": columns,
                    "summary": summary,
                }

    if best is None:
        raise RuntimeError("No model candidate completed inner validation.")

    selected_summary = dict(best["summary"]) # type: ignore
    selected_summary["selected"] = True # type: ignore
    
    for row in candidate_summaries:
        row["selected_overall"] = (
            row["sensor_set"] == selected_summary["sensor_set"] # type: ignore
            and row["model"] == selected_summary["model"]       # type: ignore
        )
        model_best_summary = best_by_model[str(row["model"])]["summary"]
        row["selected_within_model"] = (
            row["sensor_set"] == model_best_summary["sensor_set"]  # type: ignore
        )
        # Retain the original field for compatibility with earlier outputs.
        row["selected"] = row["selected_overall"]

    estimator = best["estimator"]
    columns = list(best["columns"]) # type: ignore
    selector = estimator.named_steps["selector"] # type: ignore
    support = selector.get_support()
    selected_features = [
        column for column, keep in zip(columns, support) if bool(keep)
    ]
    
    return (
        estimator,
        columns,
        selected_summary,
        candidate_summaries,
        selected_features,
        best_by_model,
    )  # type: ignore


def aligned_probabilities(
    estimator: Pipeline,
    X: pd.DataFrame,
    expected_classes: Sequence[int],
) -> np.ndarray:
    probabilities = estimator.predict_proba(X)
    observed_classes = [int(value) for value in estimator.classes_]
    aligned = np.zeros((len(X), len(expected_classes)), dtype=float)
    for source_index, class_id in enumerate(observed_classes):
        target_index = list(expected_classes).index(class_id)
        aligned[:, target_index] = probabilities[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise RuntimeError("A classifier returned invalid probability rows.")
    return aligned / row_sums


def build_regression_candidates(
    number_of_features: int,
    config: ValidationConfig,
) -> List[Tuple[str, Pipeline, Dict[str, Sequence[Any]]]]:
    feature_counts = available_feature_counts(config.feature_counts, number_of_features)
    common_steps = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_regression)),
    ]
    ridge = Pipeline(
        common_steps
        + [
            ("regressor", Ridge()),
        ]
    )
    linear_svr = Pipeline(
        common_steps
        + [
            (
                "regressor",
                LinearSVR(
                    dual="auto",
                    max_iter=10000,
                    random_state=config.random_state,
                ),
            )
        ]
    )
    return [
        (
            "Ridge Regression",
            ridge,
            {
                "selector__k": feature_counts,
                "regressor__alpha": (0.1, 1.0, 10.0),
            },
        ),
        (
            "Linear SVR",
            linear_svr,
            {
                "selector__k": feature_counts,
                "regressor__C": (0.1, 1.0),
                "regressor__epsilon": (0.0, 0.1),
            },
        ),
    ]


def nested_select_regressor(
    training_frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    all_feature_columns: Sequence[str],
    outer_participant: str,
    analysis_name: str,
    config: ValidationConfig,
) -> Tuple[Pipeline, List[str], Dict[str, object], List[Dict[str, object]], List[str]]:
    n_splits = min(config.inner_splits, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("Regression inner validation requires at least two groups.")
    inner_cv = GroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_state,
    )

    best: Dict[str, object] | None = None
    candidate_summaries: List[Dict[str, object]] = []
    for sensor_set in ("accelerometer", "gyroscope_xy", "combined"):
        columns = candidate_columns(
            all_feature_columns,
            sensor_set,
            include_waveform=False,
        )
        X = training_frame[columns]
        for model_name, pipeline, parameter_grid in build_regression_candidates(
            len(columns),
            config,
        ):
            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=parameter_grid,
                scoring="neg_mean_absolute_error",
                refit=True,
                cv=inner_cv,
                n_jobs=config.n_jobs,
                error_score="raise",
                return_train_score=False,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Features .* are constant",
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered",
                    category=RuntimeWarning,
                )
                grid.fit(X, target, groups=groups)

            inner_mae = float(-grid.best_score_)
            summary = {
                "analysis": analysis_name,
                "outer_participant": outer_participant,
                "stage": "continuous_score_regression",
                "sensor_set": sensor_set,
                "model": model_name,
                "inner_splits": n_splits,
                "include_waveform_predictors": False,
                "candidate_feature_count": len(columns),
                "best_inner_mae": inner_mae,
                "best_parameters": json.dumps(
                    grid.best_params_,
                    sort_keys=True,
                    default=json_default,
                ),
            }
            candidate_summaries.append(summary)
            if best is None or inner_mae < float(best["inner_mae"]):
                best = {
                    "inner_mae": inner_mae,
                    "estimator": grid.best_estimator_,
                    "columns": columns,
                    "summary": summary,
                }

    if best is None:
        raise RuntimeError("No regression candidate completed inner validation.")

    selected_summary = dict(best["summary"])  # type: ignore[arg-type]
    selected_summary["selected"] = True
    for row in candidate_summaries:
        row["selected"] = (
            row["sensor_set"] == selected_summary["sensor_set"]
            and row["model"] == selected_summary["model"]
        )

    estimator = best["estimator"]
    columns = list(best["columns"])  # type: ignore[arg-type]
    selector = estimator.named_steps["selector"]  # type: ignore[union-attr]
    support = selector.get_support()
    selected_features = [
        column for column, keep in zip(columns, support) if bool(keep)
    ]
    return (
        estimator,  # type: ignore[return-value]
        columns,
        selected_summary,
        candidate_summaries,
        selected_features,
    )


# =============================================================================
# Direct and hierarchical outer LOSO analyses
# =============================================================================


def merge_labels_and_features(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    metadata = {"participant_key", "cohort", "patient_folder"}
    feature_columns = [column for column in features.columns if column not in metadata]
    assert_eligible_feature_names(feature_columns)
    
    merged = labels.merge(
        features[["participant_key", *feature_columns]],
        on="participant_key",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing = merged.loc[merged["_merge"] != "both", "participant_key"].tolist()
    
    if missing:
        raise ValueError(f"No raw-data predictor row for frozen labels: {missing}")
    
    merged = merged.drop(columns="_merge")
    
    if merged[feature_columns].isna().all(axis=1).any():
        failed = merged.loc[
            merged[feature_columns].isna().all(axis=1), "participant_key"
        ].tolist()
        raise ValueError(f"Participants have no usable eligible predictors: {failed}")
    
    return merged


def run_direct_nested_loso(
    analysis_name: str,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    include_waveform: bool,
    output_dir: Path,
    config: ValidationConfig,
    candidate_sensor_sets: Sequence[str] = RAW_IMU_SENSOR_SETS,
) -> Dict[str, object]:
    frame = merge_labels_and_features(labels, features)
    metadata = set(labels.columns).union({"participant_key", "cohort", "patient_folder"})
    all_feature_columns = [column for column in frame.columns if column not in metadata]
    assert_eligible_feature_names(all_feature_columns)

    prediction_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    selected_feature_rows: List[Dict[str, object]] = []
    model_benchmark_prediction_rows: List[Dict[str, object]] = []
    model_benchmark_selection_rows: List[Dict[str, object]] = []
    model_benchmark_feature_rows: List[Dict[str, object]] = []

    for fold_number, test_index in enumerate(range(len(frame)), start=1):
        test_row = frame.iloc[[test_index]]
        train_frame = frame.drop(index=frame.index[test_index]).reset_index(drop=True)
        outer_participant = str(test_row["participant_key"].iloc[0])
        print(f"[{analysis_name} {fold_number:02d}/{len(frame):02d}] " f"held out {outer_participant}")
        y_train = train_frame["severity_class_id"].to_numpy(dtype=int)
        groups_train = train_frame["participant_key"].to_numpy(dtype=str)
        
        (
            estimator,
            columns,
            selected,
            candidates,
            selected_features,
            best_by_model,
        ) = nested_select_model(
            train_frame,
            y_train,
            groups_train,
            all_feature_columns,
            include_waveform,
            outer_participant,
            analysis_name,
            "direct_three_class",
            config,
            candidate_sensor_sets,
        )
        
        probabilities = aligned_probabilities(estimator, test_row[columns], CLASS_IDS)[0]
        predicted_class = int(estimator.predict(test_row[columns])[0])
        probability_argmax_class = int(CLASS_IDS[int(np.argmax(probabilities))])
        true_class = int(test_row["severity_class_id"].iloc[0])

        prediction_rows.append(
            {
                "analysis": analysis_name,
                "outer_fold": fold_number,
                "participant_key": outer_participant,
                "cohort": str(test_row["cohort"].iloc[0]),
                "true_severity_class_id": true_class,
                "true_severity_class": str(test_row["severity_class"].iloc[0]),
                "predicted_severity_class_id": predicted_class,
                "predicted_severity_class": CLASS_SHORT_NAMES[predicted_class],
                "probability_low": float(probabilities[0]),
                "probability_moderate": float(probabilities[1]),
                "probability_high": float(probabilities[2]),
                "probability_argmax_class_id": probability_argmax_class,
                "prediction_probability_disagreement": (
                    predicted_class != probability_argmax_class
                ),
                "correct_prediction": predicted_class == true_class,
                "sensor_derived_severity_score": float(
                    test_row["sensor_derived_severity_score"].iloc[0]
                ),
                "stroke_split_borderline": bool(
                    test_row["stroke_split_borderline"].iloc[0]
                ),
                "selected_model": selected["model"],
                "selected_sensor_set": selected["sensor_set"],
                "selected_feature_count": len(selected_features),
                "inner_macro_f1": selected["best_inner_macro_f1"],
                "inner_balanced_accuracy": selected[
                    "best_inner_balanced_accuracy"
                ],
                "selected_parameters": selected["best_parameters"],
            }
        )
        selection_rows.append(selected)
        candidate_rows.extend(candidates)
        selected_feature_rows.extend(
            {
                "analysis": analysis_name,
                "outer_participant": outer_participant,
                "stage": "direct_three_class",
                "feature_name": feature,
            }
            for feature in selected_features
        )

        for model_name in CLASSIFICATION_MODEL_NAMES:
            model_result = best_by_model[model_name]
            model_estimator = model_result["estimator"]
            model_columns = list(model_result["columns"])  # type: ignore
            model_summary = dict(model_result["summary"])  # type: ignore
            model_features = list(model_result["selected_features"])  # type: ignore
            model_probabilities = aligned_probabilities(
                model_estimator,  # type: ignore
                test_row[model_columns],
                CLASS_IDS,
            )[0]
            model_prediction = int(
                model_estimator.predict(test_row[model_columns])[0]  # type: ignore
            )
            model_probability_argmax = int(
                CLASS_IDS[int(np.argmax(model_probabilities))]
            )
            model_benchmark_prediction_rows.append(
                {
                    "analysis": analysis_name,
                    "model": model_name,
                    "outer_fold": fold_number,
                    "participant_key": outer_participant,
                    "cohort": str(test_row["cohort"].iloc[0]),
                    "true_severity_class_id": true_class,
                    "true_severity_class": str(
                        test_row["severity_class"].iloc[0]
                    ),
                    "predicted_severity_class_id": model_prediction,
                    "predicted_severity_class": CLASS_SHORT_NAMES[
                        model_prediction
                    ],
                    "probability_low": float(model_probabilities[0]),
                    "probability_moderate": float(model_probabilities[1]),
                    "probability_high": float(model_probabilities[2]),
                    "probability_argmax_class_id": model_probability_argmax,
                    "prediction_probability_disagreement": (
                        model_prediction != model_probability_argmax
                    ),
                    "correct_prediction": model_prediction == true_class,
                    "sensor_derived_severity_score": float(
                        test_row["sensor_derived_severity_score"].iloc[0]
                    ),
                    "stroke_split_borderline": bool(
                        test_row["stroke_split_borderline"].iloc[0]
                    ),
                    "selected_sensor_set": model_summary["sensor_set"],
                    "selected_feature_count": len(model_features),
                    "inner_macro_f1": model_summary["best_inner_macro_f1"],
                    "inner_balanced_accuracy": model_summary[
                        "best_inner_balanced_accuracy"
                    ],
                    "selected_parameters": model_summary["best_parameters"],
                }
            )
            benchmark_selection = dict(model_summary)
            benchmark_selection["model"] = model_name
            model_benchmark_selection_rows.append(benchmark_selection)
            model_benchmark_feature_rows.extend(
                {
                    "analysis": analysis_name,
                    "model": model_name,
                    "outer_participant": outer_participant,
                    "stage": "direct_three_class",
                    "feature_name": feature,
                }
                for feature in model_features
            )

    summary = save_analysis_outputs(
        analysis_name,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(selected_feature_rows),
        output_dir,
        config,
    )
    summary["all_model_comparison"] = save_model_benchmark_outputs(
        analysis_name=analysis_name,
        predictions=pd.DataFrame(model_benchmark_prediction_rows),
        selections=pd.DataFrame(model_benchmark_selection_rows),
        selected_features=pd.DataFrame(model_benchmark_feature_rows),
        output_dir=output_dir,
        config=config,
    )
    analysis_dir = output_dir / safe_name(analysis_name)
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    return summary


def calculate_regression_metrics(
    predictions: pd.DataFrame,
    subset_name: str,
) -> Dict[str, object]:
    true_score = predictions["true_sgas_score"].to_numpy(dtype=float)
    predicted_score = predictions["predicted_sgas_score"].to_numpy(dtype=float)
    correlation = spearmanr(true_score, predicted_score)
    return {
        "subset": subset_name,
        "participant_count": int(len(predictions)),
        "mae": float(mean_absolute_error(true_score, predicted_score)),
        "spearman_rho": float(correlation.statistic),
        "spearman_p_value": float(correlation.pvalue),
    }


def regression_bootstrap_intervals(
    predictions: pd.DataFrame,
    config: ValidationConfig,
) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(config.random_state)
    grouped_indices = [
        group.index.to_numpy(dtype=int)
        for _, group in predictions.reset_index(drop=True).groupby(
            "true_severity_class_id"
        )
    ]
    mae_values: List[float] = []
    correlation_values: List[float] = []
    true_score = predictions["true_sgas_score"].to_numpy(dtype=float)
    predicted_score = predictions["predicted_sgas_score"].to_numpy(dtype=float)
    for _ in range(config.bootstrap_iterations):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in grouped_indices
            ]
        )
        mae_values.append(
            float(mean_absolute_error(true_score[sampled], predicted_score[sampled]))
        )
        rho = float(spearmanr(true_score[sampled], predicted_score[sampled]).statistic)
        if np.isfinite(rho):
            correlation_values.append(rho)
    return {
        "mae": (
            float(np.percentile(mae_values, 2.5)),
            float(np.percentile(mae_values, 97.5)),
        ),
        "spearman_rho": (
            float(np.percentile(correlation_values, 2.5)),
            float(np.percentile(correlation_values, 97.5)),
        ),
    }


def plot_regression_predictions(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    for class_id in CLASS_IDS:
        subset = predictions.loc[
            predictions["true_severity_class_id"].eq(class_id)
        ]
        axis.scatter(
            subset["true_sgas_score"],
            subset["predicted_sgas_score"],
            color=CLASS_PLOT_COLORS[class_id],
            label=CLASS_SHORT_NAMES[class_id],
            s=55,
            alpha=0.85,
        )
    minimum = float(
        min(predictions["true_sgas_score"].min(), predictions["predicted_sgas_score"].min())
    )
    maximum = float(
        max(predictions["true_sgas_score"].max(), predictions["predicted_sgas_score"].max())
    )
    axis.plot([minimum, maximum], [minimum, maximum], "--", color=GREY_DARK)
    axis.set_xlabel("True frozen SGAS-v2 score")
    axis.set_ylabel("Outer-LOSO predicted SGAS-v2 score")
    axis.set_title("Continuous-score regression")
    axis.grid(alpha=0.25)
    axis.legend(title="Frozen class")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_regression_residuals(
    predictions: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    residual = (
        predictions["predicted_sgas_score"] - predictions["true_sgas_score"]
    )
    axis.scatter(
        predictions["true_sgas_score"], residual, color=BLUE_MEDIUM, s=50
    )
    axis.axhline(0.0, linestyle="--", color=GREY_DARK)
    axis.set_xlabel("True frozen SGAS-v2 score")
    axis.set_ylabel("Prediction error")
    axis.set_title("Continuous-score regression residuals")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def run_continuous_score_nested_loso(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    config: ValidationConfig,
) -> Dict[str, object]:
    analysis_name = "continuous_score_regression"
    analysis_dir = output_dir / analysis_name
    analysis_dir.mkdir(parents=True, exist_ok=True)
    frame = merge_labels_and_features(labels, features)
    metadata = set(labels.columns).union({"participant_key", "cohort", "patient_folder"})
    all_feature_columns = [column for column in frame.columns if column not in metadata]
    assert_eligible_feature_names(all_feature_columns)

    prediction_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    selected_feature_rows: List[Dict[str, object]] = []
    for fold_number, test_index in enumerate(range(len(frame)), start=1):
        test_row = frame.iloc[[test_index]]
        train_frame = frame.drop(index=frame.index[test_index]).reset_index(drop=True)
        outer_participant = str(test_row["participant_key"].iloc[0])
        print(
            f"[{analysis_name} {fold_number:02d}/{len(frame):02d}] "
            f"held out {outer_participant}"
        )
        target = train_frame["sensor_derived_severity_score"].to_numpy(dtype=float)
        groups = train_frame["participant_key"].to_numpy(dtype=str)
        estimator, columns, selected, candidates, selected_features = nested_select_regressor(
            train_frame,
            target,
            groups,
            all_feature_columns,
            outer_participant,
            analysis_name,
            config,
        )
        predicted_score = float(estimator.predict(test_row[columns])[0])
        true_score = float(test_row["sensor_derived_severity_score"].iloc[0])
        prediction_rows.append(
            {
                "analysis": analysis_name,
                "outer_fold": fold_number,
                "participant_key": outer_participant,
                "cohort": str(test_row["cohort"].iloc[0]),
                "true_severity_class_id": int(test_row["severity_class_id"].iloc[0]),
                "true_sgas_score": true_score,
                "predicted_sgas_score": predicted_score,
                "absolute_error": abs(predicted_score - true_score),
                "selected_model": selected["model"],
                "selected_sensor_set": selected["sensor_set"],
                "selected_feature_count": len(selected_features),
                "inner_mae": selected["best_inner_mae"],
                "selected_parameters": selected["best_parameters"],
            }
        )
        selection_rows.append(selected)
        candidate_rows.extend(candidates)
        selected_feature_rows.extend(
            {
                "analysis": analysis_name,
                "outer_participant": outer_participant,
                "stage": "continuous_score_regression",
                "feature_name": feature,
            }
            for feature in selected_features
        )

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    candidates = pd.DataFrame(candidate_rows)
    selected_features = pd.DataFrame(selected_feature_rows)
    overall_metrics = calculate_regression_metrics(predictions, "all_participants")
    stroke_predictions = predictions.loc[predictions["cohort"].eq("Stroke")].copy()
    stroke_metrics = calculate_regression_metrics(stroke_predictions, "stroke_only")
    intervals = regression_bootstrap_intervals(predictions, config)
    overall_metrics.update(
        {
            "mae_ci_95_lower": intervals["mae"][0],
            "mae_ci_95_upper": intervals["mae"][1],
            "spearman_rho_ci_95_lower": intervals["spearman_rho"][0],
            "spearman_rho_ci_95_upper": intervals["spearman_rho"][1],
        }
    )
    metrics = pd.DataFrame([overall_metrics, stroke_metrics])

    predictions.to_csv(analysis_dir / "outer_loso_score_predictions.csv", index=False)
    metrics.to_csv(analysis_dir / "regression_metrics.csv", index=False)
    selections.to_csv(analysis_dir / "outer_fold_selections.csv", index=False)
    candidates.to_csv(analysis_dir / "inner_candidate_summary.csv", index=False)
    selected_features.to_csv(
        analysis_dir / "selected_features_by_outer_fold.csv",
        index=False,
    )
    plot_regression_predictions(
        predictions,
        analysis_dir / "true_vs_predicted_sgas_score.png",
    )
    plot_regression_residuals(
        predictions,
        analysis_dir / "regression_residuals.png",
    )
    plot_selection_frequencies(
        selections,
        analysis_dir / "model_and_sensor_selection_frequencies.png",
        "Continuous-score regression: inner-selection frequencies",
    )
    plot_feature_selection_frequency(
        selected_features,
        analysis_dir / "feature_selection_frequency.png",
        "Continuous-score regression: selected predictors",
    )

    summary = {
        "analysis": analysis_name,
        "participant_count": int(len(predictions)),
        "overall_metrics": overall_metrics,
        "stroke_only_metrics": stroke_metrics,
        "output_directory": str(analysis_dir),
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    return summary


def run_hierarchical_nested_loso(
    analysis_name: str,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    include_waveform: bool,
    output_dir: Path,
    config: ValidationConfig,
    candidate_sensor_sets: Sequence[str] = RAW_IMU_SENSOR_SETS,
) -> Dict[str, object]:
    frame = merge_labels_and_features(labels, features)
    metadata = set(labels.columns).union({"participant_key", "cohort", "patient_folder"})
    all_feature_columns = [column for column in frame.columns if column not in metadata]
    assert_eligible_feature_names(all_feature_columns)

    prediction_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    candidate_rows: List[Dict[str, object]] = []
    selected_feature_rows: List[Dict[str, object]] = []

    for fold_number, test_index in enumerate(range(len(frame)), start=1):
        test_row = frame.iloc[[test_index]]
        train_frame = frame.drop(index=frame.index[test_index]).reset_index(drop=True)
        outer_participant = str(test_row["participant_key"].iloc[0])
        print(
            f"[{analysis_name} {fold_number:02d}/{len(frame):02d}] "
            f"held out {outer_participant}"
        )

        stage1_y = (train_frame["severity_class_id"].to_numpy(dtype=int) > 0).astype(int)
        stage1_groups = train_frame["participant_key"].to_numpy(dtype=str)
        (
            stage1_model,
            stage1_columns,
            stage1_selected,
            stage1_candidates,
            stage1_features,
            _stage1_models,
        ) = nested_select_model(
            train_frame,
            stage1_y,
            stage1_groups,
            all_feature_columns,
            include_waveform,
            outer_participant,
            analysis_name,
            "healthy_vs_stroke",
            config,
            candidate_sensor_sets,
        )
        p_stage1 = aligned_probabilities(
            stage1_model,
            test_row[stage1_columns],
            (0, 1),
        )[0]

        stroke_train = train_frame.loc[
            train_frame["severity_class_id"].isin((1, 2))
        ].reset_index(drop=True)
        stage2_y = (
            stroke_train["severity_class_id"].to_numpy(dtype=int) - 1
        )
        stage2_groups = stroke_train["participant_key"].to_numpy(dtype=str)
        (
            stage2_model,
            stage2_columns,
            stage2_selected,
            stage2_candidates,
            stage2_features,
            _stage2_models,
        ) = nested_select_model(
            stroke_train,
            stage2_y,
            stage2_groups,
            all_feature_columns,
            include_waveform,
            outer_participant,
            analysis_name,
            "moderate_vs_high_within_stroke",
            config,
            candidate_sensor_sets,
        )
        p_stage2 = aligned_probabilities(
            stage2_model,
            test_row[stage2_columns],
            (0, 1),
        )[0]

        probabilities = np.array(
            [
                p_stage1[0],
                p_stage1[1] * p_stage2[0],
                p_stage1[1] * p_stage2[1],
            ],
            dtype=float,
        )
        probabilities = probabilities / probabilities.sum()
        predicted_class = int(np.argmax(probabilities))
        true_class = int(test_row["severity_class_id"].iloc[0])
        prediction_rows.append(
            {
                "analysis": analysis_name,
                "outer_fold": fold_number,
                "participant_key": outer_participant,
                "cohort": str(test_row["cohort"].iloc[0]),
                "true_severity_class_id": true_class,
                "true_severity_class": str(test_row["severity_class"].iloc[0]),
                "predicted_severity_class_id": predicted_class,
                "predicted_severity_class": CLASS_SHORT_NAMES[predicted_class],
                "probability_low": float(probabilities[0]),
                "probability_moderate": float(probabilities[1]),
                "probability_high": float(probabilities[2]),
                "probability_stroke_stage1": float(p_stage1[1]),
                "conditional_probability_high_stage2": float(p_stage2[1]),
                "correct_prediction": predicted_class == true_class,
                "sensor_derived_severity_score": float(
                    test_row["sensor_derived_severity_score"].iloc[0]
                ),
                "stroke_split_borderline": bool(
                    test_row["stroke_split_borderline"].iloc[0]
                ),
                "selected_model_stage1": stage1_selected["model"],
                "selected_sensor_set_stage1": stage1_selected["sensor_set"],
                "selected_feature_count_stage1": len(stage1_features),
                "selected_model_stage2": stage2_selected["model"],
                "selected_sensor_set_stage2": stage2_selected["sensor_set"],
                "selected_feature_count_stage2": len(stage2_features),
            }
        )
        selection_rows.extend((stage1_selected, stage2_selected))
        candidate_rows.extend(stage1_candidates)
        candidate_rows.extend(stage2_candidates)
        selected_feature_rows.extend(
            {
                "analysis": analysis_name,
                "outer_participant": outer_participant,
                "stage": "healthy_vs_stroke",
                "feature_name": feature,
            }
            for feature in stage1_features
        )
        selected_feature_rows.extend(
            {
                "analysis": analysis_name,
                "outer_participant": outer_participant,
                "stage": "moderate_vs_high_within_stroke",
                "feature_name": feature,
            }
            for feature in stage2_features
        )

    return save_analysis_outputs(
        analysis_name,
        pd.DataFrame(prediction_rows),
        pd.DataFrame(selection_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(selected_feature_rows),
        output_dir,
        config,
    )


# =============================================================================
# Pooled evaluation, uncertainty and plots
# =============================================================================


def metric_values(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def stratified_bootstrap_intervals(y_true: np.ndarray, y_pred: np.ndarray, config: ValidationConfig) -> Dict[str, Tuple[float, float]]:
    rng = np.random.default_rng(config.random_state)
    
    class_indices = { class_id: np.flatnonzero(y_true == class_id) for class_id in CLASS_IDS }
    distributions: Dict[str, List[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
    }
    for _ in range(config.bootstrap_iterations):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True) for indices in class_indices.values()
            ]
        )
        
        values = metric_values(y_true[sampled], y_pred[sampled])
        for metric, value in values.items():
            distributions[metric].append(value)
    
    return {
        metric: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for metric, values in distributions.items()
    }


def build_metric_tables(predictions: pd.DataFrame, config: ValidationConfig) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    y_true = predictions["true_severity_class_id"].to_numpy(dtype=int)
    y_pred = predictions["predicted_severity_class_id"].to_numpy(dtype=int)
    point = metric_values(y_true, y_pred)
    intervals = stratified_bootstrap_intervals(y_true, y_pred, config)
    
    overall = pd.DataFrame(
        [
            {
                "metric": metric,
                "value": value,
                "ci_95_lower": intervals[metric][0],
                "ci_95_upper": intervals[metric][1],
                "evaluation_unit": "pooled outer-LOSO participant predictions",
            }
            for metric, value in point.items()
        ]
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "class_id": CLASS_IDS,
            "class_name": [CLASS_SHORT_NAMES[class_id] for class_id in CLASS_IDS],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int), # type: ignore
        }
    )
    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    return overall, per_class, matrix


def calculate_probability_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    y_true = predictions["true_severity_class_id"].to_numpy(dtype=int)
    
    probabilities = predictions[["probability_low", "probability_moderate", "probability_high"]].to_numpy(dtype=float)
    binary_truth = label_binarize(y_true, classes=CLASS_IDS)
    rows: List[Dict[str, object]] = []
    
    for index, class_id in enumerate(CLASS_IDS):
        class_truth = binary_truth[:, index] # type: ignore
        if np.unique(class_truth).size < 2:
            continue
    
        rows.append(
            {
                "class_id": class_id,
                "class_name": CLASS_SHORT_NAMES[class_id],
                "roc_auc_ovr": float(
                    roc_auc_score(class_truth, probabilities[:, index])
                ),
                "average_precision_ovr": float(
                    average_precision_score(class_truth, probabilities[:, index])
                ),
            }
        )
    rows.append(
        {
            "class_id": "macro",
            "class_name": "Macro average",
            "roc_auc_ovr": float(
                roc_auc_score(
                    binary_truth,
                    probabilities,
                    average="macro",
                    multi_class="ovr",
                )
            ),
            "average_precision_ovr": float(
                average_precision_score(
                    binary_truth,
                    probabilities,
                    average="macro",
                )
            ),
        }
    )
    return pd.DataFrame(rows)


def plot_overall_metrics(overall: pd.DataFrame, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    values = overall["value"].to_numpy(dtype=float)
    lower = values - overall["ci_95_lower"].to_numpy(dtype=float)
    upper = overall["ci_95_upper"].to_numpy(dtype=float) - values
    labels = [
        {"accuracy": "Accuracy", "balanced_accuracy": "Balanced accuracy", "macro_f1": "Macro-F1"}[metric]
        for metric in overall["metric"]
    ]
    axis.bar(labels, values, color=[BLUE_DARK, BLUE_MEDIUM, BLUE_LIGHT])
    axis.errorbar(
        np.arange(len(values)),
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor=GREY_DARK,
        capsize=5,
        linewidth=1.2,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        axis.text(index, min(value + 0.035, 0.97), f"{value:.3f}", ha="center")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(matrix: np.ndarray, output_path: Path, title: str) -> None:
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    tick_labels = [CLASS_SHORT_NAMES[class_id] for class_id in CLASS_IDS]
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=axes[0],
    )
    axes[0].set_title("Counts")
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        cbar=False,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=axes[1],
    )
    axes[1].set_title("Row-normalized")
    for axis in axes:
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_per_class_metrics(per_class: pd.DataFrame, output_path: Path, title: str) -> None:
    plot_frame = per_class.melt(
        id_vars=["class_id", "class_name"],
        value_vars=["recall", "f1"],
        var_name="metric",
        value_name="value",
    )
    figure, axis = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=plot_frame,
        x="class_name",
        y="value",
        hue="metric",
        palette={"recall": BLUE_MEDIUM, "f1": GREY_MEDIUM},
        ax=axis,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Severity class")
    axis.set_ylabel("Score")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Metric")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_probability_heatmap(
    predictions: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    ordered = predictions.sort_values(
        ["true_severity_class_id", "sensor_derived_severity_score", "participant_key"]
    )
    probability_columns = [
        "probability_low",
        "probability_moderate",
        "probability_high",
    ]
    figure_height = max(6.0, 0.30 * len(ordered))
    figure, axis = plt.subplots(figsize=(8, figure_height))
    sns.heatmap(
        ordered[probability_columns].to_numpy(dtype=float),
        annot=True,
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
        cmap="Blues",
        xticklabels=["Low", "Moderate", "High"],
        yticklabels=ordered["participant_key"],
        ax=axis,
    )
    axis.set_xlabel("Predicted class probability")
    axis.set_ylabel("Outer-LOSO participant")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_selection_frequencies(
    selections: pd.DataFrame,
    output_path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    model_counts = selections["model"].value_counts().sort_values(ascending=False)
    sensor_counts = selections["sensor_set"].value_counts().sort_values(ascending=False)
    axes[0].bar(model_counts.index, model_counts.values, color=BLUE_MEDIUM)
    axes[0].set_title("Selected model")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(sensor_counts.index, sensor_counts.values, color=GREY_MEDIUM)
    axes[1].set_title("Selected sensor set")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.set_ylabel("Outer-fold selections")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_feature_selection_frequency(
    selected_features: pd.DataFrame,
    output_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    counts = selected_features["feature_name"].value_counts().head(top_n).sort_values()
    figure_height = max(6.0, 0.35 * len(counts))
    figure, axis = plt.subplots(figsize=(10, figure_height))
    axis.barh(counts.index, counts.values, color=BLUE_MEDIUM)
    axis.set_xlabel("Outer folds in which feature was selected")
    axis.set_ylabel("Feature")
    axis.set_title(title)
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_roc_and_precision_recall(
    predictions: pd.DataFrame,
    roc_path: Path,
    precision_recall_path: Path,
    title: str,
) -> None:
    y_true = predictions["true_severity_class_id"].to_numpy(dtype=int)
    probabilities = predictions[
        ["probability_low", "probability_moderate", "probability_high"]
    ].to_numpy(dtype=float)
    binary_truth = label_binarize(y_true, classes=CLASS_IDS)
    colors = tuple(CLASS_PLOT_COLORS[class_id] for class_id in CLASS_IDS)

    figure, axis = plt.subplots(figsize=(7, 6))
    
    for index, class_id in enumerate(CLASS_IDS):
        false_positive, true_positive, _ = roc_curve(binary_truth[:, index], probabilities[:, index]) # type: ignore
        auc = roc_auc_score(binary_truth[:, index], probabilities[:, index]) # type: ignore
        axis.plot(false_positive, true_positive, color=colors[index], label=f"{CLASS_SHORT_NAMES[class_id]} (AUC={auc:.3f})")
    
    axis.plot([0, 1], [0, 1], linestyle="--", color=GREY_DARK)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title(f"{title}: one-vs-rest ROC")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    for index, class_id in enumerate(CLASS_IDS):
        precision, recall, _ = precision_recall_curve(binary_truth[:, index], probabilities[:, index]) # type: ignore
        average_precision = average_precision_score(binary_truth[:, index], probabilities[:, index]) # type: ignore
        
        axis.plot(recall, precision, color=colors[index], label=f"{CLASS_SHORT_NAMES[class_id]} (AP={average_precision:.3f})")
    
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"{title}: one-vs-rest precision-recall")
    axis.legend(loc="lower left")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(precision_recall_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_model_benchmark_outputs(
    analysis_name: str,
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    selected_features: pd.DataFrame,
    output_dir: Path,
    config: ValidationConfig,
) -> Dict[str, object]:
    """Save unbiased outer-LOSO results for every classifier family."""
    benchmark_dir = (
        output_dir / safe_name(analysis_name) / "all_model_comparison"
    )
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        benchmark_dir / "outer_loso_predictions_all_models.csv", index=False
    )
    selections.to_csv(
        benchmark_dir / "within_model_inner_selections.csv", index=False
    )
    selected_features.to_csv(
        benchmark_dir / "selected_features_all_models.csv", index=False
    )

    overall_frames: List[pd.DataFrame] = []
    per_class_frames: List[pd.DataFrame] = []
    probability_frames: List[pd.DataFrame] = []
    model_summaries: List[Dict[str, object]] = []

    available_models = [
        model
        for model in CLASSIFICATION_MODEL_NAMES
        if model in set(predictions["model"])
    ]
    for model_name in available_models:
        model_predictions = predictions.loc[
            predictions["model"].eq(model_name)
        ].copy()
        model_selections = selections.loc[
            selections["model"].eq(model_name)
        ].copy()
        model_features = selected_features.loc[
            selected_features["model"].eq(model_name)
        ].copy()
        model_dir = benchmark_dir / safe_name(model_name)
        model_dir.mkdir(parents=True, exist_ok=True)

        overall, per_class, matrix = build_metric_tables(
            model_predictions, config
        )
        probability_metrics = calculate_probability_metrics(model_predictions)
        overall.insert(0, "model", model_name)
        per_class.insert(0, "model", model_name)
        probability_metrics.insert(0, "model", model_name)
        overall_frames.append(overall)
        per_class_frames.append(per_class)
        probability_frames.append(probability_metrics)

        model_predictions.to_csv(
            model_dir / "outer_loso_predictions.csv", index=False
        )
        model_selections.to_csv(
            model_dir / "outer_fold_inner_selections.csv", index=False
        )
        model_features.to_csv(
            model_dir / "selected_features_by_outer_fold.csv", index=False
        )
        overall.drop(columns="model").to_csv(
            model_dir / "overall_metrics.csv", index=False
        )
        per_class.drop(columns="model").to_csv(
            model_dir / "per_class_metrics.csv", index=False
        )
        probability_metrics.drop(columns="model").to_csv(
            model_dir / "probability_metrics.csv", index=False
        )
        pd.DataFrame(
            matrix,
            index=[f"true_{CLASS_SHORT_NAMES[value]}" for value in CLASS_IDS],
            columns=[
                f"predicted_{CLASS_SHORT_NAMES[value]}" for value in CLASS_IDS
            ],
        ).to_csv(model_dir / "confusion_matrix.csv")

        plot_overall_metrics(
            overall.drop(columns="model"),
            model_dir / "overall_metrics.png",
            f"{analysis_name}: {model_name}",
        )
        plot_confusion_matrices(
            matrix,
            model_dir / "confusion_matrix.png",
            f"{analysis_name}: {model_name}",
        )
        plot_per_class_metrics(
            per_class.drop(columns="model"),
            model_dir / "per_class_recall_and_f1.png",
            f"{analysis_name}: {model_name}",
        )
        plot_probability_heatmap(
            model_predictions,
            model_dir / "prediction_probabilities.png",
            f"{analysis_name}: {model_name}",
        )
        plot_roc_and_precision_recall(
            model_predictions,
            model_dir / "one_vs_rest_roc.png",
            model_dir / "one_vs_rest_precision_recall.png",
            f"{analysis_name}: {model_name}",
        )
        if not model_features.empty:
            plot_feature_selection_frequency(
                model_features,
                model_dir / "feature_selection_frequency.png",
                f"{analysis_name}: {model_name} selected predictors",
            )

        metric_lookup = overall.set_index("metric")["value"]
        model_summaries.append(
            {
                "model": model_name,
                "participant_count": int(len(model_predictions)),
                "accuracy": float(metric_lookup["accuracy"]),
                "balanced_accuracy": float(
                    metric_lookup["balanced_accuracy"]
                ),
                "macro_f1": float(metric_lookup["macro_f1"]),
                "output_directory": str(model_dir),
            }
        )

    overall_all = pd.concat(overall_frames, ignore_index=True)
    per_class_all = pd.concat(per_class_frames, ignore_index=True)
    probability_all = pd.concat(probability_frames, ignore_index=True)
    overall_all.to_csv(
        benchmark_dir / "overall_metrics_all_models.csv", index=False
    )
    per_class_all.to_csv(
        benchmark_dir / "per_class_metrics_all_models.csv", index=False
    )
    probability_all.to_csv(
        benchmark_dir / "probability_metrics_all_models.csv", index=False
    )

    ranking = (
        overall_all.pivot(index="model", columns="metric", values="value")
        .reset_index()
        .sort_values(
            ["macro_f1", "balanced_accuracy", "accuracy"],
            ascending=False,
        )
    )
    ranking.insert(0, "rank_by_outer_loso_macro_f1", range(1, len(ranking) + 1))
    ranking.to_csv(benchmark_dir / "model_ranking.csv", index=False)

    plot_order = ranking["model"].tolist()
    figure_height = max(6.0, 0.55 * len(plot_order))
    figure, axis = plt.subplots(figsize=(11, figure_height))
    sns.barplot(
        data=overall_all,
        y="model",
        x="value",
        hue="metric",
        order=plot_order,
        hue_order=["accuracy", "balanced_accuracy", "macro_f1"],
        palette={
            "accuracy": BLUE_DARK,
            "balanced_accuracy": BLUE_MEDIUM,
            "macro_f1": BLUE_LIGHT,
        },
        ax=axis,
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Pooled outer-LOSO score")
    axis.set_ylabel("Classifier")
    axis.set_title(f"{analysis_name}: unbiased classifier comparison")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(title="Metric", loc="lower right")
    figure.tight_layout()
    figure.savefig(
        benchmark_dir / "all_model_outer_loso_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    sensor_counts = (
        selections.groupby(["model", "sensor_set"])
        .size()
        .unstack(fill_value=0)
        .reindex(available_models)
    )
    sensor_counts.to_csv(
        benchmark_dir / "sensor_selection_frequency_by_model.csv"
    )
    figure, axis = plt.subplots(figsize=(12, figure_height))
    sensor_counts.plot(
        kind="barh",
        stacked=True,
        color=[BLUE_LIGHT, BLUE_MEDIUM, BLUE_DARK],
        ax=axis,
    )
    axis.set_xlabel("Outer folds")
    axis.set_ylabel("Classifier")
    axis.set_title(f"{analysis_name}: sensor-set selections by classifier")
    axis.grid(axis="x", alpha=0.25)
    axis.legend(title="Sensor set")
    figure.tight_layout()
    figure.savefig(
        benchmark_dir / "sensor_selection_frequency_by_model.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    summary = {
        "analysis": analysis_name,
        "comparison_type": (
            "Each classifier receives one held-out outer-LOSO prediction per "
            "participant; sensor set, feature count and hyperparameters are "
            "selected independently inside each outer-training fold."
        ),
        "models_evaluated": available_models,
        "ranking_metric": "pooled outer-LOSO macro-F1",
        "best_model": str(ranking.iloc[0]["model"]),
        "model_results": model_summaries,
        "output_directory": str(benchmark_dir),
    }
    (benchmark_dir / "model_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    return summary


def save_analysis_outputs(
    analysis_name: str,
    predictions: pd.DataFrame,
    selections: pd.DataFrame,
    candidates: pd.DataFrame,
    selected_features: pd.DataFrame,
    output_dir: Path,
    config: ValidationConfig,
) -> Dict[str, object]:
    analysis_dir = output_dir / safe_name(analysis_name)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    overall, per_class, matrix = build_metric_tables(predictions, config)
    probability_metrics = calculate_probability_metrics(predictions)
    predictions.to_csv(analysis_dir / "outer_loso_predictions.csv", index=False)
    overall.to_csv(analysis_dir / "overall_metrics.csv", index=False)
    per_class.to_csv(analysis_dir / "per_class_metrics.csv", index=False)
    probability_metrics.to_csv(
        analysis_dir / "probability_metrics.csv",
        index=False,
    )
    pd.DataFrame(
        matrix,
        index=[f"true_{CLASS_SHORT_NAMES[value]}" for value in CLASS_IDS],
        columns=[f"predicted_{CLASS_SHORT_NAMES[value]}" for value in CLASS_IDS],
    ).to_csv(analysis_dir / "confusion_matrix.csv")
    selections.to_csv(analysis_dir / "outer_fold_selections.csv", index=False)
    candidates.to_csv(analysis_dir / "inner_candidate_summary.csv", index=False)
    selected_features.to_csv(
        analysis_dir / "selected_features_by_outer_fold.csv",
        index=False,
    )

    plot_overall_metrics(
        overall,
        analysis_dir / "overall_metrics.png",
        f"{analysis_name}: pooled outer-LOSO performance",
    )
    plot_confusion_matrices(
        matrix,
        analysis_dir / "confusion_matrix.png",
        f"{analysis_name}: confusion matrix",
    )
    plot_per_class_metrics(
        per_class,
        analysis_dir / "per_class_recall_and_f1.png",
        f"{analysis_name}: per-class recall and F1",
    )
    plot_probability_heatmap(
        predictions,
        analysis_dir / "prediction_probabilities.png",
        f"{analysis_name}: outer-LOSO probabilities",
    )
    plot_selection_frequencies(
        selections,
        analysis_dir / "model_and_sensor_selection_frequencies.png",
        f"{analysis_name}: inner-selection frequencies",
    )
    plot_feature_selection_frequency(
        selected_features,
        analysis_dir / "feature_selection_frequency.png",
        f"{analysis_name}: most frequently selected predictors",
    )
    plot_roc_and_precision_recall(
        predictions,
        analysis_dir / "one_vs_rest_roc.png",
        analysis_dir / "one_vs_rest_precision_recall.png",
        analysis_name,
    )

    summary = {
        "analysis": analysis_name,
        "participant_count": int(len(predictions)),
        "metrics": {
            row.metric: {
                "value": float(row.value), # type: ignore 
                "ci_95_lower": float(row.ci_95_lower), # type: ignore
                "ci_95_upper": float(row.ci_95_upper), # type: ignore
            }
            for row in overall.itertuples(index=False)
        },
        "class_counts": {
            str(key): int(value)
            for key, value in predictions[
                "true_severity_class_id"
            ].value_counts().sort_index().items()
        },
        "output_directory": str(analysis_dir),
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default),
        encoding="utf-8",
    )
    return summary


def save_cross_analysis_comparison(
    summaries: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for summary in summaries:
        for metric, values in summary["metrics"].items(): # type: ignore
            rows.append(
                {
                    "analysis": summary["analysis"],
                    "participant_count": summary["participant_count"],
                    "metric": metric,
                    **values,
                }
            )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "analysis_comparison.csv", index=False)

    figure, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=comparison,
        x="analysis",
        y="value",
        hue="metric",
        palette={
            "accuracy": BLUE_DARK,
            "balanced_accuracy": BLUE_MEDIUM,
            "macro_f1": GREY_MEDIUM,
        },
        ax=axis,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Analysis")
    axis.set_ylabel("Pooled outer-LOSO score")
    axis.set_title("Primary and sensitivity analysis comparison")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Metric")
    figure.tight_layout()
    figure.savefig(
        output_dir / "analysis_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_matched_modality_comparison(
    summaries: Sequence[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Compare modalities only when participant membership is identical."""
    display_names = {
        "matched_imu_only_three_class": "IMU only",
        "matched_emg_only_three_class": "EMG only",
        "matched_imu_emg_three_class": "IMU + EMG",
        "sensitivity_matched_imu_emg_with_legacy_imu": (
            "IMU + EMG + legacy IMU"
        ),
    }
    rows: List[Dict[str, object]] = []
    participant_counts: set[int] = set()
    for summary in summaries:
        analysis = str(summary["analysis"])
        if analysis not in display_names:
            continue
        participant_count = int(summary["participant_count"])
        participant_counts.add(participant_count)
        for metric, values in summary["metrics"].items():  # type: ignore
            rows.append(
                {
                    "analysis": analysis,
                    "modality": display_names[analysis],
                    "participant_count": participant_count,
                    "metric": metric,
                    **values,
                }
            )
    if not rows:
        return
    if len(participant_counts) != 1:
        raise AssertionError(
            "Matched modality analyses do not use identical participant counts."
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        output_dir / "matched_modality_comparison.csv",
        index=False,
    )
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=comparison,
        x="modality",
        y="value",
        hue="metric",
        palette={
            "accuracy": BLUE_DARK,
            "balanced_accuracy": BLUE_MEDIUM,
            "macro_f1": GREY_MEDIUM,
        },
        ax=axis,
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Predictor modality")
    axis.set_ylabel("Pooled outer-LOSO score")
    axis.set_title(
        "Matched-participant modality comparison "
        f"(n={next(iter(participant_counts))})"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Metric")
    figure.tight_layout()
    figure.savefig(
        output_dir / "matched_modality_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# =============================================================================
# Main entry point
# =============================================================================


def main() -> int:
    prepare_output_directory(OUTPUT_DIR, OVERWRITE)
    score_definition = validate_score_definition(SCORE_DEFINITION_PATH)
    labels = load_frozen_labels(LABELS_PATH, score_definition)
    snapshot_hash = save_frozen_label_snapshot(
        labels,
        score_definition,
        OUTPUT_DIR,
    )

    print("Frozen label counts")
    print(labels["severity_class"].value_counts().to_string())
    borderline_keys = labels.loc[
        labels["stroke_split_borderline"], "participant_key"
    ].tolist()
    print(f"Predeclared borderline participants: {borderline_keys}")

    cycle_intervals = load_frozen_cycle_intervals(CYCLE_METRICS_PATH, labels)

    features, predictor_qc, feature_manifest = build_predictor_dataset(
        labels,
        DATA_DIR,
        cycle_intervals,
        SIGNAL_CONFIG,
    )
    external_features, external_feature_manifest = (
        load_mapped_external_features(MAPPED_EXTERNAL_FEATURES_PATH)
    )
    matched_labels, emg_mapping_qc = create_matched_emg_cohort(
        labels,
        external_features,
    )
    matched_imu_features = subset_predictors_to_labels(
        features,
        matched_labels,
    )
    matched_external_features = subset_predictors_to_labels(
        external_features,
        matched_labels,
    )
    emg_only_features = select_external_modality(
        matched_external_features,
        include_legacy_imu=False,
    )
    imu_emg_features = combine_predictor_frames(
        matched_imu_features,
        emg_only_features,
    )
    all_external_features = select_external_modality(
        matched_external_features,
        include_legacy_imu=True,
    )
    imu_emg_legacy_features = combine_predictor_frames(
        matched_imu_features,
        all_external_features,
    )

    features.to_csv(OUTPUT_DIR / "eligible_imu_predictors.csv", index=False)
    external_features.to_csv(
        OUTPUT_DIR / "mapped_external_predictors_used.csv",
        index=False,
    )
    matched_labels.to_csv(
        OUTPUT_DIR / "matched_emg_frozen_labels.csv",
        index=False,
    )
    emg_mapping_qc.to_csv(
        OUTPUT_DIR / "emg_mapping_quality_control.csv",
        index=False,
    )
    matched_imu_features.to_csv(
        OUTPUT_DIR / "matched_imu_predictors.csv",
        index=False,
    )
    emg_only_features.to_csv(
        OUTPUT_DIR / "matched_emg_predictors.csv",
        index=False,
    )
    imu_emg_features.to_csv(
        OUTPUT_DIR / "matched_imu_emg_predictors.csv",
        index=False,
    )
    imu_emg_legacy_features.to_csv(
        OUTPUT_DIR / "matched_imu_emg_legacy_predictors.csv",
        index=False,
    )
    predictor_qc.to_csv(OUTPUT_DIR / "predictor_extraction_qc.csv", index=False)
    combined_feature_manifest = pd.concat(
        [feature_manifest, external_feature_manifest],
        ignore_index=True,
    )
    combined_feature_manifest.to_csv(
        OUTPUT_DIR / "feature_manifest.csv",
        index=False,
    )

    all_feature_columns = [
        column
        for column in features.columns
        if column not in {"participant_key", "cohort", "patient_folder"}
    ]
    assert_eligible_feature_names(all_feature_columns)

    protocol = {
        "label_status": "Frozen before model testing",
        "score_definition_version": score_definition["score_definition_version"],
        "score_definition_sha256": score_definition["sha256"],
        "frozen_label_assignment_sha256": EXPECTED_FROZEN_LABEL_ASSIGNMENT_SHA256,
        "frozen_label_snapshot_sha256": snapshot_hash,
        "primary_validation": "Outer LOSO with inner stratified grouped CV",
        "primary_selection_metric": VALIDATION_CONFIG.primary_metric,
        "outer_evaluation": "Metrics from pooled held-out participant predictions",
        "sensor_sets_selected_in_inner_cv": [
            "accelerometer",
            "gyroscope_xy",
            "combined",
        ],
        "multimodal_experiment_sensor_sets": {
            "matched_imu_only": list(RAW_IMU_SENSOR_SETS),
            "emg_only": list(EMG_ONLY_SENSOR_SETS),
            "imu_plus_emg": list(IMU_EMG_SENSOR_SETS),
            "imu_plus_emg_plus_legacy_imu_sensitivity": list(
                IMU_EMG_LEGACY_SENSOR_SETS
            ),
        },
        "models_selected_in_inner_cv": list(CLASSIFICATION_MODEL_NAMES),
        "all_model_evaluation": (
            "Every classifier family receives pooled held-out outer-LOSO "
            "predictions. Within each model, sensor set, feature count and "
            "hyperparameters are selected using inner grouped CV only."
        ),
        "feature_selection": (
            "Classification ANOVA and regression f_regression SelectKBest, "
            "each fitted only inside its inner-training fold."
        ),
        "leakage_barrier": {
            "gyroscope_z_predictors": "excluded",
            "gyroscope_z_cycle_boundaries": (
                "Frozen start/end timestamps used only for segmentation; "
                "duration, amplitude, count and polarity are excluded."
            ),
            "label_components": "excluded",
            "continuous_sgas_score": "excluded",
            "rank_confidence_borderline_fields": "excluded",
            "cohort_as_predictor": "excluded",
            "mapped_source_id_and_label": "excluded",
            "mapped_legacy_gyroscope_z_predictors": "excluded upstream",
        },
        "primary_predictors": (
            "Cycle-aligned accelerometer-magnitude and gyroscope-x/y-magnitude "
            "features aggregated into side-invariant bilateral means, absolute "
            "differences and symmetric differences; waveform predictors excluded."
        ),
        "sensitivity_analyses": [
            "Exclude predeclared borderline participants",
            "Add eligible cycle-normalized waveform predictors",
            "Hierarchical Healthy-vs-Stroke then Moderate-vs-High",
            "Matched-participant IMU-only baseline",
            "EMG-only classification on matched participants",
            "Cycle-aligned IMU plus EMG on matched participants",
            "Optional mapped legacy non-z IMU feature sensitivity",
        ],
        "emg_feature_design": (
            "Previously extracted six-muscle EMG features are represented as "
            "order-invariant bilateral means and absolute bilateral "
            "differences. They are joined only after SGAS-v2 labels are frozen."
        ),
        "fair_modality_comparison": {
            "matched_participant_count": int(len(matched_labels)),
            "matched_class_counts": {
                str(key): int(value)
                for key, value in matched_labels[
                    "severity_class_id"
                ].value_counts().sort_index().items()
            },
            "rule": (
                "Matched IMU-only, EMG-only and IMU+EMG analyses use exactly "
                "the same participants and frozen outer LOSO folds."
            ),
        },
        "continuous_score_regression": (
            "Nested LOSO Ridge/Linear-SVR prediction of the frozen SGAS-v2 "
            "continuous score, reporting MAE and Spearman correlation."
        ),
        "interpretation": (
            "Classification of frozen relative sensor-derived gait-asymmetry "
            "severity labels; not clinical stroke-severity diagnosis."
        ),
        "signal_configuration": asdict(SIGNAL_CONFIG),
        "validation_configuration": asdict(VALIDATION_CONFIG),
    }
    (OUTPUT_DIR / "classification_protocol.json").write_text(
        json.dumps(protocol, indent=2, default=json_default),
        encoding="utf-8",
    )

    summaries: List[Dict[str, object]] = []
    if RUN_PRIMARY_ANALYSIS:
        summaries.append(
            run_direct_nested_loso(
                "primary_direct_three_class",
                labels,
                features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
            )
        )

    if RUN_MATCHED_IMU_BASELINE:
        summaries.append(
            run_direct_nested_loso(
                "matched_imu_only_three_class",
                matched_labels,
                matched_imu_features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
                candidate_sensor_sets=RAW_IMU_SENSOR_SETS,
            )
        )

    if RUN_EMG_ONLY_ANALYSIS:
        summaries.append(
            run_direct_nested_loso(
                "matched_emg_only_three_class",
                matched_labels,
                emg_only_features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
                candidate_sensor_sets=EMG_ONLY_SENSOR_SETS,
            )
        )

    if RUN_IMU_EMG_ANALYSIS:
        summaries.append(
            run_direct_nested_loso(
                "matched_imu_emg_three_class",
                matched_labels,
                imu_emg_features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
                candidate_sensor_sets=IMU_EMG_SENSOR_SETS,
            )
        )

    if RUN_LEGACY_IMU_FEATURE_SENSITIVITY:
        summaries.append(
            run_direct_nested_loso(
                "sensitivity_matched_imu_emg_with_legacy_imu",
                matched_labels,
                imu_emg_legacy_features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
                candidate_sensor_sets=IMU_EMG_LEGACY_SENSOR_SETS,
            )
        )

    if RUN_BORDERLINE_EXCLUSION_SENSITIVITY:
        non_borderline = labels.loc[~labels["stroke_split_borderline"]].copy()
        removed = sorted(
            set(labels["participant_key"]) - set(non_borderline["participant_key"]),
            key=natural_sort_key,
        )
        if not removed:
            raise ValueError(
                "Borderline sensitivity was enabled but no frozen borderline "
                "participants were identified."
            )
        print(f"Borderline sensitivity excludes only: {removed}")
        summaries.append(
            run_direct_nested_loso(
                "sensitivity_excluding_borderline",
                non_borderline,
                features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
            )
        )

    if RUN_WAVEFORM_SENSITIVITY:
        summaries.append(
            run_direct_nested_loso(
                "sensitivity_with_eligible_waveform_predictors",
                labels,
                features,
                include_waveform=True,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
            )
        )

    if RUN_HIERARCHICAL_SENSITIVITY:
        summaries.append(
            run_hierarchical_nested_loso(
                "exploratory_hierarchical_classifier",
                labels,
                features,
                include_waveform=False,
                output_dir=OUTPUT_DIR,
                config=VALIDATION_CONFIG,
            )
        )

    regression_summary: Dict[str, object] | None = None
    if RUN_CONTINUOUS_SCORE_REGRESSION:
        regression_summary = run_continuous_score_nested_loso(
            labels,
            features,
            OUTPUT_DIR,
            VALIDATION_CONFIG,
        )

    if not summaries and regression_summary is None:
        raise ValueError("At least one analysis must be enabled.")
    if summaries:
        save_cross_analysis_comparison(summaries, OUTPUT_DIR)
        save_matched_modality_comparison(summaries, OUTPUT_DIR)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(
            {
                "analyses": summaries,
                "continuous_score_regression": regression_summary,
                "borderline_participants": borderline_keys,
                "eligible_predictor_count": len(all_feature_columns),
                "waveform_predictor_count": int(
                    sum("__wave_" in column for column in all_feature_columns)
                ),
                "non_waveform_predictor_count": int(
                    sum("__wave_" not in column for column in all_feature_columns)
                ),
                "mapped_emg_predictor_count": int(
                    sum(
                        column.startswith("emg__")
                        for column in external_features.columns
                    )
                ),
                "mapped_legacy_imu_predictor_count": int(
                    sum(
                        column.startswith("legacy_imu__")
                        for column in external_features.columns
                    )
                ),
                "emg_complete_case_participant_count": int(len(matched_labels)),
                "emg_mapping_status_counts": {
                    str(key): int(value)
                    for key, value in emg_mapping_qc[
                        "mapping_status"
                    ].value_counts().items()
                },
            },
            indent=2,
            default=json_default,
        ),
        encoding="utf-8",
    )

    print(f"\nCompleted {len(summaries)} classification analyses.")
    if regression_summary is not None:
        print("Completed continuous SGAS-v2 score regression analysis.")
    print(f"Results saved under: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
