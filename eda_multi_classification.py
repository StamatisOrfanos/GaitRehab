from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif, RFE
from sklearn.linear_model import LogisticRegression


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path("data.csv")
OUTPUT_DIR = Path("outputs/eda_plots")

ID_COLUMN = "ID"
LABEL_COLUMN = "Label"

CLASS_NAMES = {
    0: "Healthy leg",
    1: "Affected side",
    2: "Non-affected side",
}

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

# Number of features to select with RFE.
# The script automatically skips values larger than the available number of features.
RFE_FEATURE_COUNTS = [5, 10, 15, 20, 30, 40, 50]

# Top features to export from ANOVA and combined ranking.
TOP_N_ANOVA_EXPORT = 50
TOP_N_COMBINED_EXPORT = 50


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

CLASS_COLORS = {
    0: "#2E7D32",
    1: "#C62828",
    2: "#1565C0",
}


# =============================================================================
# Utility functions
# =============================================================================

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "top_features_by_source").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_selection").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_selection" / "rfe_selected_features").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "feature_selection" / "plots").mkdir(parents=True, exist_ok=True)


def safe_filename(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[+]", "plus", name)
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Put this script in the same folder as data.csv "
            f"or change DATA_PATH."
        )

    df = pd.read_csv(path, sep=None, engine="python")

    # Remove fully empty columns and common Excel-export blank columns.
    df = df.dropna(axis=1, how="all")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    # Clean column names.
    df.columns = [str(col).strip() for col in df.columns]

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Expected label column '{LABEL_COLUMN}', but found columns:\n{df.columns.tolist()}"
        )

    if ID_COLUMN not in df.columns:
        warnings.warn(
            f"Expected ID column '{ID_COLUMN}' was not found. "
            f"The script will continue without subject/sample ID plots."
        )

    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce").astype("Int64")

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


def get_all_sensor_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    return {
        source_name: get_feature_columns(df, prefixes)
        for source_name, prefixes in SENSOR_PREFIXES.items()
    }


def clean_feature_matrix(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    subset = df[feature_columns + [LABEL_COLUMN]].copy()
    subset = subset.dropna(subset=[LABEL_COLUMN])

    y = subset[LABEL_COLUMN].astype(int)

    X = subset[feature_columns].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.fillna(X.median(numeric_only=True))

    # Drop constant columns.
    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]

    return X, y


def standardize_matrix(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X)


def save_current_figure(filename: str) -> None:
    output_path = OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_feature_selection_figure(filename: str) -> None:
    output_path = OUTPUT_DIR / "feature_selection" / "plots" / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


# =============================================================================
# General EDA plots
# =============================================================================

def plot_class_distribution(df: pd.DataFrame) -> None:
    counts = df[LABEL_COLUMN].value_counts().sort_index()

    labels = [CLASS_NAMES.get(int(label), str(label)) for label in counts.index]
    colors = [CLASS_COLORS.get(int(label), "#666666") for label in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="black", linewidth=0.8)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(int(height)),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title("Class distribution")
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Class")
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)

    save_current_figure("00_class_distribution.png")


def plot_feature_counts(sensor_columns: Dict[str, List[str]]) -> None:
    counts = {source: len(cols) for source, cols in sensor_columns.items()}

    fig, ax = plt.subplots(figsize=(11, 6))
    source_names = list(counts.keys())
    values = list(counts.values())

    bars = ax.barh(source_names, values, edgecolor="black", linewidth=0.8)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            str(int(width)),
            va="center",
            fontweight="bold",
        )

    ax.set_title("Number of features per sensor combination")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Feature source")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    save_current_figure("01_feature_counts_by_source.png")


def plot_missing_values_by_source(
    df: pd.DataFrame,
    sensor_columns: Dict[str, List[str]],
) -> None:
    missing_percentages = {}

    for source, cols in sensor_columns.items():
        if not cols:
            missing_percentages[source] = 0.0
            continue

        missing_percentages[source] = df[cols].isna().mean().mean() * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    source_names = list(missing_percentages.keys())
    values = list(missing_percentages.values())

    bars = ax.barh(source_names, values, edgecolor="black", linewidth=0.8)

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.2f}%",
            va="center",
            fontweight="bold",
        )

    ax.set_title("Missing values by sensor source")
    ax.set_xlabel("Average missing values (%)")
    ax.set_ylabel("Feature source")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    save_current_figure("02_missing_values_by_source.png")


# =============================================================================
# Source-specific EDA plots
# =============================================================================

def plot_pca_scatter(
    df: pd.DataFrame,
    source_name: str,
    feature_columns: List[str],
) -> None:
    X, y = clean_feature_matrix(df, feature_columns)

    if X.shape[1] < 2:
        print(f"[SKIP] PCA for {source_name}: fewer than 2 usable features.")
        return

    X_scaled = standardize_matrix(X)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    for label_id, class_name in CLASS_NAMES.items():
        mask = y.values == label_id
        if mask.sum() == 0:
            continue

        ax.scatter(
            components[mask, 0],
            components[mask, 1],
            label=class_name,
            s=70,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.6,
            color=CLASS_COLORS.get(label_id),
        )

    ax.axhline(0, color="black", linewidth=0.6, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.35)

    ax.set_title(f"PCA projection: {source_name}")
    ax.set_xlabel(f"PC1 ({explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1]:.1f}% variance)")
    ax.legend(frameon=True)
    ax.grid(alpha=0.25)

    filename = f"pca_{safe_filename(source_name)}.png"
    save_current_figure(filename)


def compute_anova_scores(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    X, y = clean_feature_matrix(df, feature_columns)

    if X.shape[1] == 0:
        return pd.DataFrame(columns=["feature", "anova_f_score", "p_value", "anova_rank"])

    X_scaled = standardize_matrix(X)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores, p_values = f_classif(X_scaled, y)

    ranking = pd.DataFrame(
        {
            "feature": X.columns,
            "anova_f_score": scores,
            "p_value": p_values,
        }
    )

    ranking = ranking.replace([np.inf, -np.inf], np.nan)
    ranking = ranking.dropna(subset=["anova_f_score"])
    ranking = ranking.sort_values("anova_f_score", ascending=False).reset_index(drop=True)
    ranking["anova_rank"] = np.arange(1, len(ranking) + 1)

    return ranking


def plot_top_anova_features(
    df: pd.DataFrame,
    source_name: str,
    feature_columns: List[str],
    top_n: int = 20,
) -> pd.DataFrame:
    ranking = compute_anova_scores(df, feature_columns)

    if ranking.empty:
        print(f"[SKIP] ANOVA feature ranking for {source_name}: no usable features.")
        return ranking

    top_features = ranking.head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(11, 8))

    bars = ax.barh(
        top_features["feature"],
        top_features["anova_f_score"],
        edgecolor="black",
        linewidth=0.7,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + max(top_features["anova_f_score"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            fontsize=9,
        )

    ax.set_title(f"Top {min(top_n, len(ranking))} class-separating features: {source_name}")
    ax.set_xlabel("ANOVA F-score")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    filename = f"top_anova_features_{safe_filename(source_name)}.png"
    save_current_figure(filename)

    return ranking


def plot_class_mean_heatmap(
    df: pd.DataFrame,
    source_name: str,
    feature_columns: List[str],
    ranking: pd.DataFrame,
    top_n: int = 20,
) -> None:
    if ranking.empty:
        print(f"[SKIP] Class mean heatmap for {source_name}: empty ranking.")
        return

    selected_features = ranking.head(top_n)["feature"].tolist()
    X, y = clean_feature_matrix(df, selected_features)

    if X.shape[1] == 0:
        print(f"[SKIP] Class mean heatmap for {source_name}: no usable selected features.")
        return

    X_scaled = pd.DataFrame(
        standardize_matrix(X),
        columns=X.columns,
        index=X.index,
    )

    plot_df = X_scaled.copy()
    plot_df[LABEL_COLUMN] = y.values

    class_means = plot_df.groupby(LABEL_COLUMN)[X.columns].mean()
    class_means.index = [CLASS_NAMES.get(int(idx), str(idx)) for idx in class_means.index]

    fig, ax = plt.subplots(figsize=(14, 5.5))

    im = ax.imshow(class_means.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)

    ax.set_title(f"Standardized class means of top features: {source_name}")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Class")

    ax.set_xticks(np.arange(len(class_means.columns)))
    ax.set_xticklabels(class_means.columns, rotation=60, ha="right")

    ax.set_yticks(np.arange(len(class_means.index)))
    ax.set_yticklabels(class_means.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean z-score")

    ax.grid(False)

    filename = f"class_mean_heatmap_{safe_filename(source_name)}.png"
    save_current_figure(filename)


def plot_top_feature_boxplots(
    df: pd.DataFrame,
    source_name: str,
    ranking: pd.DataFrame,
    top_n: int = 6,
) -> None:
    if ranking.empty:
        print(f"[SKIP] Boxplots for {source_name}: empty ranking.")
        return

    selected_features = ranking.head(top_n)["feature"].tolist()
    X, y = clean_feature_matrix(df, selected_features)

    if X.shape[1] == 0:
        print(f"[SKIP] Boxplots for {source_name}: no usable selected features.")
        return

    X_scaled = pd.DataFrame(
        standardize_matrix(X),
        columns=X.columns,
        index=X.index,
    )

    plot_df = X_scaled.copy()
    plot_df["Class"] = y.map(CLASS_NAMES).values

    n_features = len(selected_features)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(14, 4.2 * n_rows),
        squeeze=False,
    )

    class_order = [CLASS_NAMES[0], CLASS_NAMES[1], CLASS_NAMES[2]]

    for idx, feature in enumerate(selected_features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        data_by_class = [
            plot_df.loc[plot_df["Class"] == class_name, feature].dropna().values
            for class_name in class_order
        ]

        box = ax.boxplot(
            data_by_class,
            labels=class_order,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.5},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )

        for patch, class_id in zip(box["boxes"], [0, 1, 2]):
            patch.set_facecolor(CLASS_COLORS[class_id])
            patch.set_alpha(0.55)

        ax.set_title(feature)
        ax.set_ylabel("Standardized value")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)

    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row][col])

    fig.suptitle(f"Top feature distributions by class: {source_name}", y=1.02, fontsize=16)

    filename = f"boxplots_top_features_{safe_filename(source_name)}.png"
    save_current_figure(filename)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    source_name: str,
    feature_columns: List[str],
    max_features: int = 25,
) -> None:
    X, _ = clean_feature_matrix(df, feature_columns)

    if X.shape[1] < 2:
        print(f"[SKIP] Correlation heatmap for {source_name}: fewer than 2 usable features.")
        return

    variances = X.var(axis=0).sort_values(ascending=False)
    selected_features = variances.head(max_features).index.tolist()

    corr = X[selected_features].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_title(f"Spearman correlation heatmap: {source_name}")
    ax.set_xticks(np.arange(len(selected_features)))
    ax.set_yticks(np.arange(len(selected_features)))

    ax.set_xticklabels(selected_features, rotation=60, ha="right")
    ax.set_yticklabels(selected_features)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman correlation")

    ax.grid(False)

    filename = f"correlation_heatmap_{safe_filename(source_name)}.png"
    save_current_figure(filename)


def plot_block_level_importance(
    df: pd.DataFrame,
    source_name: str,
    feature_columns: List[str],
    prefixes: List[str],
) -> None:
    ranking = compute_anova_scores(df, feature_columns)

    if ranking.empty:
        print(f"[SKIP] Block-level importance for {source_name}: empty ranking.")
        return

    rows = []

    for prefix in prefixes:
        prefix_scores = ranking.loc[
            ranking["feature"].str.startswith(prefix + "_"),
            "anova_f_score",
        ]

        if len(prefix_scores) == 0:
            continue

        rows.append(
            {
                "block": prefix,
                "mean_anova_f_score": prefix_scores.mean(),
                "median_anova_f_score": prefix_scores.median(),
                "n_features": len(prefix_scores),
            }
        )

    block_df = pd.DataFrame(rows)

    if block_df.empty:
        print(f"[SKIP] Block-level importance for {source_name}: no block data.")
        return

    block_df = block_df.sort_values("mean_anova_f_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(
        block_df["block"],
        block_df["mean_anova_f_score"],
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, n_features in zip(bars, block_df["n_features"]):
        width = bar.get_width()
        ax.text(
            width + block_df["mean_anova_f_score"].max() * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"n={int(n_features)}",
            va="center",
            fontsize=9,
        )

    ax.set_title(f"Average class-separation score by axis/muscle: {source_name}")
    ax.set_xlabel("Mean ANOVA F-score")
    ax.set_ylabel("Axis / muscle")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    filename = f"block_level_anova_{safe_filename(source_name)}.png"
    save_current_figure(filename)


# =============================================================================
# Feature selection: RFE
# =============================================================================

def build_rfe_estimator() -> LogisticRegression:
    """
    Logistic regression is used here because RFE needs a model with coefficients
    or feature importances.

    The liblinear solver handles multiclass through one-vs-rest, which is fine
    for feature ranking in this small tabular dataset.
    """

    return LogisticRegression(
        penalty="l2",
        solver="liblinear",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )


def run_rfe_for_source(
    df: pd.DataFrame,
    source_name: str,
    feature_columns: List[str],
    n_features_to_select: int,
) -> pd.DataFrame:
    X, y = clean_feature_matrix(df, feature_columns)

    if X.shape[1] == 0:
        return pd.DataFrame()

    n_features_to_select = min(n_features_to_select, X.shape[1])

    X_scaled = standardize_matrix(X)

    estimator = build_rfe_estimator()

    rfe = RFE(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        step=1,
    )

    rfe.fit(X_scaled, y)

    selected_mask = rfe.support_
    rfe_ranking = rfe.ranking_

    # Fit the same estimator on the selected features to get coefficient magnitude.
    selected_feature_names = X.columns[selected_mask].tolist()
    selected_X = X_scaled[:, selected_mask]

    selected_estimator = build_rfe_estimator()
    selected_estimator.fit(selected_X, y)

    coefficients = selected_estimator.coef_

    if coefficients.ndim == 1:
        importance = np.abs(coefficients)
    else:
        importance = np.mean(np.abs(coefficients), axis=0)

    importance_map = {
        feature: importance_value
        for feature, importance_value in zip(selected_feature_names, importance)
    }

    result = pd.DataFrame(
        {
            "source": source_name,
            "feature": X.columns,
            "rfe_rank": rfe_ranking,
            "rfe_selected": selected_mask,
            "n_features_selected_target": n_features_to_select,
            "selected_model_abs_coefficient_mean": [
                importance_map.get(feature, np.nan) for feature in X.columns
            ],
        }
    )

    result = result.sort_values(
        by=["rfe_rank", "selected_model_abs_coefficient_mean"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return result


def plot_rfe_selected_features(
    rfe_result: pd.DataFrame,
    source_name: str,
    n_features_to_select: int,
) -> None:
    selected = rfe_result[rfe_result["rfe_selected"]].copy()

    if selected.empty:
        return

    selected = selected.sort_values(
        "selected_model_abs_coefficient_mean",
        ascending=True,
    )

    fig, ax = plt.subplots(figsize=(11, max(5, 0.35 * len(selected))))

    bars = ax.barh(
        selected["feature"],
        selected["selected_model_abs_coefficient_mean"],
        edgecolor="black",
        linewidth=0.7,
    )

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + selected["selected_model_abs_coefficient_mean"].max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.3f}",
            va="center",
            fontsize=9,
        )

    ax.set_title(
        f"RFE-selected features: {source_name}\n"
        f"Top {n_features_to_select} features"
    )
    ax.set_xlabel("Mean absolute logistic-regression coefficient")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    filename = (
        f"rfe_selected_features_{safe_filename(source_name)}_"
        f"top_{n_features_to_select}.png"
    )
    save_feature_selection_figure(filename)


def create_rfe_feature_selection_outputs(
    df: pd.DataFrame,
    sensor_columns: Dict[str, List[str]],
) -> pd.DataFrame:
    """
    Runs RFE for every sensor source and every requested target feature count.

    Exports:
        - one CSV per source and feature-count
        - one global long-format RFE table
    """

    all_rfe_results = []

    output_dir = OUTPUT_DIR / "feature_selection" / "rfe_selected_features"

    print()
    print("=" * 80)
    print("Running RFE feature selection")
    print("=" * 80)

    for source in SOURCE_ORDER:
        columns = sensor_columns[source]

        if len(columns) == 0:
            print(f"[SKIP] RFE for {source}: no features found.")
            continue

        X, _ = clean_feature_matrix(df, columns)
        usable_feature_count = X.shape[1]

        print(f"{source}: {usable_feature_count} usable features")

        valid_feature_counts = [
            count for count in RFE_FEATURE_COUNTS
            if count <= usable_feature_count
        ]

        if usable_feature_count not in valid_feature_counts:
            valid_feature_counts.append(usable_feature_count)

        valid_feature_counts = sorted(set(valid_feature_counts))

        for feature_count in valid_feature_counts:
            print(f"  - RFE selecting top {feature_count} features")

            rfe_result = run_rfe_for_source(
                df=df,
                source_name=source,
                feature_columns=columns,
                n_features_to_select=feature_count,
            )

            if rfe_result.empty:
                continue

            output_path = (
                output_dir /
                f"rfe_{safe_filename(source)}_top_{feature_count}.csv"
            )

            rfe_result.to_csv(output_path, index=False)
            print(f"    [SAVED] {output_path}")

            plot_rfe_selected_features(
                rfe_result=rfe_result,
                source_name=source,
                n_features_to_select=feature_count,
            )

            all_rfe_results.append(rfe_result)

    if not all_rfe_results:
        return pd.DataFrame()

    all_rfe_df = pd.concat(all_rfe_results, ignore_index=True)

    all_rfe_path = OUTPUT_DIR / "feature_selection" / "all_rfe_feature_selection_results.csv"
    all_rfe_df.to_csv(all_rfe_path, index=False)

    print(f"[SAVED] {all_rfe_path}")

    return all_rfe_df


def create_combined_feature_ranking_outputs(
    df: pd.DataFrame,
    sensor_columns: Dict[str, List[str]],
    all_rfe_df: pd.DataFrame,
) -> None:
    """
    Creates combined ANOVA + RFE ranking files.

    The combined score is simple and interpretable:
        combined_rank_score = anova_rank + rfe_rank

    Lower combined_rank_score is better.
    """

    output_dir = OUTPUT_DIR / "feature_selection"

    combined_tables = []

    print()
    print("=" * 80)
    print("Creating combined ANOVA + RFE rankings")
    print("=" * 80)

    if all_rfe_df.empty:
        print("[SKIP] No RFE results available.")
        return

    for source in SOURCE_ORDER:
        columns = sensor_columns[source]

        if len(columns) == 0:
            continue

        anova_df = compute_anova_scores(df, columns)

        if anova_df.empty:
            continue

        # Use the smallest available RFE count for this source as the strict RFE ranking.
        source_rfe = all_rfe_df[all_rfe_df["source"] == source].copy()

        if source_rfe.empty:
            continue

        smallest_target = source_rfe["n_features_selected_target"].min()

        source_rfe = source_rfe[
            source_rfe["n_features_selected_target"] == smallest_target
        ].copy()

        combined = anova_df.merge(
            source_rfe[
                [
                    "source",
                    "feature",
                    "rfe_rank",
                    "rfe_selected",
                    "n_features_selected_target",
                    "selected_model_abs_coefficient_mean",
                ]
            ],
            on="feature",
            how="left",
        )

        combined["source"] = source
        combined["combined_rank_score"] = combined["anova_rank"] + combined["rfe_rank"]
        combined["combined_rank"] = combined["combined_rank_score"].rank(
            method="dense",
            ascending=True,
        ).astype(int)

        combined = combined.sort_values(
            by=[
                "combined_rank",
                "anova_rank",
                "rfe_rank",
                "anova_f_score",
            ],
            ascending=[True, True, True, False],
        ).reset_index(drop=True)

        combined = combined[
            [
                "source",
                "feature",
                "combined_rank",
                "combined_rank_score",
                "anova_rank",
                "anova_f_score",
                "p_value",
                "rfe_rank",
                "rfe_selected",
                "n_features_selected_target",
                "selected_model_abs_coefficient_mean",
            ]
        ]

        source_path = (
            output_dir /
            f"combined_feature_ranking_{safe_filename(source)}.csv"
        )

        combined.to_csv(source_path, index=False)
        print(f"[SAVED] {source_path}")

        top_path = (
            output_dir /
            f"top_{TOP_N_COMBINED_EXPORT}_combined_features_{safe_filename(source)}.csv"
        )

        combined.head(TOP_N_COMBINED_EXPORT).to_csv(top_path, index=False)
        print(f"[SAVED] {top_path}")

        combined_tables.append(combined)

    if combined_tables:
        global_combined = pd.concat(combined_tables, ignore_index=True)

        global_path = output_dir / "all_combined_feature_rankings.csv"
        global_combined.to_csv(global_path, index=False)

        print(f"[SAVED] {global_path}")


def create_final_selected_feature_lists(
    all_rfe_df: pd.DataFrame,
) -> None:
    """
    Creates small, practical selected-feature-list CSVs for later classification.

    Output format:
        source
        n_features
        feature

    This is easy to read from another classification script.
    """

    if all_rfe_df.empty:
        print("[SKIP] No RFE results available for final selected feature lists.")
        return

    selected = all_rfe_df[all_rfe_df["rfe_selected"]].copy()

    selected = selected.rename(
        columns={
            "n_features_selected_target": "n_features",
        }
    )

    selected = selected[
        [
            "source",
            "n_features",
            "feature",
            "rfe_rank",
            "selected_model_abs_coefficient_mean",
        ]
    ].sort_values(
        by=[
            "source",
            "n_features",
            "rfe_rank",
            "selected_model_abs_coefficient_mean",
        ],
        ascending=[True, True, True, False],
    )

    output_path = OUTPUT_DIR / "feature_selection" / "selected_features_for_classification.csv"
    selected.to_csv(output_path, index=False)

    print(f"[SAVED] {output_path}")

    # Also save one simple text file per source/count.
    text_dir = OUTPUT_DIR / "feature_selection" / "selected_feature_text_files"
    text_dir.mkdir(parents=True, exist_ok=True)

    for (source, n_features), group in selected.groupby(["source", "n_features"]):
        filename = f"selected_features_{safe_filename(source)}_top_{n_features}.txt"
        output_txt_path = text_dir / filename

        with open(output_txt_path, "w", encoding="utf-8") as file:
            for feature in group["feature"].tolist():
                file.write(f"{feature}\n")

        print(f"[SAVED] {output_txt_path}")


# =============================================================================
# Summary tables
# =============================================================================

def create_feature_summary_table(
    df: pd.DataFrame,
    sensor_columns: Dict[str, List[str]],
) -> pd.DataFrame:
    rows = []

    for source, columns in sensor_columns.items():
        if len(columns) == 0:
            rows.append(
                {
                    "source": source,
                    "n_features": 0,
                    "missing_percentage": np.nan,
                    "mean_variance": np.nan,
                    "median_variance": np.nan,
                }
            )
            continue

        X = df[columns].replace([np.inf, -np.inf], np.nan)
        missing_percentage = X.isna().mean().mean() * 100.0
        variances = X.var(axis=0, skipna=True)

        rows.append(
            {
                "source": source,
                "n_features": len(columns),
                "missing_percentage": missing_percentage,
                "mean_variance": variances.mean(),
                "median_variance": variances.median(),
            }
        )

    return pd.DataFrame(rows)


def create_top_feature_tables(
    df: pd.DataFrame,
    sensor_columns: Dict[str, List[str]],
    top_n: int = 30,
) -> None:
    """
    Saves the top ANOVA-ranked features for each sensor source as separate CSV files.
    """

    top_features_dir = OUTPUT_DIR / "top_features_by_source"
    top_features_dir.mkdir(parents=True, exist_ok=True)

    for source in SOURCE_ORDER:
        columns = sensor_columns[source]
        ranking = compute_anova_scores(df, columns)

        output_path = top_features_dir / f"top_features_{safe_filename(source)}.csv"

        if ranking.empty:
            pd.DataFrame({"message": ["No usable features"]}).to_csv(
                output_path,
                index=False,
            )
        else:
            ranking.head(top_n).to_csv(
                output_path,
                index=False,
            )

        print(f"[SAVED] {output_path}")


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    ensure_output_dir()

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

    print()

    sensor_columns = get_all_sensor_columns(df)

    print("=" * 80)
    print("Feature counts by source")
    print("=" * 80)

    for source in SOURCE_ORDER:
        print(f"{source}: {len(sensor_columns[source])} features")

    print()

    # -------------------------------------------------------------------------
    # General EDA plots
    # -------------------------------------------------------------------------

    plot_class_distribution(df)
    plot_feature_counts(sensor_columns)
    plot_missing_values_by_source(df, sensor_columns)

    # -------------------------------------------------------------------------
    # Summary tables
    # -------------------------------------------------------------------------

    feature_summary = create_feature_summary_table(df, sensor_columns)
    feature_summary_path = OUTPUT_DIR / "eda_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)
    print(f"[SAVED] {feature_summary_path}")

    create_top_feature_tables(
        df=df,
        sensor_columns=sensor_columns,
        top_n=TOP_N_ANOVA_EXPORT,
    )

    # -------------------------------------------------------------------------
    # Source-specific EDA plots
    # -------------------------------------------------------------------------

    for source in SOURCE_ORDER:
        print("=" * 80)
        print(f"Creating EDA plots for: {source}")
        print("=" * 80)

        columns = sensor_columns[source]
        prefixes = SENSOR_PREFIXES[source]

        if len(columns) == 0:
            print(f"[SKIP] No features found for {source}")
            continue

        ranking = plot_top_anova_features(
            df=df,
            source_name=source,
            feature_columns=columns,
            top_n=20,
        )

        plot_pca_scatter(
            df=df,
            source_name=source,
            feature_columns=columns,
        )

        plot_class_mean_heatmap(
            df=df,
            source_name=source,
            feature_columns=columns,
            ranking=ranking,
            top_n=20,
        )

        plot_top_feature_boxplots(
            df=df,
            source_name=source,
            ranking=ranking,
            top_n=6,
        )

        plot_correlation_heatmap(
            df=df,
            source_name=source,
            feature_columns=columns,
            max_features=25,
        )

        plot_block_level_importance(
            df=df,
            source_name=source,
            feature_columns=columns,
            prefixes=prefixes,
        )

    # -------------------------------------------------------------------------
    # Feature selection
    # -------------------------------------------------------------------------

    all_rfe_df = create_rfe_feature_selection_outputs(
        df=df,
        sensor_columns=sensor_columns,
    )

    create_combined_feature_ranking_outputs(
        df=df,
        sensor_columns=sensor_columns,
        all_rfe_df=all_rfe_df,
    )

    create_final_selected_feature_lists(
        all_rfe_df=all_rfe_df,
    )

    print()
    print("=" * 80)
    print("EDA + feature selection completed")
    print("=" * 80)
    print(f"Main output folder: {OUTPUT_DIR.resolve()}")
    print(f"Feature-selection folder: {(OUTPUT_DIR / 'feature_selection').resolve()}")


if __name__ == "__main__":
    main()