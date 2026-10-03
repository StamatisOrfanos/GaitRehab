"""
Create sensor-derived gait-asymmetry severity classes from raw bilateral IMU data.

The script intentionally does not read the existing feature dataset. It discovers
participants from this directory structure:

    Data/
        Healthy/Patient_1/...Patient_N/
        Stroke/Patient_1/...Patient_N/

Each patient directory must contain:

    LeftShank-Accelerometer.csv
    LeftShank-Gyroscope.csv
    RightShank-Accelerometer.csv
    RightShank-Gyroscope.csv

Method:
    1. Synchronise all four files on their common epoch-time interval at 100 Hz.
    2. Filter left/right gyroscope z-axis signals.
    3. Detect gait cycles independently on each leg.
    4. Calculate four fixed, participant-level asymmetry components:
       temporal asymmetry, angular-velocity amplitude asymmetry,
       variability asymmetry, and bilateral waveform dissimilarity.
    5. Robustly standardise each component using healthy controls only.
    6. Average the four equally weighted positive deviations to create one
       continuous Sensor-Derived Gait Asymmetry Severity Score.
    7. Define Low/reference using the observed healthy reference envelope.
       Split participants above that envelope into Moderate and High groups
       with a two-component Gaussian mixture model, with a median fallback.

Important interpretation:
    The output represents sensor-derived gait-asymmetry severity. It is not a
    clinically validated stroke-severity score.

Primary outputs:
    sensor_derived_gait_asymmetry_severity.csv
    gait_cycle_metrics.csv
    healthy_reference_statistics.csv
    severity_method_summary.json
    excluded_participants.csv (only when exclusions occur)
    synchronised/*.csv (unless --no-save-synchronised is used)

Example:
    python3 create_sensor_derived_severity_dataset.py \
        --data-dir Data \
        --output-dir outputs/sensor_derived_severity
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from sklearn.mixture import GaussianMixture


REQUIRED_FILES = (
    "LeftShank-Accelerometer.csv",
    "LeftShank-Gyroscope.csv",
    "RightShank-Accelerometer.csv",
    "RightShank-Gyroscope.csv",
)

COMPONENT_COLUMNS = (
    "temporal_asymmetry",
    "amplitude_asymmetry",
    "variability_asymmetry",
    "waveform_dissimilarity",
)

CLASS_NAMES = {
    0: "Low/reference asymmetry",
    1: "Moderate pathological asymmetry",
    2: "High pathological asymmetry",
}


@dataclass(frozen=True)
class AnalysisConfig:
    sampling_rate_hz: float = 100.0
    lowpass_cutoff_hz: float = 35.0
    lowpass_order: int = 2
    maximum_interpolation_gap_seconds: float = 0.05
    minimum_valid_segment_seconds: float = 8.0
    minimum_peak_distance_seconds: float = 0.80
    minimum_stride_seconds: float = 0.60
    maximum_stride_seconds: float = 2.00
    minimum_peak_height_deg_s: float = 5.0
    minimum_peak_prominence_deg_s: float = 3.0
    minimum_cycles_per_leg: int = 5
    waveform_points: int = 101
    maximum_component_z: float = 10.0
    healthy_cutoff_method: str = "maximum"
    gmm_initialisations: int = 50
    bootstrap_iterations: int = 500
    random_state: int = 42


@dataclass
class LegSummary:
    side: str
    cycle_count: int
    median_stride_seconds: float
    stride_cv: float
    median_cycle_amplitude_deg_s: float
    amplitude_cv: float
    waveform_template: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create participant-level sensor-derived gait-asymmetry severity "
            "classes from raw bilateral shank IMU files."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        help="Directory containing Healthy/ and Stroke/ (default: Data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sensor_derived_severity"),
        help="Output directory (default: outputs/sensor_derived_severity).",
    )
    parser.add_argument(
        "--no-save-synchronised",
        action="store_true",
        help="Do not save the per-participant synchronized IMU CSV files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files already present in the output directory.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=500,
        help="Bootstrap repetitions for the moderate/high threshold (default: 500).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used by mixture modelling and bootstrap (default: 42).",
    )
    parser.add_argument(
        "--healthy-cutoff-method",
        choices=("maximum", "q95"),
        default="maximum",
        help=(
            "Healthy reference boundary: maximum keeps every valid healthy "
            "participant in the reference class; q95 is less conservative "
            "(default: maximum)."
        ),
    )
    parser.add_argument(
        "--minimum-cycles-per-leg",
        type=int,
        default=5,
        help="Minimum accepted gait cycles on each leg (default: 5).",
    )
    parser.add_argument(
        "--minimum-peak-height",
        type=float,
        default=5.0,
        help="Minimum adaptive gyro-z peak height in deg/s (default: 5).",
    )
    parser.add_argument(
        "--minimum-peak-prominence",
        type=float,
        default=3.0,
        help="Minimum adaptive gyro-z peak prominence in deg/s (default: 3).",
    )
    return parser.parse_args()


def natural_sort_key(value: str) -> Tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. "
            "Use --overwrite to replace named output files."
        )
    path.mkdir(parents=True, exist_ok=True)


def discover_participants(data_dir: Path) -> List[Tuple[str, Path]]:
    participants: List[Tuple[str, Path]] = []
    for cohort in ("Healthy", "Stroke"):
        cohort_dir = data_dir / cohort
        if not cohort_dir.is_dir():
            raise FileNotFoundError(f"Missing cohort directory: {cohort_dir}")
        folders = sorted(
            (p for p in cohort_dir.iterdir() if p.is_dir()),
            key=lambda p: natural_sort_key(p.name),
        )
        for folder in folders:
            missing = [name for name in REQUIRED_FILES if not (folder / name).is_file()]
            if missing:
                warnings.warn(
                    f"Skipping {cohort}/{folder.name}; missing: {', '.join(missing)}"
                )
                continue
            participants.append((cohort, folder))
    if not participants:
        raise RuntimeError(f"No valid participant folders found under {data_dir}")
    return participants


def _find_column(columns: Sequence[str], candidates: Iterable[str]) -> str:
    normalized = {re.sub(r"\s+", "", c.lower()): c for c in columns}
    for candidate in candidates:
        key = re.sub(r"\s+", "", candidate.lower())
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Could not find any of {list(candidates)} in columns: {list(columns)}"
    )


def read_imu_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    epoch_column = _find_column(
        frame.columns,
        ("epoc (ms)", "epoch (ms)", "epoch_ms", "epoc_ms"),
    )
    x_column = _find_column(frame.columns, ("x-axis (deg/s)", "x-axis (g)", "x"))
    y_column = _find_column(frame.columns, ("y-axis (deg/s)", "y-axis (g)", "y"))
    z_column = _find_column(frame.columns, ("z-axis (deg/s)", "z-axis (g)", "z"))
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
    config: AnalysisConfig,
) -> pd.DataFrame:
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
    if common_end - common_start < config.minimum_valid_segment_seconds * 1000:
        raise ValueError("The four sensor streams do not have sufficient time overlap.")
    grid_ms = np.arange(common_start, common_end + step_ms / 2, step_ms)

    synchronized = pd.DataFrame(
        {
            "epoch_ms": grid_ms.astype(np.int64),
            "elapsed_common_s": (grid_ms - grid_ms[0]) / 1000.0,
        }
    )
    validity: Dict[str, np.ndarray] = {}
    maximum_gap_ms = config.maximum_interpolation_gap_seconds * 1000.0
    for stream_name, stream in streams.items():
        values, valid = interpolate_stream(stream, grid_ms, maximum_gap_ms)
        synchronized[f"{stream_name}_x"] = values[:, 0]
        synchronized[f"{stream_name}_y"] = values[:, 1]
        synchronized[f"{stream_name}_z"] = values[:, 2]
        synchronized[f"{stream_name}_valid"] = valid.astype(int)
        validity[stream_name] = valid
    synchronized["all_streams_valid"] = np.logical_and.reduce(
        list(validity.values())
    ).astype(int)
    synchronized["both_gyroscopes_valid"] = (
        validity["left_gyro"] & validity["right_gyro"]
    ).astype(int)
    return synchronized


def contiguous_true_ranges(mask: np.ndarray, minimum_samples: int) -> List[Tuple[int, int]]:
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    ranges: List[Tuple[int, int]] = []
    for start, end in zip(changes[0::2], changes[1::2]):
        if end - start >= minimum_samples:
            ranges.append((int(start), int(end)))
    return ranges


def filter_signal_segments(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    nyquist = config.sampling_rate_hz / 2.0
    if not 0 < config.lowpass_cutoff_hz < nyquist:
        raise ValueError("Low-pass cutoff must be between 0 and Nyquist frequency.")
    sos = butter(
        config.lowpass_order,
        config.lowpass_cutoff_hz / nyquist,
        btype="low",
        output="sos",
    )
    filtered = np.full(len(signal), np.nan, dtype=float)
    minimum_samples = max(
        int(round(config.minimum_valid_segment_seconds * config.sampling_rate_hz)),
        3 * (2 * config.lowpass_order + 1),
    )
    for start, end in contiguous_true_ranges(valid_mask, minimum_samples):
        block = np.asarray(signal[start:end], dtype=float)
        if np.isfinite(block).all():
            filtered[start:end] = sosfiltfilt(sos, block)
    return filtered


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    if not np.isfinite(mean) or abs(mean) < 1e-12:
        return float("nan")
    return float(np.std(values, ddof=1) / abs(mean)) if len(values) > 1 else 0.0


def resample_waveform(values: np.ndarray, points: int) -> np.ndarray:
    original = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.interp(target, original, values)


def extract_leg_cycles(
    filtered_signal: np.ndarray,
    valid_mask: np.ndarray,
    epoch_ms: np.ndarray,
    side: str,
    participant_key: str,
    cohort: str,
    config: AnalysisConfig,
) -> Tuple[LegSummary, List[Dict[str, object]]]:
    minimum_segment_samples = int(
        round(config.minimum_valid_segment_seconds * config.sampling_rate_hz)
    )
    cycle_rows: List[Dict[str, object]] = []
    stride_values: List[float] = []
    amplitude_values: List[float] = []
    waveforms: List[np.ndarray] = []
    cycle_number = 0

    finite_valid = valid_mask.astype(bool) & np.isfinite(filtered_signal)
    for segment_index, (start, end) in enumerate(
        contiguous_true_ranges(finite_valid, minimum_segment_samples), start=1
    ):
        segment = filtered_signal[start:end]
        lower, median, upper = np.percentile(segment, [5, 50, 95])
        polarity = 1.0 if abs(upper - median) >= abs(lower - median) else -1.0
        oriented = polarity * segment
        median_oriented = float(np.median(oriented))
        mad = float(np.median(np.abs(oriented - median_oriented)))
        robust_sigma = max(1.4826 * mad, 1e-9)
        peak_height = median_oriented + max(
            config.minimum_peak_height_deg_s, 1.5 * robust_sigma
        )
        prominence = max(
            config.minimum_peak_prominence_deg_s, 0.75 * robust_sigma
        )
        peaks, _ = find_peaks(
            oriented,
            height=peak_height,
            prominence=prominence,
            distance=int(
                round(config.minimum_peak_distance_seconds * config.sampling_rate_hz)
            ),
        )
        if len(peaks) < 2:
            continue

        for first, second in zip(peaks[:-1], peaks[1:]):
            absolute_first = start + int(first)
            absolute_second = start + int(second)
            stride_seconds = (epoch_ms[absolute_second] - epoch_ms[absolute_first]) / 1000.0
            if not config.minimum_stride_seconds <= stride_seconds <= config.maximum_stride_seconds:
                continue
            waveform = filtered_signal[absolute_first : absolute_second + 1]
            if len(waveform) < 4 or not np.isfinite(waveform).all():
                continue
            amplitude = float(np.ptp(waveform))
            if amplitude <= 0:
                continue
            normalized = resample_waveform(waveform, config.waveform_points)
            stride_values.append(float(stride_seconds))
            amplitude_values.append(amplitude)
            waveforms.append(normalized)
            cycle_number += 1
            cycle_rows.append(
                {
                    "participant_key": participant_key,
                    "cohort": cohort,
                    "side": side,
                    "segment_index": segment_index,
                    "cycle_index": cycle_number,
                    "start_epoch_ms": int(epoch_ms[absolute_first]),
                    "end_epoch_ms": int(epoch_ms[absolute_second]),
                    "stride_duration_seconds": float(stride_seconds),
                    "cycle_peak_to_peak_deg_s": amplitude,
                    "detected_polarity": int(polarity),
                }
            )

    if len(stride_values) < config.minimum_cycles_per_leg:
        raise ValueError(
            f"Only {len(stride_values)} valid {side.lower()} gait cycles detected; "
            f"at least {config.minimum_cycles_per_leg} are required."
        )
    strides = np.asarray(stride_values, dtype=float)
    amplitudes = np.asarray(amplitude_values, dtype=float)
    template = np.median(np.vstack(waveforms), axis=0)
    summary = LegSummary(
        side=side,
        cycle_count=len(strides),
        median_stride_seconds=float(np.median(strides)),
        stride_cv=coefficient_of_variation(strides),
        median_cycle_amplitude_deg_s=float(np.median(amplitudes)),
        amplitude_cv=coefficient_of_variation(amplitudes),
        waveform_template=template,
    )
    return summary, cycle_rows


def symmetric_difference(left: float, right: float, epsilon: float = 1e-12) -> float:
    denominator = abs(left) + abs(right) + epsilon
    return float(2.0 * abs(left - right) / denominator)


def waveform_dissimilarity(left: np.ndarray, right: np.ndarray) -> float:
    left_sd = float(np.std(left))
    right_sd = float(np.std(right))
    if left_sd < 1e-12 or right_sd < 1e-12:
        return float("nan")
    correlation = float(np.corrcoef(left, right)[0, 1])
    return float(np.clip(1.0 - abs(correlation), 0.0, 1.0))


def calculate_components(left: LegSummary, right: LegSummary) -> Dict[str, float]:
    temporal = symmetric_difference(
        left.median_stride_seconds, right.median_stride_seconds
    )
    amplitude = symmetric_difference(
        left.median_cycle_amplitude_deg_s,
        right.median_cycle_amplitude_deg_s,
    )
    variability = float(
        np.mean(
            [
                symmetric_difference(left.stride_cv, right.stride_cv),
                symmetric_difference(left.amplitude_cv, right.amplitude_cv),
            ]
        )
    )
    waveform = waveform_dissimilarity(
        left.waveform_template, right.waveform_template
    )
    return {
        "temporal_asymmetry": temporal,
        "amplitude_asymmetry": amplitude,
        "variability_asymmetry": variability,
        "waveform_dissimilarity": waveform,
    }


def process_participant(
    cohort: str,
    participant_dir: Path,
    output_dir: Path,
    save_synchronised: bool,
    config: AnalysisConfig,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    participant_key = f"{cohort}_{participant_dir.name}"
    synchronized = synchronise_participant(participant_dir, config)
    gyro_valid = synchronized["both_gyroscopes_valid"].to_numpy(dtype=bool)
    epoch_ms = synchronized["epoch_ms"].to_numpy(dtype=np.int64)
    left_filtered = filter_signal_segments(
        synchronized["left_gyro_z"].to_numpy(dtype=float), gyro_valid, config
    )
    right_filtered = filter_signal_segments(
        synchronized["right_gyro_z"].to_numpy(dtype=float), gyro_valid, config
    )
    synchronized["left_gyro_z_filtered"] = left_filtered
    synchronized["right_gyro_z_filtered"] = right_filtered

    left, left_cycles = extract_leg_cycles(
        left_filtered,
        gyro_valid,
        epoch_ms,
        "Left",
        participant_key,
        cohort,
        config,
    )
    right, right_cycles = extract_leg_cycles(
        right_filtered,
        gyro_valid,
        epoch_ms,
        "Right",
        participant_key,
        cohort,
        config,
    )
    components = calculate_components(left, right)
    if not all(np.isfinite(value) for value in components.values()):
        raise ValueError("At least one severity component is non-finite.")

    if save_synchronised:
        synchronized_dir = output_dir / "synchronised"
        synchronized_dir.mkdir(parents=True, exist_ok=True)
        synchronized.to_csv(
            synchronized_dir / f"{participant_key}_synchronised.csv", index=False
        )

    valid_seconds = float(
        gyro_valid.sum() / config.sampling_rate_hz
    )
    row: Dict[str, object] = {
        "participant_key": participant_key,
        "cohort": cohort,
        "patient_folder": participant_dir.name,
        "common_overlap_seconds": float(synchronized["elapsed_common_s"].iloc[-1]),
        "valid_bilateral_gyro_seconds": valid_seconds,
        "left_cycle_count": left.cycle_count,
        "right_cycle_count": right.cycle_count,
        "left_median_stride_seconds": left.median_stride_seconds,
        "right_median_stride_seconds": right.median_stride_seconds,
        "left_stride_cv": left.stride_cv,
        "right_stride_cv": right.stride_cv,
        "left_median_cycle_amplitude_deg_s": left.median_cycle_amplitude_deg_s,
        "right_median_cycle_amplitude_deg_s": right.median_cycle_amplitude_deg_s,
        "left_amplitude_cv": left.amplitude_cv,
        "right_amplitude_cv": right.amplitude_cv,
        **components,
    }
    return row, left_cycles + right_cycles


def robust_scale(values: np.ndarray) -> Tuple[float, float, str]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    method = "1.4826 × MAD"
    if not np.isfinite(scale) or scale <= 1e-12:
        q25, q75 = np.percentile(values, [25, 75])
        scale = float((q75 - q25) / 1.349)
        method = "IQR / 1.349 fallback"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        method = "standard-deviation fallback"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
        method = "unit-scale fallback"
    return median, scale, method


def add_healthy_referenced_scores(
    participants: pd.DataFrame,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    healthy = participants[participants["cohort"] == "Healthy"]
    if len(healthy) < 5:
        warnings.warn(
            f"Only {len(healthy)} healthy participants are valid. "
            "Reference statistics and class thresholds are preliminary."
        )
    references: List[Dict[str, object]] = []
    output = participants.copy()
    contribution_columns: List[str] = []
    for component in COMPONENT_COLUMNS:
        values = healthy[component].to_numpy(dtype=float)
        median, scale, scale_method = robust_scale(values)
        z_column = f"{component}_healthy_robust_z"
        contribution_column = f"{component}_severity_contribution"
        output[z_column] = (output[component] - median) / scale
        output[contribution_column] = output[z_column].clip(
            lower=0.0, upper=config.maximum_component_z
        )
        contribution_columns.append(contribution_column)
        references.append(
            {
                "component": component,
                "healthy_n": len(values),
                "healthy_median": median,
                "healthy_scale": scale,
                "scale_method": scale_method,
                "healthy_minimum": float(np.min(values)),
                "healthy_maximum": float(np.max(values)),
            }
        )
    output["sensor_derived_severity_score"] = output[contribution_columns].mean(
        axis=1
    )

    # Use leave-one-out healthy scores to estimate the reference boundary without
    # letting each control participant help define the distribution against which
    # that same participant is judged. The final severity scores above still use
    # the complete healthy reference cohort, which is the frozen study reference.
    output["healthy_leave_one_out_reference_score"] = np.nan
    healthy_indices = list(output.index[output["cohort"] == "Healthy"])
    if len(healthy_indices) >= 5:
        for held_out_index in healthy_indices:
            reference_indices = [
                index for index in healthy_indices if index != held_out_index
            ]
            contributions: List[float] = []
            for component in COMPONENT_COLUMNS:
                reference_values = output.loc[
                    reference_indices, component
                ].to_numpy(dtype=float)
                median, scale, _ = robust_scale(reference_values)
                held_out_value = float(output.at[held_out_index, component])
                z_value = (held_out_value - median) / scale
                contributions.append(
                    float(np.clip(z_value, 0.0, config.maximum_component_z))
                )
            output.at[
                held_out_index, "healthy_leave_one_out_reference_score"
            ] = float(np.mean(contributions))
    return output, pd.DataFrame(references)


def fit_gmm_boundary(
    values: np.ndarray,
    config: AnalysisConfig,
    seed: Optional[int] = None,
) -> Tuple[float, GaussianMixture]:
    array = np.asarray(values, dtype=float).reshape(-1, 1)
    if len(array) < 2 or len(np.unique(array)) < 2:
        raise ValueError("At least two distinct scores are required for GMM.")
    model = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=config.gmm_initialisations,
        random_state=config.random_state if seed is None else seed,
        reg_covar=1e-6,
    ).fit(array)
    means = model.means_.ravel()
    low_component, high_component = np.argsort(means)
    labels = model.predict(array)
    low_values = array.ravel()[labels == low_component]
    high_values = array.ravel()[labels == high_component]
    if len(low_values) == 0 or len(high_values) == 0:
        raise ValueError("GMM produced an empty severity subgroup.")
    if np.max(low_values) < np.min(high_values):
        boundary = float((np.max(low_values) + np.min(high_values)) / 2.0)
    else:
        boundary = float((means[low_component] + means[high_component]) / 2.0)
    return boundary, model


def bootstrap_gmm_boundaries(
    values: np.ndarray,
    config: AnalysisConfig,
) -> np.ndarray:
    if config.bootstrap_iterations <= 0 or len(values) < 4:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(config.random_state)
    boundaries: List[float] = []
    for iteration in range(config.bootstrap_iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        if len(np.unique(sample)) < 2:
            continue
        try:
            boundary, _ = fit_gmm_boundary(
                sample, config, seed=config.random_state + iteration + 1
            )
            if np.isfinite(boundary):
                boundaries.append(boundary)
        except Exception:
            continue
    return np.asarray(boundaries, dtype=float)


def assign_severity_classes(
    participants: pd.DataFrame,
    config: AnalysisConfig,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    output = participants.copy()
    score_column = "sensor_derived_severity_score"
    healthy_scores = output.loc[output["cohort"] == "Healthy", score_column].to_numpy(
        dtype=float
    )
    if len(healthy_scores) == 0:
        raise RuntimeError("No valid healthy participants are available.")

    leave_one_out_scores = output.loc[
        output["cohort"] == "Healthy", "healthy_leave_one_out_reference_score"
    ].dropna().to_numpy(dtype=float)
    if len(leave_one_out_scores) == len(healthy_scores) and len(healthy_scores) >= 5:
        cutoff_scores = leave_one_out_scores
        cutoff_score_source = "leave-one-out healthy reference scores"
    else:
        cutoff_scores = healthy_scores
        cutoff_score_source = "full-reference healthy scores (small-sample fallback)"

    if config.healthy_cutoff_method == "maximum":
        healthy_cutoff = float(np.max(cutoff_scores))
    elif config.healthy_cutoff_method == "q95":
        healthy_cutoff = float(np.quantile(cutoff_scores, 0.95))
    else:
        raise ValueError(
            f"Unsupported healthy cutoff method: {config.healthy_cutoff_method}"
        )

    above_reference = output[score_column] > healthy_cutoff
    pathological_scores = output.loc[above_reference, score_column].to_numpy(dtype=float)
    moderate_high_method = "not available"
    model: Optional[GaussianMixture] = None
    if len(pathological_scores) >= 4 and len(np.unique(pathological_scores)) >= 2:
        try:
            moderate_high_cutoff, model = fit_gmm_boundary(pathological_scores, config)
            moderate_high_method = "two-component Gaussian mixture"
        except Exception as exc:
            warnings.warn(f"GMM split failed ({exc}); using the median fallback.")
            moderate_high_cutoff = float(np.median(pathological_scores))
            moderate_high_method = "median fallback after GMM failure"
    elif len(pathological_scores) >= 2:
        moderate_high_cutoff = float(np.median(pathological_scores))
        moderate_high_method = "median fallback for small pathological group"
        warnings.warn(
            "Fewer than four participants exceeded the healthy reference. "
            "The Moderate/High division is preliminary."
        )
    elif len(pathological_scores) == 1:
        moderate_high_cutoff = float(pathological_scores[0])
        moderate_high_method = "insufficient data: single above-reference participant"
        warnings.warn("Only one participant exceeded the healthy reference.")
    else:
        moderate_high_cutoff = float("inf")
        moderate_high_method = "insufficient data: no above-reference participants"
        warnings.warn("No participant exceeded the healthy reference.")

    class_id = np.zeros(len(output), dtype=int)
    class_id[
        (output[score_column] > healthy_cutoff)
        & (output[score_column] <= moderate_high_cutoff)
    ] = 1
    class_id[output[score_column] > moderate_high_cutoff] = 2
    output["severity_class_id"] = class_id
    output["severity_class"] = output["severity_class_id"].map(CLASS_NAMES)
    output["above_healthy_reference"] = above_reference.astype(int)

    output["moderate_high_assignment_confidence"] = np.nan
    if model is not None and above_reference.any():
        probabilities = model.predict_proba(
            output.loc[above_reference, [score_column]].to_numpy(dtype=float)
        )
        output.loc[above_reference, "moderate_high_assignment_confidence"] = (
            probabilities.max(axis=1)
        )

    bootstrap_boundaries = bootstrap_gmm_boundaries(pathological_scores, config)
    output["bootstrap_probability_high"] = np.nan
    if len(bootstrap_boundaries) > 0:
        for index in output.index[above_reference]:
            score = float(output.at[index, score_column])
            output.at[index, "bootstrap_probability_high"] = float(
                np.mean(score > bootstrap_boundaries)
            )

    threshold_summary: Dict[str, object] = {
        "healthy_cutoff_method": config.healthy_cutoff_method,
        "healthy_cutoff_score_source": cutoff_score_source,
        "healthy_reference_cutoff": healthy_cutoff,
        "moderate_high_method": moderate_high_method,
        "moderate_high_cutoff": (
            moderate_high_cutoff if np.isfinite(moderate_high_cutoff) else None
        ),
        "participants_above_healthy_reference": int(above_reference.sum()),
        "bootstrap_successful_iterations": int(len(bootstrap_boundaries)),
        "bootstrap_boundary_2_5_percentile": (
            float(np.quantile(bootstrap_boundaries, 0.025))
            if len(bootstrap_boundaries)
            else None
        ),
        "bootstrap_boundary_97_5_percentile": (
            float(np.quantile(bootstrap_boundaries, 0.975))
            if len(bootstrap_boundaries)
            else None
        ),
        "class_counts": {
            CLASS_NAMES[class_number]: int((class_id == class_number).sum())
            for class_number in sorted(CLASS_NAMES)
        },
    }
    return output, threshold_summary


def serialize_json(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> int:
    args = parse_args()
    config = AnalysisConfig(
        bootstrap_iterations=args.bootstrap_iterations,
        random_state=args.random_state,
        healthy_cutoff_method=args.healthy_cutoff_method,
        minimum_cycles_per_leg=args.minimum_cycles_per_leg,
        minimum_peak_height_deg_s=args.minimum_peak_height,
        minimum_peak_prominence_deg_s=args.minimum_peak_prominence,
    )
    prepare_output_directory(args.output_dir, args.overwrite)
    participants = discover_participants(args.data_dir)
    print(f"Found {len(participants)} participant folders.")

    participant_rows: List[Dict[str, object]] = []
    cycle_rows: List[Dict[str, object]] = []
    exclusion_rows: List[Dict[str, object]] = []
    for number, (cohort, participant_dir) in enumerate(participants, start=1):
        key = f"{cohort}_{participant_dir.name}"
        print(f"[{number:02d}/{len(participants):02d}] {key}")
        try:
            participant_row, participant_cycles = process_participant(
                cohort,
                participant_dir,
                args.output_dir,
                save_synchronised=not args.no_save_synchronised,
                config=config,
            )
            participant_rows.append(participant_row)
            cycle_rows.extend(participant_cycles)
            print(
                "  cycles: "
                f"L={participant_row['left_cycle_count']}, "
                f"R={participant_row['right_cycle_count']}"
            )
        except Exception as exc:
            exclusion_rows.append(
                {
                    "participant_key": key,
                    "cohort": cohort,
                    "patient_folder": participant_dir.name,
                    "reason": str(exc),
                }
            )
            print(f"  EXCLUDED: {exc}")

    if not participant_rows:
        raise RuntimeError("No participant produced a valid severity record.")

    participant_frame = pd.DataFrame(participant_rows)
    scored, reference_frame = add_healthy_referenced_scores(participant_frame, config)
    final_frame, threshold_summary = assign_severity_classes(scored, config)
    final_frame = final_frame.sort_values(
        ["severity_class_id", "sensor_derived_severity_score", "cohort", "participant_key"]
    ).reset_index(drop=True)

    final_path = args.output_dir / "sensor_derived_gait_asymmetry_severity.csv"
    cycles_path = args.output_dir / "gait_cycle_metrics.csv"
    references_path = args.output_dir / "healthy_reference_statistics.csv"
    summary_path = args.output_dir / "severity_method_summary.json"
    final_frame.to_csv(final_path, index=False)
    pd.DataFrame(cycle_rows).to_csv(cycles_path, index=False)
    reference_frame.to_csv(references_path, index=False)
    if exclusion_rows:
        pd.DataFrame(exclusion_rows).to_csv(
            args.output_dir / "excluded_participants.csv", index=False
        )

    method_summary = {
        "interpretation": (
            "Sensor-derived gait-asymmetry severity; not a clinically validated "
            "stroke-severity scale."
        ),
        "data_directory": str(args.data_dir),
        "valid_participants": int(len(final_frame)),
        "excluded_participants": int(len(exclusion_rows)),
        "valid_participants_by_cohort": {
            str(key): int(value)
            for key, value in final_frame["cohort"].value_counts().to_dict().items()
        },
        "component_definition": {
            "temporal_asymmetry": (
                "Symmetric absolute difference between left and right median stride durations."
            ),
            "amplitude_asymmetry": (
                "Symmetric absolute difference between left and right median "
                "gait-cycle gyroscope-z peak-to-peak amplitudes."
            ),
            "variability_asymmetry": (
                "Mean of the symmetric differences in stride-duration CV and "
                "cycle-amplitude CV."
            ),
            "waveform_dissimilarity": (
                "One minus the absolute correlation between median time-normalised "
                "left and right gyroscope-z gait-cycle templates."
            ),
            "sensor_derived_severity_score": (
                "Equal-weight mean of the four positive, healthy-referenced robust "
                "z contributions, each capped at the configured maximum."
            ),
        },
        "configuration": asdict(config),
        "thresholds": threshold_summary,
        "circularity_prevention": {
            "label_definition": (
                "Severity labels are derived only from bilateral raw gyroscope-z "
                "gait-cycle measurements."
            ),
            "primary_classifier_rule": (
                "Do not use the exact gyroscope-z variables used to define the "
                "severity score as predictors in the primary classification "
                "experiment. Use accelerometer, gyroscope x/y and EMG features; "
                "an all-feature model may be reported only as an index-"
                "reconstruction sensitivity analysis."
            ),
            "affected_side_rule": (
                "No affected-side inference is used. All label components are "
                "absolute bilateral comparisons and are unchanged if left and "
                "right are swapped."
            ),
        },
        "outputs": {
            "severity_dataset": str(final_path),
            "cycle_metrics": str(cycles_path),
            "healthy_reference_statistics": str(references_path),
            "synchronised_files_saved": not args.no_save_synchronised,
        },
    }
    summary_path.write_text(
        json.dumps(method_summary, indent=2, default=serialize_json),
        encoding="utf-8",
    )

    print("\nClass counts")
    print(final_frame["severity_class"].value_counts().to_string())
    print(f"\nSaved severity dataset: {final_path}")
    print(f"Saved gait-cycle audit data: {cycles_path}")
    print(f"Saved method summary: {summary_path}")
    if exclusion_rows:
        print(
            f"Warning: {len(exclusion_rows)} participant(s) were excluded. "
            "Review excluded_participants.csv before analysis."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
