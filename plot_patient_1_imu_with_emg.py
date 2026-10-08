"""Plot raw Patient 11 IMU data and an EMG example.

Each CSV is plotted at its original timestamps without overlap trimming,
resampling, or bilateral synchronization. Time zero is the earliest timestamp
among the four files, which leaves the original recording offsets visible.
The EMG plot is illustrative rather than measured and is generated separately
from each leg's native sensor timing.

Run from the repository root:

    python3 plot_patient_1_imu_with_emg.py

Edit the parameters at the start of ``main()`` if a different input folder,
output folder, or time interval is needed. Three PNG files are produced: one
each for accelerometer, gyroscope, and illustrative EMG results.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt


SAMPLE_RATE_HZ = 100.0
EMG_SAMPLE_RATE_HZ = 1000.0
STREAM_FILES = {
    "left_acc": "LeftShank-Accelerometer.csv",
    "left_gyro": "LeftShank-Gyroscope.csv",
    "right_acc": "RightShank-Accelerometer.csv",
    "right_gyro": "RightShank-Gyroscope.csv",
}

AXIS_COLORS = {"x": "#244a73", "y": "#b45f06", "z": "#2f7d32"}
EVENT_COLOR = "#c43c39"


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> str:
    """Find a column while ignoring whitespace and case."""
    normalized = {re.sub(r"\s+", "", col.lower()): col for col in columns}
    for candidate in candidates:
        match = normalized.get(re.sub(r"\s+", "", candidate.lower()))
        if match is not None:
            return match
    raise ValueError(f"Could not find any of {tuple(candidates)} in {list(columns)}")


def read_sensor(path: Path) -> pd.DataFrame:
    """Read one accelerometer or gyroscope CSV into standard columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing sensor file: {path}")

    raw = pd.read_csv(path)
    epoch = find_column(
        raw.columns, ("epoc (ms)", "epoch (ms)", "epoch_ms", "epoc_ms")
    )
    axes = {
        axis: find_column(
            raw.columns,
            (f"{axis}-axis (g)", f"{axis}-axis (deg/s)", axis),
        )
        for axis in "xyz"
    }
    clean = pd.DataFrame(
        {
            "epoch_ms": pd.to_numeric(raw[epoch], errors="coerce"),
            **{
                axis: pd.to_numeric(raw[column], errors="coerce")
                for axis, column in axes.items()
            },
        }
    ).dropna()
    clean = clean.sort_values("epoch_ms").drop_duplicates("epoch_ms")
    if len(clean) < 10:
        raise ValueError(f"Too few valid rows in {path}")
    return clean.reset_index(drop=True)


def load_native_streams(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all four files without resampling or synchronizing them."""
    streams = {
        name: read_sensor(data_dir / filename)
        for name, filename in STREAM_FILES.items()
    }
    earliest_epoch = min(float(frame["epoch_ms"].iloc[0]) for frame in streams.values())
    for frame in streams.values():
        frame["time_s"] = (frame["epoch_ms"] - earliest_epoch) / 1000.0
    return streams


def lowpass(values: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Filter finite contiguous blocks without bridging recording gaps."""
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan)
    finite = np.isfinite(values)
    changes = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    sos = butter(2, cutoff_hz / (SAMPLE_RATE_HZ / 2.0), btype="low", output="sos")
    for start, end in zip(changes[0::2], changes[1::2]):
        block = values[start:end]
        if len(block) > 15:
            output[start:end] = sosfiltfilt(sos, block)
    return output


def detect_gait_events(gyro_z: np.ndarray) -> np.ndarray:
    """Detect dominant shank angular-velocity peaks with automatic polarity."""
    filtered = lowpass(gyro_z, cutoff_hz=6.0)
    finite = np.isfinite(filtered)
    if finite.sum() < SAMPLE_RATE_HZ:
        return np.array([], dtype=int)

    signal = filtered.copy()
    fill = float(np.nanmedian(signal))
    signal[~finite] = fill
    q05, median, q95 = np.percentile(signal[finite], [5, 50, 95])
    polarity = 1.0 if q95 - median >= median - q05 else -1.0
    oriented = polarity * signal
    center = float(np.median(oriented[finite]))
    mad = float(np.median(np.abs(oriented[finite] - center)))
    robust_sigma = max(1.4826 * mad, 1e-6)
    events, _ = find_peaks(
        oriented,
        height=center + max(20.0, 1.5 * robust_sigma),
        prominence=max(15.0, 0.75 * robust_sigma),
        distance=int(0.80 * SAMPLE_RATE_HZ),
    )
    return events[finite[events]]


def synthetic_emg_envelope(
    time_s: np.ndarray,
    gyro_z: np.ndarray,
    acceleration: np.ndarray,
    events: np.ndarray,
) -> np.ndarray:
    """Create a deterministic TA-like envelope tied to measured gait events."""
    envelope = np.full_like(time_s, 0.025, dtype=float)
    if not len(events):
        return envelope

    gyro_scale = max(float(np.nanpercentile(np.abs(gyro_z), 95)), 1e-6)
    acc_dynamic = np.abs(acceleration - np.nanmedian(acceleration))
    acc_scale = max(float(np.nanpercentile(acc_dynamic, 95)), 1e-6)

    for number, event in enumerate(events):
        if number + 1 < len(events):
            stride = time_s[events[number + 1]] - time_s[event]
        elif number:
            stride = time_s[event] - time_s[events[number - 1]]
        else:
            stride = 1.1
        stride = float(np.clip(stride, 0.65, 1.8))
        intensity = 0.55 + 0.25 * min(abs(gyro_z[event]) / gyro_scale, 1.5)
        intensity += 0.20 * min(acc_dynamic[event] / acc_scale, 1.5)

        # TA-like activity: a strong swing/heel-strike burst and a smaller
        # early-stance burst. Gaussian pulses form an illustrative envelope.
        main_center = time_s[event]
        stance_center = main_center + 0.18 * stride
        envelope += intensity * np.exp(-0.5 * ((time_s - main_center) / 0.075) ** 2)
        envelope += 0.38 * intensity * np.exp(
            -0.5 * ((time_s - stance_center) / 0.10) ** 2
        )

    maximum = float(np.nanmax(envelope))
    return envelope / maximum if maximum > 0 else envelope


def synthetic_raw_emg(
    time_s: np.ndarray,
    gyro_z: np.ndarray,
    acceleration: np.ndarray,
    events: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate bipolar, raw-style surface EMG from the gait-linked envelope."""
    envelope = synthetic_emg_envelope(time_s, gyro_z, acceleration, events)
    emg_time = np.arange(
        time_s[0], time_s[-1] + 0.5 / EMG_SAMPLE_RATE_HZ, 1.0 / EMG_SAMPLE_RATE_HZ
    )
    interpolated = np.interp(emg_time, time_s, envelope)
    baseline = float(np.min(interpolated))
    activation = (interpolated - baseline) / max(
        float(np.max(interpolated) - baseline), 1e-9
    )

    # Band-limited stochastic interference approximates the summation of many
    # motor-unit action potentials. A fixed seed makes every run reproducible.
    rng = np.random.default_rng(random_seed)
    white_noise = rng.standard_normal(len(emg_time))
    bandpass = butter(
        4,
        (20.0, 450.0),
        btype="bandpass",
        fs=EMG_SAMPLE_RATE_HZ,
        output="sos",
    )
    interference = sosfiltfilt(bandpass, white_noise)
    interference /= max(float(np.std(interference)), 1e-9)

    # Rest retains a thin electrical-noise band; contractions increase the
    # density and amplitude of the positive/negative interference pattern.
    amplitude = 0.025 + 0.975 * np.power(activation, 0.75)
    raw_emg = amplitude * interference
    scale = max(float(np.percentile(np.abs(raw_emg), 99.8)), 1e-9)
    raw_emg /= scale
    return emg_time, raw_emg


def select_native_intervals(
    streams: dict[str, pd.DataFrame], start_s: float, duration_s: float | None
) -> dict[str, pd.DataFrame]:
    """Select a display interval without changing any native samples."""
    if start_s < 0:
        raise ValueError("start_s must be non-negative")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be positive")
    end_s = (
        max(float(frame["time_s"].iloc[-1]) for frame in streams.values())
        if duration_s is None
        else start_s + duration_s
    )
    selected = {
        name: frame.loc[frame["time_s"].between(start_s, end_s)].copy()
        for name, frame in streams.items()
    }
    empty = [name for name, frame in selected.items() if frame.empty]
    if empty:
        raise ValueError(f"Requested interval contains no data for: {', '.join(empty)}")
    return selected


def add_xyz_panel(
    axis: plt.Axes,
    stream: pd.DataFrame,
    ylabel: str,
    cutoff_hz: float,
) -> None:
    """Plot a native stream's filtered x/y/z channels."""
    time_s = stream["time_s"].to_numpy(float)
    for coordinate in "xyz":
        axis.plot(
            time_s,
            lowpass(stream[coordinate].to_numpy(float), cutoff_hz),
            color=AXIS_COLORS[coordinate],
            linewidth=0.75,
            label=f"{coordinate.upper()}-axis",
        )
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.20)
    axis.legend(loc="upper right", ncols=3, fontsize=7, framealpha=0.9)


def make_plots(
    streams: dict[str, pd.DataFrame], output_dir: Path
) -> tuple[int, int, list[Path]]:
    """Save three plots while preserving every file's native timestamps."""
    left_acc = streams["left_acc"]
    right_acc = streams["right_acc"]
    left_gyro = streams["left_gyro"]
    right_gyro = streams["right_gyro"]
    left_events = detect_gait_events(left_gyro["z"].to_numpy(float))
    right_events = detect_gait_events(right_gyro["z"].to_numpy(float))

    def acceleration_at_gyro_times(
        acceleration: pd.DataFrame, gyroscope: pd.DataFrame
    ) -> np.ndarray:
        # This is used only to scale the illustrative EMG bursts. The plotted
        # IMU streams themselves remain entirely raw.
        magnitude = np.sqrt(
            sum(acceleration[axis].to_numpy(float) ** 2 for axis in "xyz")
        )
        return np.interp(
            gyroscope["epoch_ms"].to_numpy(float),
            acceleration["epoch_ms"].to_numpy(float),
            magnitude,
        )

    left_sensor_time = left_gyro["time_s"].to_numpy(float)
    right_sensor_time = right_gyro["time_s"].to_numpy(float)
    left_emg_time, left_emg = synthetic_raw_emg(
        left_sensor_time,
        left_gyro["z"].to_numpy(float),
        acceleration_at_gyro_times(left_acc, left_gyro),
        left_events,
        random_seed=1101,
    )
    right_emg_time, right_emg = synthetic_raw_emg(
        right_sensor_time,
        right_gyro["z"].to_numpy(float),
        acceleration_at_gyro_times(right_acc, right_gyro),
        right_events,
        random_seed=1102,
    )
    x_min = min(float(frame["time_s"].iloc[0]) for frame in streams.values())
    x_max = max(float(frame["time_s"].iloc[-1]) for frame in streams.values())

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "Healthy_Patient_11_raw_accelerometer.png",
        output_dir / "Healthy_Patient_11_raw_gyroscope.png",
        output_dir / "Healthy_Patient_11_raw_emg.png",
    ]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    add_xyz_panel(axes[0], left_acc, "Left acc.\n(g)", 15.0)
    add_xyz_panel(axes[1], right_acc, "Right acc.\n(g)", 15.0)
    axes[1].set_xlabel("Time from earliest sensor recording (s)")
    axes[0].set_xlim(x_min, x_max)
    fig.suptitle("Healthy Patient 11: raw accelerometer signals", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_paths[0], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    add_xyz_panel(axes[0], left_gyro, "Left gyro\n(deg/s)", 20.0)
    add_xyz_panel(axes[1], right_gyro, "Right gyro\n(deg/s)", 20.0)
    for axis, events, stream in (
        (axes[0], left_events, left_gyro),
        (axes[1], right_events, right_gyro),
    ):
        time_s = stream["time_s"].to_numpy(float)
        gyro_z = lowpass(stream["z"].to_numpy(float), 20.0)
        axis.scatter(
            time_s[events],
            gyro_z[events],
            s=13,
            color=EVENT_COLOR,
            label=f"Detected gait events (n={len(events)})",
            zorder=4,
        )
        axis.legend(loc="upper right", ncols=4, fontsize=7, framealpha=0.9)
    axes[1].set_xlabel("Time from earliest sensor recording (s)")
    axes[0].set_xlim(x_min, x_max)
    fig.suptitle("Healthy Patient 11: raw gyroscope signals", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_paths[1], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, sharey=True)
    for axis, emg_time, emg, side, color in (
        (axes[0], left_emg_time, left_emg, "Left", "#244a73"),
        (axes[1], right_emg_time, right_emg, "Right", "#c46a32"),
    ):
        axis.plot(
            emg_time,
            emg,
            color=color,
            linewidth=0.35,
            label=f"{side} TA raw-style example",
        )
        axis.axhline(0.0, color="#333333", linewidth=0.5, alpha=0.65)
        axis.set_ylabel(f"{side} EMG\n(a.u.)")
        axis.set_ylim(-1.15, 1.15)
        axis.grid(alpha=0.20)
        axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[1].set_xlabel("Time from earliest sensor recording (s)")
    axes[0].set_xlim(x_min, x_max)
    fig.suptitle("Healthy Patient 11: simulated raw surface EMG signals", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_paths[2], dpi=150)
    plt.close(fig)
    return len(left_events), len(right_events), output_paths


def main() -> None:
    # ---------------------------------------------------------------------
    # Plot parameters: edit these values, then run this file directly.
    # Use duration_s = None to plot every file's complete native interval.
    # ---------------------------------------------------------------------
    data_dir = Path("Data/Healthy/Patient_11")
    output_dir = Path("third_article/outputs/sensor_derived_severity_with_emg")
    start_s = 0.0
    duration_s = None

    native_streams = load_native_streams(data_dir)
    selected = select_native_intervals(native_streams, start_s, duration_s)
    left_count, right_count, output_paths = make_plots(selected, output_dir)
    for output_path in output_paths:
        print(f"Saved {output_path}")
    print(
        f"Plotted raw streams from {start_s:.2f} s; "
        f"detected {left_count} left and {right_count} right gait events."
    )


if __name__ == "__main__":
    main()
