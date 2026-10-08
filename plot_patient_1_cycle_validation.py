"""Create the Patient 1 filtered gyroscope-z cycle-validation plot.

The processing deliberately matches the sensor-derived severity workflow:
the four IMU streams are synchronized to their common 100 Hz time grid,
gyroscope-z is low-pass filtered, and gait cycles are detected independently
for the left and right legs. Accepted cycle starts are shown as red markers.

Run from the repository root:

    python3 plot_patient_1_cycle_validation.py

Edit the paths at the start of ``main()`` if needed. There are no command-line
arguments.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from create_sensor_derived_severity_dataset_with_emg import (
    ANALYSIS_CONFIG,
    extract_leg_cycles,
    filter_signal_segments,
    synchronise_participant,
)


def create_cycle_validation_plot(
    patient_dir: Path, output_path: Path
) -> tuple[int, int]:
    """Process Patient 1 and save its bilateral cycle-validation figure."""
    synchronized = synchronise_participant(patient_dir, ANALYSIS_CONFIG)
    elapsed = synchronized["elapsed_common_s"].to_numpy(dtype=float)
    epoch_ms = synchronized["epoch_ms"].to_numpy(dtype=np.int64)
    valid = synchronized["both_gyroscopes_valid"].to_numpy(dtype=bool)

    left_filtered = filter_signal_segments(
        synchronized["left_gyro_z"].to_numpy(dtype=float),
        valid,
        ANALYSIS_CONFIG,
    )
    right_filtered = filter_signal_segments(
        synchronized["right_gyro_z"].to_numpy(dtype=float),
        valid,
        ANALYSIS_CONFIG,
    )

    _, left_cycles = extract_leg_cycles(
        left_filtered,
        valid,
        epoch_ms,
        "Left",
        "Healthy_Patient_1",
        "Healthy",
        ANALYSIS_CONFIG,
    )
    _, right_cycles = extract_leg_cycles(
        right_filtered,
        valid,
        epoch_ms,
        "Right",
        "Healthy_Patient_1",
        "Healthy",
        ANALYSIS_CONFIG,
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for axis, side, signal, cycles in (
        (axes[0], "Left", left_filtered, left_cycles),
        (axes[1], "Right", right_filtered, right_cycles),
    ):
        axis.plot(
            elapsed,
            signal,
            color="#244a73",
            linewidth=0.8,
            label="Filtered gyroscope-z",
        )

        start_epochs = np.asarray(
            [float(cycle["start_epoch_ms"]) for cycle in cycles], dtype=float
        )
        start_seconds = (start_epochs - float(epoch_ms[0])) / 1000.0
        indices = np.searchsorted(elapsed, start_seconds)
        indices = np.clip(indices, 0, len(signal) - 1)
        axis.scatter(
            elapsed[indices],
            signal[indices],
            s=14,
            color="#c43c39",
            label=f"Accepted cycle starts (n={len(cycles)})",
            zorder=3,
        )
        axis.set_ylabel(f"{side} gyro-z\n(deg/s)")
        axis.grid(alpha=0.20)
        axis.legend(loc="upper right", fontsize=8)

    axes[1].set_xlabel("Time from common synchronized start (s)")
    fig.suptitle(
        "Cycle validation: Healthy Patient 1\n"
        "Filtered gyroscope-z and accepted cycle starts",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return len(left_cycles), len(right_cycles)


def main() -> None:
    # ---------------------------------------------------------------------
    # Parameters: edit these values, then run this file directly.
    # ---------------------------------------------------------------------
    patient_dir = Path("Data/Healthy/Patient_1")
    output_path = Path(
        "third_article/outputs/sensor_derived_severity_with_emg/"
        "cycle_validation_plots/"
        "Healthy_Patient_1_filtered_gyro_z_cycle_validation.png"
    )

    left_count, right_count = create_cycle_validation_plot(
        patient_dir, output_path
    )
    print(f"Saved {output_path}")
    print(
        f"Accepted cycle starts: {left_count} left and {right_count} right."
    )


if __name__ == "__main__":
    main()
