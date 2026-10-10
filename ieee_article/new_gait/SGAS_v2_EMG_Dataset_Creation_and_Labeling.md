# SGAS-v2 Dataset Creation, Severity Labeling and EMG Mapping

## 1. Purpose

This document describes the process implemented in `create_sensor_derived_severity_dataset_with_emg.py`.

The script performs two deliberately separated tasks:

1. It constructs the continuous Sensor-Derived Gait Asymmetry Severity score (SGAS-v2) and the Low, Moderate and High severity labels from raw bilateral shank IMU recordings.
2. After the labels have been created and frozen, it maps previously extracted EMG features to the verified participant identities for later classification.

The EMG features do not participate in the severity-score calculation or class creation. This separation prevents the target labels from being influenced by the predictors later used to classify them.

SGAS-v2 represents relative, sensor-derived gait-asymmetry severity. It is not a clinically validated stroke-severity scale and should not be described as a substitute for Fugl–Meyer, FAC, TUG, Berg Balance Scale, NIHSS or another clinical assessment.

## 2. Required data

### 2.1 Raw bilateral IMU recordings

The expected structure is:

```text
Data/
├── Healthy/
│   ├── Patient_1/
│   └── ...
└── Stroke/
    ├── Patient_1/
    └── ...
```

Every participant directory must contain:

```text
LeftShank-Accelerometer.csv
LeftShank-Gyroscope.csv
RightShank-Accelerometer.csv
RightShank-Gyroscope.csv
```

The raw files must provide an epoch timestamp and x, y and z sensor axes. The reader accepts the timestamp spelling variants included in the recordings and converts the required columns to numeric values. Rows with unusable values are removed, timestamps are sorted and duplicate timestamps are discarded.

### 2.2 Extracted feature table

The mapped feature table is expected at:

```text
features_dataset.csv
```

It is read as a semicolon-delimited file with decimal commas. The table contains previously extracted IMU and EMG features at limb-row level.

The verified participant mapping is frozen as follows:

| Feature-table ID | Raw-data participant |
|---|---|
| 1–15 | `Healthy_Patient_1` to `Healthy_Patient_15` |
| 16–30 | `Stroke_Patient_1` to `Stroke_Patient_15` |

Each feature-table participant must have exactly two rows:

| Cohort | Expected source labels |
|---|---|
| Healthy IDs 1–15 | `(0, 0)` |
| Stroke IDs 16–30 | `(1, 2)` |

These source labels are used only to validate the expected row pair. They are not severity labels and are never exported as classifier predictors.

## 3. Configuration

The default main paths are:

```python
DATA_DIR = Path("Data")
FEATURES_DATASET_PATH = Path("features_dataset.csv")
OUTPUT_DIR = Path("third_article/outputs/sensor_derived_severity_with_emg")
```

The script does not use command-line arguments. Configuration is changed directly in the code.

Important defaults include:

| Parameter | Default | Role |
|---|---:|---|
| Sampling frequency | 100 Hz | Common synchronization grid |
| Low-pass cutoff | 35 Hz | Gyroscope-z filtering for cycle detection |
| Filter order | 2 | Butterworth low-pass filter |
| Maximum interpolation gap | 0.05 s | Rejects grid positions too far from a recorded sample |
| Minimum valid segment | 8 s | Minimum contiguous segment for filtering and cycle detection |
| Minimum peak distance | 0.80 s | Minimum separation between gait-cycle peaks |
| Accepted stride duration | 0.60–2.00 s | Removes implausible cycle intervals |
| Minimum peak height | 5 degrees/s | Lower bound for adaptive peak height |
| Minimum prominence | 3 degrees/s | Lower bound for adaptive peak prominence |
| Minimum cycles per leg | 10 | Participant inclusion requirement |
| Normalized waveform length | 101 points | Common gait-cycle representation |
| Positive-z soft scale | 3.0 | Controls the monotonic asinh compression |
| Stroke split | Median-rank | Transparent Moderate/High division |
| Minimum stroke class size | 5 | Prevents unusably small stroke subgroups |
| Bootstrap iterations | 500 | Split-stability assessment |
| Borderline confidence threshold | 0.70 | Flags unstable Moderate/High assignments |

`SAVE_SYNCHRONISED` and `SAVE_CYCLE_VALIDATION_PLOTS` are enabled by default. `OVERWRITE` is disabled, so an existing non-empty output directory must be handled deliberately before rerunning the script.

## 4. Stage A: validate the extracted feature table

The feature table is checked before processing the raw IMU participants. The script verifies:

- the presence of `ID` and `Label`;
- unique column headings;
- integer participant IDs and labels;
- the exact ID range 1–30;
- two rows for every ID;
- the expected `(0, 0)` or `(1, 2)` row-label pattern;
- recognized IMU or EMG feature prefixes;
- numeric, finite predictor values;
- unique participant keys after mapping.

The source columns are divided into three groups:

1. EMG features from `GMinter`, `RFinter`, `BFinter`, `MGinter`, `TAinter` and `PLinter`.
2. Eligible legacy IMU summaries from gyroscope x/y and accelerometer x/y/z.
3. Prohibited gyroscope-z summaries.

Three source headings use `PLLinter` rather than `PLinter`. These names are standardized to `PLinter` after checking that the corrected names remain unique.

## 5. Stage B: synchronize the four raw IMU streams

For each participant, the script finds the common time interval shared by all four raw files. The start and end are aligned to the 100 Hz grid, producing samples every 10 ms.

Each axis is linearly interpolated onto this grid. A grid position is marked invalid if its nearest recorded sample is more than 50 ms away. The synchronized dataset records separate validity indicators and, specifically, whether both gyroscopes are valid.

The usable bilateral gyroscope fraction is calculated correctly as:

```python
valid_fraction = gyro_valid.mean()
```

This represents the proportion of synchronized samples for which both gyroscope streams are valid.

## 6. Stage C: filter gyroscope-z and detect gait cycles

Only contiguous valid bilateral-gyroscope segments lasting at least eight seconds are filtered. A second-order 35 Hz Butterworth low-pass filter is applied with zero-phase `sosfiltfilt` processing.

Gait cycles are detected independently for the left and right legs.

### 6.1 Automatic polarity handling

For every valid segment, the algorithm compares the upper and lower signal tails with the median. It orients the signal toward whichever tail has the stronger excursion. This permits positive-peak detection without assuming that every sensor was mounted with identical polarity.

The polarity decision is stored for audit purposes but is not used as a classifier predictor.

### 6.2 Adaptive peak threshold

The robust signal spread is estimated as:

```text
robust sigma = 1.4826 × MAD
```

The peak-height threshold is the segment median plus the larger of:

- 5 degrees/s; or
- 1.5 times the robust sigma.

Peak prominence is the larger of:

- 3 degrees/s; or
- 0.75 times the robust sigma.

Peaks must be at least 0.80 seconds apart. Consecutive accepted peaks form candidate gait cycles. A cycle is retained only when:

- its duration is between 0.60 and 2.00 seconds;
- its samples are finite;
- it contains at least four samples;
- its peak-to-peak amplitude is positive.

At least ten valid cycles are required independently for each leg. If this requirement is not met, the participant is excluded and the reason is written to the exclusion table.

### 6.3 Per-leg summaries

For each leg, the script calculates:

- cycle count;
- median stride duration;
- stride-duration coefficient of variation;
- median gyroscope-z cycle peak-to-peak amplitude;
- amplitude coefficient of variation;
- median 101-point time-normalized gait-cycle waveform.

## 7. Stage D: calculate the four fixed asymmetry components

The symmetric difference between left and right measurements is:

```text
D(L, R) = 2 × |L − R| / (|L| + |R| + ε)
```

This definition is invariant to swapping the legs.

The four SGAS-v2 components are:

| Component | Definition |
|---|---|
| Temporal asymmetry | Symmetric difference between left and right median stride durations |
| Amplitude asymmetry | Symmetric difference between left and right median gyroscope-z cycle amplitudes |
| Variability asymmetry | Mean of the symmetric differences for stride-duration CV and amplitude CV |
| Waveform dissimilarity | `1 − |correlation|` between the two median time-normalized gyroscope-z waveforms |

Waveform dissimilarity is restricted to the interval 0–1. A value near zero means that the bilateral waveform shapes are similar after allowing for polarity reversal. A larger value means greater shape dissimilarity.

No affected-side identity is required. All components are absolute bilateral comparisons and therefore remain unchanged when left and right are exchanged.

## 8. Stage E: healthy-referenced standardization

Each component is standardized using only valid healthy participants.

The primary center and scale are:

```text
center = healthy median
scale  = 1.4826 × healthy MAD
z      = (participant value − center) / scale
```

If the MAD scale is unusable, the script falls back sequentially to:

1. `IQR / 1.349`;
2. sample standard deviation;
3. unit scale.

Only positive deviations from the healthy reference contribute to severity. Negative robust z-scores are set to zero before transformation.

## 9. Stage F: construct the continuous SGAS-v2 score

For every component, the contribution is:

```text
contribution = asinh(max(z, 0) / 3) / asinh(1)
```

The asinh transform is monotonic and unbounded. It reduces domination by extreme values without using a hard cap and without changing participant order.

The final continuous score is the equal-weight arithmetic mean of the four transformed contributions:

```text
SGAS-v2 score = mean(
    temporal contribution,
    amplitude contribution,
    variability contribution,
    waveform contribution
)
```

Each component therefore has a weight of 0.25.

## 10. Stage G: create the three severity classes

### 10.1 Low/reference class

All valid healthy participants are assigned:

```text
Class 0 — Low/reference asymmetry (healthy)
```

A healthy-reference cutoff is also calculated, using the maximum leave-one-out healthy score by default. This cutoff is diagnostic only. It reports whether a score exceeds the healthy reference range but does not decide whether a known stroke participant belongs to a stroke class.

### 10.2 Moderate and High stroke classes

Only the known stroke cohort is divided by its continuous SGAS-v2 score:

```text
Class 1 — Moderate sensor-derived asymmetry (stroke)
Class 2 — High sensor-derived asymmetry (stroke)
```

The default median-rank procedure:

1. sorts stroke scores;
2. considers only boundaries that keep at least five participants in each group;
3. prevents a boundary from separating identical scores;
4. selects the valid split closest to the sample median;
5. places the cutoff halfway between the largest lower-group score and the smallest upper-group score.

The code also supports a constrained one-dimensional within-group SSE split as a sensitivity option. The default median-rank split is preferred because it is transparent and does not claim that a natural clinical cluster has been discovered.

The Moderate/High cutoff is relative to this stroke sample. It is not an externally validated clinical threshold.

## 11. Stage H: assess boundary stability

The stroke scores are resampled 500 times with replacement. The Moderate/High boundary is recalculated for each successful bootstrap sample.

For each stroke participant:

```text
P(High) = proportion of bootstrap boundaries below the participant's score
assignment confidence = max(P(High), 1 − P(High))
```

An assignment is flagged as borderline when its confidence is below 0.70. These flags support a predeclared sensitivity analysis; they do not silently change or delete the original label.

## 12. Stage I: cycle-detection quality control

The automatic QC table flags participants with:

- fewer than 15 cycles on the smaller-count leg;
- a left/right cycle-count ratio greater than 1.50;
- stride-duration CV greater than 0.25 on either leg;
- more than 10% of cycles close to the accepted duration limits;
- bilateral gyroscope valid fraction below 0.80.

A QC flag is not an automatic exclusion. It indicates that the participant’s validation plot should be reviewed. Manual decision and note fields are retained for documented review.

The cycle-validation plots display filtered left and right gyroscope-z signals with accepted cycle starts. They are audit evidence and do not tune the detector or alter labels.

## 13. Stage J: freeze and fingerprint the score definition

The frozen score definition records:

- SGAS version (`SGAS-v2.0`);
- component order and weights;
- healthy median and scale for each component;
- standardization and transformation rules;
- absence of a hard cap;
- class rules;
- frozen Moderate/High cutoff;
- minimum class size and borderline-confidence rule.

The canonical JSON definition is hashed with SHA-256. The version and hash are copied into every severity-dataset row. The classifier later verifies this fingerprint before any model is fitted.

## 14. Stage K: construct order-invariant EMG predictors

The source feature table does not provide a trusted left/right column for the two rows belonging to each participant. Therefore, the code does not guess row laterality.

For every eligible source feature with two row values `v1` and `v2`, it creates:

```text
bilateral mean                = (v1 + v2) / 2
absolute bilateral difference = |v1 − v2|
```

Both transformations are unchanged if the two source rows are reversed.

For the supplied feature table, the resulting feature groups are:

| Source group | Source features | Derived participant predictors |
|---|---:|---:|
| Six-muscle EMG | 156 | 312 |
| Eligible legacy IMU x/y and accelerometer features | 85 | 170 |
| Legacy gyroscope-z | 17 | 0; deliberately excluded |

Gyroscope-z source features are excluded because gyroscope-z produced the SGAS-v2 target. Including them in the primary classifier would encourage target reconstruction rather than independent severity classification.

## 15. Stage L: merge severity labels and mapped predictors

The script creates a one-to-one mapping QC table containing:

- participants present in both datasets;
- feature-only participants without a valid SGAS-v2 record;
- severity-only participants without a mapped feature record;
- complete-case eligibility for EMG analysis.

For the current frozen study data, the expected outcome is:

- 30 valid SGAS-v2 participants;
- 30 mapped feature-table participants;
- 28 participants in the intersection;
- `Healthy_Patient_10` and `Stroke_Patient_8` are feature-only because their raw-IMU records were excluded;
- `Healthy_Patient_16` and `Stroke_Patient_16` are severity-only because the verified feature mapping ends at Patient 15.

This gives an EMG complete-case classification cohort of 28 participants. The classifier performs a matched IMU-only analysis on these same 28 participants so modality comparisons are not confounded by different participant membership.

## 16. Main outputs

The output directory is:

```text
third_article/outputs/sensor_derived_severity_with_emg/
```

| Output | Purpose |
|---|---|
| `sensor_derived_gait_asymmetry_severity.csv` | Frozen participant-level score and severity labels |
| `gait_cycle_metrics.csv` | Accepted gait-cycle boundaries and cycle audit information |
| `healthy_reference_statistics.csv` | Healthy median, scale and fallback method for each component |
| `cycle_detection_quality_control.csv` | Automatic flags plus manual-review fields |
| `cycle_validation_plots/` | Participant-level visual cycle audits |
| `excluded_participants.csv` | Participants excluded during processing and the exact reason |
| `frozen_score_definition.json` | Versioned and SHA-256-fingerprinted score definition |
| `severity_method_summary.json` | Configuration, thresholds, interpretation and output manifest |
| `synchronised/` | Synchronized participant signals when enabled |
| `participant_level_mapped_external_features.csv` | Classifier-safe EMG and eligible legacy IMU predictors without source ID/Label |
| `feature_mapping_quality_control.csv` | Participant mapping and complete-case audit |
| `sensor_derived_gait_asymmetry_severity_with_emg.csv` | All severity rows with EMG where available |
| `sensor_derived_gait_asymmetry_severity_with_emg_complete_cases.csv` | Matched SGAS-v2 and EMG participants only |
| `sensor_derived_gait_asymmetry_severity_with_emg_and_eligible_legacy_imu.csv` | Severity, EMG and eligible non-z legacy IMU features |

## 17. Running the script

From the GaitRehab project root:

```bash
python3 create_sensor_derived_severity_dataset_with_emg.py
```

Run this script before the classifier. If the output directory already contains results and `OVERWRITE` remains `False`, the script stops rather than silently replacing the existing analysis.

## 18. Interpretation and limitations

- The classes describe sensor-derived gait asymmetry, not global neurological impairment.
- The Moderate/High boundary is sample-relative and must not be presented as a universal clinical cutoff.
- Healthy membership defines the Low/reference superclass; the healthy cutoff is diagnostic rather than a class gate.
- EMG is available only as extracted features. Raw-signal preprocessing, synchronization and gait-cycle alignment of EMG cannot be independently verified in this version.
- The mapped EMG cohort contains fewer participants than the complete IMU cohort.
- The large number of EMG predictors relative to the participant count makes nested feature selection and regularization essential.
- Borderline flags, QC flags and exclusions must be reported transparently rather than adjusted after observing classifier performance.

## 19. Core methodological principle

The target and predictors have separate roles:

```text
Raw bilateral gyroscope-z
    → gait-cycle asymmetry components
    → healthy-referenced continuous SGAS-v2 score
    → frozen Low / Moderate / High labels

Raw accelerometer and gyroscope-x/y + mapped EMG
    → classifier predictors added only after label freezing
```

This separation is what makes the subsequent multimodal classification analysis defensible.
