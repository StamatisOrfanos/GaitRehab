# SGAS-v2 Dataset Creation and Stroke Severity Classification

## 1. Purpose

`create_sensor_derived_severity_dataset(1).py` creates a participant-level dataset from raw bilateral shank IMU recordings. The resulting labels represent **relative sensor-derived gait-asymmetry severity**. They are not clinical stroke-severity diagnoses because no clinical score, timed walking-speed measurement, or confirmed affected-side annotation is available.

The final structure is hierarchical:

1. **Class 0 — Low/reference asymmetry:** valid healthy participants.
2. **Class 1 — Moderate sensor-derived asymmetry:** the lower-scoring subgroup within the stroke cohort.
3. **Class 2 — High sensor-derived asymmetry:** the higher-scoring subgroup within the stroke cohort.

The algorithm first constructs and freezes a continuous asymmetry score. It then divides only the stroke cohort into two relative severity groups. This prevents an extreme observation from becoming a one-person High class, which occurred in the earlier GMM-based version.

## 2. Expected directory structure

The script reads:

```text
Data/
├── Healthy/
│   ├── Patient_1/
│   └── ...
└── Stroke/
    ├── Patient_1/
    └── ...
```

Each participant folder must contain:

```text
LeftShank-Accelerometer.csv
LeftShank-Gyroscope.csv
RightShank-Accelerometer.csv
RightShank-Gyroscope.csv
```

The cohort name becomes part of the participant key. Therefore, `Healthy_Patient_1` and `Stroke_Patient_1` remain distinct.

The script uses editable constants rather than command-line arguments:

```python
DATA_DIR = Path("Data")
OUTPUT_DIR = Path("third_article/outputs/sensor_derived_severity")
SAVE_SYNCHRONISED = True
SAVE_CYCLE_VALIDATION_PLOTS = True
OVERWRITE = False
```

Set `OVERWRITE = True` when intentionally replacing files in an existing output directory.

## 3. Overall workflow

The processing sequence is:

1. Discover participant folders and verify the four required sensor files.
2. Read and clean the timestamped IMU signals.
3. Synchronize all four sensor streams onto one 100 Hz time grid.
4. Filter left and right gyroscope-z signals.
5. Detect gait cycles independently on each leg.
6. Save visual and tabular cycle-detection quality-control information.
7. Summarize the accepted cycles for each leg.
8. Calculate four bilateral asymmetry components.
9. Standardize the components relative to the healthy cohort.
10. Apply a non-saturating transformation and calculate a continuous score.
11. Retain the healthy reference boundary as a diagnostic measurement.
12. Assign healthy participants to Class 0.
13. Divide the stroke cohort into Moderate and High relative severity groups.
14. Bootstrap the stroke threshold to quantify assignment stability.
15. Save a frozen, versioned score definition and all audit outputs.

## 4. Reading and cleaning the raw files

The code identifies the epoch-time column and the x, y, and z sensor axes despite minor naming differences. Values are converted to numeric form, invalid rows are removed, timestamps are sorted, and duplicate timestamps are removed.

A file with fewer than ten valid rows is rejected. Missing files are reported and the associated participant folder is skipped.

This stage ensures that synchronization does not operate on malformed or duplicated timestamps.

## 5. Synchronization

The four recordings can start and stop at slightly different times. The script therefore calculates:

- the latest starting timestamp across the four streams;
- the earliest ending timestamp across the four streams.

Only this common interval is retained. A uniform 100 Hz grid, corresponding to one point every 10 ms, is created over the interval.

Every sensor axis is interpolated onto the grid. An interpolated value is considered valid only when the nearest real measurement is within 0.05 seconds. Larger gaps remain invalid rather than being filled with potentially misleading values.

The synchronized table contains:

- the common epoch time;
- elapsed time from the common start;
- all twelve synchronized sensor axes;
- validity indicators for each stream;
- an all-streams-valid indicator;
- a both-gyroscopes-valid indicator.

Severity construction uses portions in which both gyroscopes are valid. This prevents timestamp offsets or missing samples from being interpreted as biological asymmetry.

## 6. Gyroscope filtering

Left and right gyroscope-z signals are filtered using a second-order Butterworth low-pass filter with a 35 Hz cutoff. Forward-backward filtering is used, which avoids a systematic phase shift.

Filtering is performed separately within valid contiguous segments of at least eight seconds. Invalid gaps are not bridged. This prevents the filter from creating artificial transitions across missing data.

## 7. Gait-cycle detection

Cycles are detected independently for each leg.

For every valid signal segment, the algorithm:

1. Examines the 5th, 50th, and 95th percentiles.
2. Determines whether the main gait peaks appear in the positive or negative signal direction.
3. Reorients the signal internally for peak detection when necessary.
4. Estimates a robust signal scale using `1.4826 × MAD`.
5. Uses an adaptive peak-height requirement equal to the segment median plus the larger of:
   - 5 deg/s; or
   - 1.5 robust standard deviations.
6. Uses a peak-prominence requirement equal to the larger of:
   - 3 deg/s; or
   - 0.75 robust standard deviations.
7. Requires at least 0.80 seconds between detected peaks.
8. Creates cycles from consecutive accepted peaks.
9. Accepts only cycles lasting between 0.60 and 2.00 seconds.
10. Rejects incomplete, non-finite, or zero-amplitude cycles.

Every accepted cycle is resampled to 101 points. At least ten valid cycles are required on each leg. A participant failing this requirement is listed in `excluded_participants.csv` with the exact reason.

## 8. Cycle-detection validation

The revised script does not assume that a successful detector is automatically correct. It creates two forms of quality-control evidence.

### 8.1 Validation plots

For every participant, `cycle_validation_plots/` contains a figure showing:

- filtered left gyroscope-z;
- filtered right gyroscope-z;
- accepted cycle-start locations;
- the number of accepted cycles per side;
- a processing note when cycle extraction fails.

These plots are intended for manual review. They are particularly important for excluded participants, low-cycle recordings, recordings with unequal left/right cycle counts, and participants close to the severity threshold.

### 8.2 QC table

`cycle_detection_quality_control.csv` reports:

- left and right cycle counts;
- left/right cycle-count ratio;
- proportion of the common recording with valid bilateral gyroscope data;
- proportion of cycles close to the accepted duration limits;
- automatic warning flags;
- empty fields for the manual decision and reviewer notes.

The current warnings identify:

- fewer than 15 cycles on either leg;
- left/right cycle-count ratio greater than 1.50;
- stride-duration coefficient of variation greater than 0.25;
- more than 10% of cycles close to the 0.60 or 2.00 second limits;
- less than 80% valid bilateral gyroscope coverage.

An automatic warning never changes a label and never excludes a participant. It only determines which records should be reviewed first. Manual decisions should be documented before final analysis.

## 9. Per-leg summaries

For each leg, the accepted cycles are reduced to:

- number of cycles;
- median stride duration;
- stride-duration coefficient of variation;
- median cycle peak-to-peak gyroscope-z amplitude;
- cycle-amplitude coefficient of variation;
- median time-normalized waveform template.

Medians reduce the influence of isolated abnormal cycles. Coefficients of variation describe relative dispersion rather than absolute scale.

## 10. Four asymmetry components

For scalar left and right measurements, the code uses the symmetric absolute difference:

\[
A(L,R)=\frac{2|L-R|}{|L|+|R|+\varepsilon}.
\]

The value is unchanged when left and right are exchanged. Therefore, the method does not require affected-side information.

The four participant-level components are:

### 10.1 Temporal asymmetry

The symmetric difference between left and right median stride durations.

### 10.2 Amplitude asymmetry

The symmetric difference between left and right median gyroscope-z cycle amplitudes.

### 10.3 Variability asymmetry

The average of:

- the symmetric difference between left and right stride-duration coefficients of variation;
- the symmetric difference between left and right cycle-amplitude coefficients of variation.

### 10.4 Waveform dissimilarity

The left and right median cycle templates are compared using:

\[
D_{wave}=1-|r|,
\]

where \(r\) is their Pearson correlation. A value near zero indicates similar waveform shapes, while a larger value indicates greater bilateral dissimilarity.

The absolute correlation makes the measurement insensitive to a complete sign reversal caused by sensor orientation.

## 11. Healthy-referenced standardization

Each component is standardized using only valid healthy participants:

\[
z_j=\frac{x_j-\operatorname{median}(x_{j,healthy})}
{1.4826\times\operatorname{MAD}(x_{j,healthy})}.
\]

If the MAD scale is zero, the code attempts the following fallbacks:

1. IQR divided by 1.349;
2. conventional standard deviation;
3. unit scale.

Negative robust z-scores are set to zero because a value below the healthy median is not treated as additional asymmetry severity.

## 12. Corrected component scaling

The previous algorithm clipped every positive component z-score at 10. This caused the waveform contribution to equal 10 for every valid stroke participant, destroying within-stroke ordering.

SGAS-v2 replaces that hard cap with:

\[
c_j=\frac{\operatorname{asinh}\left(\max(z_j,0)/3\right)}
{\operatorname{asinh}(1)}.
\]

This transformation has four useful properties:

- negative deviations still contribute zero;
- small and moderate deviations remain distinguishable;
- extreme deviations are compressed rather than dominating the score;
- no hard ceiling is applied, so extreme participants retain their ordering.

A robust z-score of 3 produces a transformed contribution of 1. Larger values can exceed 1 but grow progressively more slowly.

## 13. Continuous severity score

The continuous Sensor-Derived Gait Asymmetry Severity score is the equal-weight mean:

\[
S=\frac{c_{temporal}+c_{amplitude}+c_{variability}+c_{waveform}}{4}.
\]

All four components currently receive weight 0.25. The score is calculated before class construction and represents the primary severity ordering.

The score definition is identified as `SGAS-v2.0`.

## 14. Healthy reference boundary

The script still calculates a healthy reference boundary for diagnostic interpretation.

Each healthy participant is temporarily held out. Healthy medians and scales are recalculated from the remaining healthy participants, and the held-out participant is scored against them. The maximum leave-one-out healthy score is used as the healthy reference envelope when `healthy_cutoff_method="maximum"`.

Importantly, this boundary no longer controls the stroke subclass assignment. A confirmed stroke participant is always assigned to one of the two stroke groups, even when their asymmetry score overlaps the healthy distribution.

This reflects the hierarchical study design:

- cohort membership defines Healthy versus Stroke;
- the continuous score defines relative severity within Stroke.

## 15. Creating two classes inside the stroke cohort

Only stroke scores are considered when constructing Classes 1 and 2.

### 15.1 Primary method: median-rank split

The default method is `median_rank`:

1. Sort all valid stroke participants by their continuous score.
2. Consider only boundaries that place at least five participants on each side.
3. Select the valid split closest to the centre of the ordered stroke cohort.
4. Place the threshold halfway between the largest lower-group score and the smallest upper-group score.
5. Assign stroke scores at or below the threshold to Class 1.
6. Assign stroke scores above the threshold to Class 2.

For 15 valid stroke participants, the expected division is 7 versus 8. This is transparent and statistically usable, but it is sample-relative. It does not prove that two natural clinical subtypes exist.

### 15.2 Optional sensitivity method: constrained SSE split

The code retains `constrained_sse` as an optional sensitivity analysis. It evaluates all ordered splits satisfying the minimum class size and selects the boundary with the smallest total within-group sum of squared deviations.

This method can reveal a possible data-driven separation, but it should not replace the median-rank primary analysis unless it is stable and scientifically defensible. The earlier unconstrained GMM utilities remain in the file for historical sensitivity analysis but no longer control the primary labels.

## 16. Bootstrap stability

The stroke scores are resampled with replacement 500 times. A new permitted stroke threshold is calculated for every successful bootstrap sample.

The script reports:

- number of successful bootstrap iterations;
- 2.5th and 97.5th percentiles of the threshold distribution;
- probability that each stroke participant is classified as High;
- assignment confidence, calculated as the larger of the High and Moderate probabilities;
- a borderline indicator when confidence is below 0.70.

Borderline participants are not automatically removed. They should remain in the primary analysis and can be excluded or treated probabilistically in a predeclared sensitivity analysis.

Testing the revised algorithm on the previously generated participant-level measurements produced:

| Group | Participants |
|---|---:|
| Healthy/reference | 15 |
| Moderate stroke asymmetry | 7 |
| High stroke asymmetry | 8 |

The corresponding test threshold was approximately 2.0236, with a bootstrap interval of approximately 1.5378–2.3164. These values must be recalculated and confirmed by running the complete revised raw-data pipeline.

## 17. Freezing the score definition

`frozen_score_definition.json` records:

- score version;
- component order;
- equal component weights;
- healthy median and scale for every component;
- standardization formula;
- positive-z transformation;
- soft-scale value;
- absence of a hard cap;
- stroke split method;
- minimum stroke class size;
- frozen stroke threshold;
- minimum bootstrap confidence;
- interpretation statement.

The definition receives a SHA-256 fingerprint. The version and fingerprint are also written into the participant-level severity dataset. This makes later datasets auditable and allows the exact score definition to be identified.

The formula, weights, reference values, and threshold must not be altered after classifier results are examined. Changing them in response to classification accuracy would reintroduce model-selection bias at the label-definition stage.

## 18. Preventing circular classification

The labels are constructed from bilateral gyroscope-z cycle measurements. A classifier trained on those same measurements could simply reconstruct the label formula.

Therefore, the primary classification experiment must exclude the exact gyroscope-z variables used to define:

- cycle timing;
- cycle amplitude;
- variability;
- waveform dissimilarity.

The primary predictor set may use accelerometer features, gyroscope x/y features, and available EMG features. A model containing the label-defining gyroscope-z variables may be reported only as an explicitly labelled index-reconstruction sensitivity analysis.

Nested subject-level validation remains necessary:

- outer loop: leave one subject out for final evaluation;
- inner loop: model, feature, hyperparameter, and sensor-set selection using only outer-training subjects.

## 19. Main outputs

| Output | Purpose |
|---|---|
| `sensor_derived_gait_asymmetry_severity.csv` | Participant summaries, components, scores, stability information, and final classes. |
| `gait_cycle_metrics.csv` | One audit row per accepted cycle. |
| `healthy_reference_statistics.csv` | Healthy medians, scales, and transformation definition. |
| `cycle_detection_quality_control.csv` | Automatic QC indicators and fields for manual review. |
| `cycle_validation_plots/*.png` | Visual inspection of filtered signals and accepted cycle starts. |
| `frozen_score_definition.json` | Complete versioned score and class definition. |
| `severity_method_summary.json` | Configuration, thresholds, class counts, outputs, and circularity restrictions. |
| `excluded_participants.csv` | Explicit exclusion reasons when exclusions occur. |
| `synchronised/*.csv` | Synchronized signals and validity indicators when enabled. |

## 20. Interpretation and reporting language

Recommended wording:

> Participants were categorized using a versioned sensor-derived gait-asymmetry index constructed from bilateral shank gyroscope recordings. Healthy participants formed the reference class. Stroke participants were ordered by the continuous index and divided into lower- and higher-asymmetry groups using a predeclared median-rank threshold with minimum subgroup-size constraints. The resulting categories represent relative sensor-derived asymmetry within the study sample and are not clinically validated stroke-severity grades.

Avoid wording that implies:

- a clinical diagnosis;
- a validated functional-impairment threshold;
- confirmed mild, moderate, or severe neurological stroke;
- discovery of natural clinical clusters;
- independence from the sensors used to construct the labels.

## 21. Required checks before machine-learning classification

Before freezing the final dataset:

1. Review every cycle-validation plot.
2. Document the decision for every automatic QC flag.
3. Review all exclusions using synchronized raw signals.
4. Confirm that the revised contributions no longer saturate across a cohort.
5. Confirm the expected class sizes.
6. Report the bootstrap threshold interval and borderline participants.
7. Save the frozen definition and its fingerprint.
8. Do not modify the labels after comparing classifier performance.
9. Exclude label-defining gyroscope-z variables from the primary classifier.
10. Use subject-level nested validation for all model-selection decisions.

## 22. Final methodological position

SGAS-v2 creates two meaningful **relative** classes within the stroke superclass without claiming unsupported clinical thresholds. Its strongest features are an auditable raw-data pipeline, non-saturating continuous scoring, transparent class construction, minimum group-size protection, explicit bootstrap uncertainty, cycle-level quality control, and a frozen score definition.

The method is appropriate for investigating multiclass sensor-derived gait-asymmetry classification. Clinical validation will still require an independent clinical outcome or functional assessment in a future cohort.
