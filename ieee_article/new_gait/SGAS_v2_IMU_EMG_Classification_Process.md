# SGAS-v2 IMU and EMG Classification Process

## 1. Purpose

This document describes the process implemented in `classify_sgas_v2_nested_loso_with_emg.py`.

The classifier predicts the three frozen SGAS-v2 classes:

| Class ID | Interpretation |
|---:|---|
| 0 | Low/reference asymmetry — healthy |
| 1 | Moderate sensor-derived asymmetry — stroke |
| 2 | High sensor-derived asymmetry — stroke |

The script never recalculates or modifies the severity labels. It verifies the previously frozen definition, builds eligible predictor sets and evaluates classification using nested subject-level validation.

The analysis answers two related questions:

1. How reliably can eligible IMU and EMG predictors classify the frozen SGAS-v2 severity groups?
2. Does mapped EMG add useful information beyond cycle-aligned IMU predictors when evaluated on exactly the same participants?

The outcome remains classification of relative sensor-derived gait-asymmetry severity, not clinical stroke-severity diagnosis.

## 2. Required execution order

The dataset-creation script must run first:

```bash
python3 create_sensor_derived_severity_dataset_with_emg.py
python3 classify_sgas_v2_nested_loso_with_emg.py
```

The classifier expects:

```text
third_article/outputs/sensor_derived_severity_with_emg/
├── sensor_derived_gait_asymmetry_severity.csv
├── frozen_score_definition.json
├── gait_cycle_metrics.csv
└── participant_level_mapped_external_features.csv
```

It also reads the original bilateral shank IMU recordings under `Data/Healthy` and `Data/Stroke`.

## 3. Main configuration

The default output directory is:

```text
third_article/outputs/sgas_v2_nested_classification_with_emg/
```

The principal validation settings are:

| Parameter | Default |
|---|---:|
| Maximum inner folds | 5 |
| Random seed | 20261004 |
| Bootstrap iterations | 2,000 |
| Parallel jobs | All available cores (`-1`) |
| Primary inner-selection metric | Macro-F1 |
| Candidate selected-feature counts | 5 and 10 |

The output directory must be empty unless `OVERWRITE` is deliberately enabled.

## 4. Frozen-label verification

Before extracting predictors or fitting a model, the script verifies the modeling target.

It checks:

- required label columns;
- unique participant keys;
- presence of all three class IDs;
- agreement between every row and the frozen score version;
- agreement between every row and the frozen score-definition SHA-256 hash;
- agreement with the exact predeclared participant-label assignment hash.

The configured assignment hash protects the participant-to-class mapping independently of CSV row order or formatting. If any label changes, the script stops.

A copy of the exact labels used in the run is saved as `frozen_labels_used.csv`, accompanied by a manifest and a snapshot hash.

## 5. Frozen gait-cycle boundaries

The classifier reads only these fields from `gait_cycle_metrics.csv`:

- participant key;
- side;
- cycle start epoch;
- cycle end epoch.

The gyroscope-z signal values are not loaded into the predictor matrix. Neither cycle duration, gyroscope-z amplitude, cycle count nor detected polarity is used as a predictor.

The frozen timestamps serve only as segmentation markers that align eligible accelerometer and gyroscope-x/y signals to the previously validated gait cycles.

## 6. Cycle-aligned IMU predictor extraction

### 6.1 Signal synchronization and filtering

The four raw streams are synchronized at 100 Hz over their common interval. Samples more than 50 ms from the nearest observation are invalid.

The classifier uses a second-order 20 Hz Butterworth low-pass filter for eligible predictor signals. Filtering is applied only to contiguous valid segments of at least four seconds.

The eligible raw axes are:

- accelerometer x/y/z for both shanks;
- gyroscope x/y for both shanks.

Every gyroscope-z axis is excluded.

### 6.2 Vector magnitudes

The script calculates cycle-aligned magnitudes separately for:

- three-axis accelerometer signals;
- two-axis gyroscope x/y signals.

Using magnitudes reduces dependence on sensor orientation and creates side-comparable signals.

### 6.3 Per-cycle scalar descriptors

For each usable cycle, the script calculates:

- mean;
- standard deviation;
- median;
- interquartile range;
- range;
- root mean square;
- mean absolute value;
- peak absolute value.

Across cycles, each descriptor is summarized by its median and interquartile range.

For every left/right summary pair, the participant-level feature representation contains:

```text
bilateral mean
absolute bilateral difference
symmetric bilateral difference
```

This produces side-invariant predictors that do not require an affected-side label.

### 6.4 Optional waveform predictors

Each eligible cycle can be resampled to 101 points. The code creates median standardized left and right shape templates and derives:

- mean absolute template difference;
- RMS template difference;
- correlation dissimilarity;
- five DCT coefficients from the bilateral average template;
- five DCT coefficients from the absolute difference template.

These waveform predictors are excluded from the primary model and included only in the declared waveform sensitivity analysis.

## 7. EMG predictor loading

The classifier reads the participant-level mapped predictor table produced by the dataset script. It expects:

- exactly 30 mapped participant rows;
- one unique row per participant key;
- predictor names beginning only with `emg__` or `legacy_imu__`;
- numeric, finite values;
- no source ID, source Label or audit metadata.

The primary EMG predictors describe six muscles:

- gluteus medius (`GMinter`);
- rectus femoris (`RFinter`);
- biceps femoris (`BFinter`);
- medial gastrocnemius (`MGinter`);
- tibialis anterior (`TAinter`);
- peroneus longus (`PLinter`).

Each source feature is represented by a bilateral mean and absolute bilateral difference. The current mapped table supplies 312 EMG predictors.

The table also supplies 170 eligible legacy IMU predictors. These are reserved for an optional sensitivity analysis because they are whole-recording summaries rather than the preferred cycle-aligned raw-IMU features.

## 8. Matched multimodal cohort

The full frozen SGAS-v2 dataset contains 30 participants. The EMG mapping and frozen labels intersect for 28 participants.

The expected matched class distribution is:

| Class | Participants |
|---|---:|
| Low/reference | 14 |
| Moderate | 6 |
| High | 8 |

The code requires exactly 28 matched participants and verifies that all three classes remain represented.

An `emg_mapping_quality_control.csv` file identifies:

- 28 matched participants;
- two frozen-label-only participants;
- two mapped-feature-only participants.

The crucial comparison rule is that matched IMU-only, EMG-only and IMU+EMG experiments all use the same 28 participants and therefore the same 28 outer LOSO test cases.

The original 30-participant IMU result remains useful as the full-cohort primary analysis, but it must not be directly interpreted as a controlled modality comparison with the 28-participant EMG models.

## 9. Leakage barriers

The classifier rejects predictor names associated with:

- gyroscope-z;
- SGAS-v2 component values;
- robust z-scores and severity contributions;
- continuous SGAS-v2 score;
- severity class fields;
- stroke split, bootstrap probability or assignment confidence;
- mapped source ID and row labels.

Only these predictor prefixes are accepted:

```text
acc__
gyro_xy__
emg__
legacy_imu__
```

Participant cohort and folder names remain metadata. They are not provided to the estimators.

All data-dependent preprocessing is performed inside the inner-training pipeline:

1. median imputation;
2. standard scaling;
3. ANOVA `SelectKBest` feature selection;
4. model fitting.

Consequently, the held-out outer participant cannot influence imputation values, scaling parameters, selected predictors, model selection or hyperparameter selection.

## 10. Classification experiments

The enabled analyses are controlled by independent Boolean constants.

### 10.1 Full-cohort primary IMU analysis

```text
primary_direct_three_class
```

Uses all 30 frozen participants and non-waveform cycle-aligned IMU features. Inner validation considers accelerometer, gyroscope-x/y and their combination.

### 10.2 Matched IMU-only baseline

```text
matched_imu_only_three_class
```

Repeats the non-waveform IMU analysis on only the 28 EMG-matched participants. This is the correct baseline for evaluating whether EMG adds information.

### 10.3 EMG-only analysis

```text
matched_emg_only_three_class
```

Uses the 312 mapped EMG predictors on the same 28 participants.

### 10.4 IMU+EMG analysis

```text
matched_imu_emg_three_class
```

Combines non-waveform cycle-aligned IMU predictors with mapped EMG predictors for the same 28 participants.

The combined candidate pool is allowed to select the most informative 5 or 10 predictors within each outer-training fold. Therefore, selected features may come from either or both modalities without exposing the held-out participant.

### 10.5 Legacy IMU sensitivity

```text
sensitivity_matched_imu_emg_with_legacy_imu
```

Adds the 170 eligible non-z legacy IMU summaries to cycle-aligned IMU and EMG. This analysis is disabled by default:

```python
RUN_LEGACY_IMU_FEATURE_SENSITIVITY = False
```

If enabled, it should be reported as a sensitivity analysis rather than the primary multimodal result.

### 10.6 Borderline-label sensitivity

```text
sensitivity_excluding_borderline
```

Removes only the stroke participants predeclared as borderline during label creation. Their status is fixed before model evaluation.

### 10.7 Waveform sensitivity

```text
sensitivity_with_eligible_waveform_predictors
```

Adds the eligible accelerometer and gyroscope-x/y waveform-shape predictors omitted from the primary experiment.

### 10.8 Exploratory hierarchical classifier

```text
exploratory_hierarchical_classifier
```

Uses two nested models:

1. Healthy versus Stroke.
2. Moderate versus High conditional on Stroke.

The final probabilities are:

```text
P(Low)      = P(Healthy)
P(Moderate) = P(Stroke) × P(Moderate | Stroke)
P(High)     = P(Stroke) × P(High | Stroke)
```

The probabilities are normalized before selecting the final class. This is exploratory because errors from the first stage propagate to the second stage.

### 10.9 Continuous-score regression

```text
continuous_score_regression
```

This complementary analysis predicts the frozen continuous SGAS-v2 score from eligible non-z IMU predictors. Ridge Regression and Linear SVR are compared using nested LOSO. Feature selection uses `f_regression`, and inner selection minimizes MAE.

The regression reports:

- overall MAE;
- stroke-only MAE;
- Spearman correlation and p-value;
- bootstrap confidence intervals;
- true-versus-predicted and residual plots.

The current regression analysis is IMU-based; the new EMG extension focuses on three-class classification.

## 11. Evaluated classifiers

Every direct classification analysis evaluates eleven common model families:

| Model | Compact tuning choices |
|---|---|
| Logistic Regression | `C = 0.1, 1.0`; balanced class weights |
| Linear SVM | `C = 0.1, 1.0`; probabilities enabled; balanced class weights |
| RBF SVM | `C = 0.1, 1.0`; `gamma = scale, 0.1`; probabilities enabled |
| k-Nearest Neighbours | 3 or 5 neighbours; uniform or distance weights |
| Gaussian Naive Bayes | Two variance-smoothing values |
| Shrinkage LDA | Automatic or 0.5 shrinkage |
| Decision Tree | Depth 2 or unlimited; leaf size 1 or 3 |
| Random Forest | 300 trees; depth 3 or unlimited; leaf size 1 or 2 |
| Extra Trees | 300 trees; depth 3 or unlimited; leaf size 1 or 2 |
| Gradient Boosting | 50 or 100 estimators; learning rate 0.05 or 0.1 |
| AdaBoost | 50 or 100 estimators; learning rate 0.1 or 0.5 |

Feature counts of 5 and 10 are included in the grid when valid for the candidate predictor set.

## 12. Nested subject-level validation

### 12.1 Outer LOSO

For every outer fold:

1. one complete participant is held out;
2. every modeling decision is made using the remaining participants;
3. the selected pipeline predicts the held-out participant once;
4. the held-out prediction and class probabilities are stored.

After every participant has been held out, the outer predictions are pooled. Final performance is calculated from this pooled vector. The script does not average metrics calculated on one-participant folds.

### 12.2 Inner grouped validation

The outer-training participants are divided using shuffled `StratifiedGroupKFold`. The number of folds is the smaller of:

- five; or
- the smallest class count in the current outer-training set.

At least two folds are required. Participant key is used as the group identifier.

The inner loop selects:

- permitted sensor or modality set;
- classifier family;
- model hyperparameters;
- 5 versus 10 selected predictors.

Macro-F1 is the primary inner metric. Balanced accuracy is the secondary tie-breaking metric.

### 12.3 Unbiased comparison of all models

In addition to selecting one best overall pipeline per outer fold, the code retains the best inner configuration for each of the eleven model families. Every family therefore receives exactly one held-out outer prediction per participant.

The final model ranking is based on pooled outer-LOSO macro-F1, followed by balanced accuracy and accuracy. It is not based on the inner-validation score.

## 13. Probability handling

SVM classifiers are constructed with probability estimation enabled. For every classifier, predicted probability columns are explicitly aligned to the expected class order:

```text
0 → Low/reference
1 → Moderate
2 → High
```

Missing class positions are initialized safely, rows are checked for positive probability mass, and probabilities are normalized. The script also records whether `predict()` disagrees with the probability argmax.

## 14. Evaluation metrics

The primary pooled outer-LOSO metrics are:

- accuracy;
- balanced accuracy;
- macro-F1.

The analysis also reports:

- per-class precision;
- per-class recall;
- per-class F1;
- per-class support;
- confusion matrix;
- participant-level class probabilities;
- one-vs-rest ROC AUC;
- one-vs-rest average precision;
- macro probability summaries.

Stratified bootstrap resampling with 2,000 iterations produces 95% confidence intervals for accuracy, balanced accuracy and macro-F1.

## 15. Plots retained for every classification analysis

Each analysis produces:

- overall accuracy, balanced accuracy and macro-F1 plot;
- confusion-matrix plot;
- per-class recall and F1 plot;
- participant probability heatmap;
- model and sensor/modality selection-frequency plot;
- selected-feature frequency plot;
- one-vs-rest ROC curves;
- one-vs-rest precision–recall curves.

Every model family also receives its own metrics, confusion matrix, probability plot, ROC/precision–recall plots and selected-feature summary under `all_model_comparison`.

The root output includes:

- `analysis_comparison.csv` and `analysis_comparison.png` for all analyses;
- `matched_modality_comparison.csv` and `matched_modality_comparison.png` for the controlled 28-participant IMU, EMG and IMU+EMG comparison.

## 16. Main audit and predictor outputs

Before model results, the classifier saves:

| Output | Purpose |
|---|---|
| `frozen_labels_used.csv` | Exact frozen labels used in the run |
| `frozen_label_manifest.json` | Label and score hashes, counts and read-only status |
| `classification_protocol.json` | Preprocessing, validation and leakage-control protocol |
| `eligible_imu_predictors.csv` | Cycle-aligned raw-IMU predictors for the full cohort |
| `mapped_external_predictors_used.csv` | Validated EMG and eligible legacy IMU predictors |
| `matched_emg_frozen_labels.csv` | Frozen labels for the 28 matched participants |
| `emg_mapping_quality_control.csv` | Matched and unmatched participant audit |
| `matched_imu_predictors.csv` | IMU predictors for the matched comparison |
| `matched_emg_predictors.csv` | EMG-only predictor table |
| `matched_imu_emg_predictors.csv` | Combined primary multimodal table |
| `matched_imu_emg_legacy_predictors.csv` | Combined table for optional sensitivity analysis |
| `predictor_extraction_qc.csv` | Valid fractions and usable cycle counts |
| `feature_manifest.csv` | Feature source, modality, waveform, cycle-alignment and eligibility metadata |
| `run_summary.json` | Analyses, feature counts, mapping counts and regression summary |

## 17. Standard contents of each analysis directory

A normal classification-analysis directory contains:

```text
outer_loso_predictions.csv
overall_metrics.csv
per_class_metrics.csv
probability_metrics.csv
confusion_matrix.csv
outer_fold_selections.csv
inner_candidate_summary.csv
selected_features_by_outer_fold.csv
analysis_summary.json
overall_metrics.png
confusion_matrix.png
per_class_recall_and_f1.png
prediction_probabilities.png
model_and_sensor_selection_frequencies.png
feature_selection_frequency.png
one_vs_rest_roc.png
one_vs_rest_precision_recall.png
all_model_comparison/
```

The detailed tables allow every outer prediction, inner choice and frequently selected predictor to be audited.

## 18. Recommended interpretation sequence

The results should be interpreted in this order:

1. Report the full-cohort IMU result as the main 30-participant IMU analysis.
2. Use only the matched 28-participant analyses to compare modalities.
3. Compare matched IMU-only with EMG-only.
4. Compare matched IMU-only with IMU+EMG to assess incremental multimodal value.
5. Check balanced accuracy, macro-F1 and per-class recall rather than relying only on accuracy.
6. Inspect confusion matrices and probabilities to identify whether Moderate cases are being confused with Low or High.
7. Examine feature-selection frequency across outer folds; one selection in one fold is not stable evidence.
8. Treat hierarchical, waveform, borderline-exclusion and legacy-IMU results as sensitivity or exploratory analyses.
9. Interpret wide bootstrap intervals as evidence of uncertainty caused by the small sample.

## 19. Computational implications

The script is intentionally computationally expensive. Each enabled direct analysis performs outer LOSO, inner grouped validation, feature selection, hyperparameter tuning and held-out evaluation for eleven models.

The default run enables:

- full-cohort primary IMU classification;
- matched IMU-only classification;
- matched EMG-only classification;
- matched IMU+EMG classification;
- borderline-exclusion sensitivity;
- waveform sensitivity;
- hierarchical classification;
- continuous-score regression.

The legacy-IMU sensitivity is disabled by default. Individual analyses can be disabled during development, but the final reported run should use a documented, frozen configuration.

## 20. Limitations

- Only 28 participants have both valid SGAS-v2 labels and mapped EMG features.
- The Moderate class contains fewer matched participants than Low or High.
- EMG is available as previously extracted features rather than raw synchronized signals.
- The feature dimension is large relative to the sample size, even though only 5 or 10 predictors are selected inside each training fold.
- The severity labels are sensor-derived and sample-relative.
- Model-family comparison does not eliminate uncertainty from the small cohort.
- The optional legacy IMU features are not cycle-aligned and should not replace the primary raw-IMU representation.
- External validation on an independent cohort is required before claiming generalizable clinical performance.

## 21. Core methodological principle

The classifier follows this sequence in every outer fold:

```text
Hold out one participant
    ↓
Use only remaining participants for inner validation
    ↓
Fit imputation, scaling and feature selection inside inner training
    ↓
Select modality, model and hyperparameters
    ↓
Refit the selected pipeline on all outer-training participants
    ↓
Predict the untouched held-out participant once
    ↓
Pool all held-out predictions for final evaluation
```

This structure prevents the optimistic model-selection bias that occurs when the final test participant influences preprocessing, feature selection or model choice.
