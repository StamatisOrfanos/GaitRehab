# Inner and Outer Subject-Level Cross-Validation

## Overview

This project uses nested subject-level cross-validation to separate two different tasks:

1. Selecting the best machine-learning configuration.
2. Estimating how well the complete modelling procedure generalizes to an unseen subject.

The inner and outer validation loops have different purposes. Their scores should not be interpreted in the same way.

Strictly speaking, only the outer loop in this project uses Leave-One-Subject-Out (LOSO) validation. The inner loop uses five-fold `StratifiedGroupKFold`. It is therefore more precise to call the procedure:

> Nested subject-level cross-validation with outer LOSO and inner stratified grouped cross-validation.

## Why subject-level splitting is necessary

The dataset contains multiple observations associated with individual subjects. Observations from the same person are usually more similar to one another than observations from different people.

If rows were randomly divided between training and test sets, measurements from the same subject could appear in both sets. A model could then exploit subject-specific patterns instead of learning patterns that generalize to new people. This would produce an overly optimistic estimate of performance.

Subject-level validation prevents this problem by treating each subject as an indivisible group. All observations belonging to a subject remain together in either the training partition or the validation/test partition.

## The outer LOSO loop

The outer loop estimates final generalization performance.

For each outer fold:

1. One complete subject is held out.
2. Every observation belonging to that subject becomes the outer test set.
3. All remaining subjects form the outer training set.
4. Model and feature-selection decisions are made using only the outer training set.
5. The selected configuration is fitted using the outer training subjects.
6. The fitted configuration predicts the untouched outer test subject.

This process is repeated until every subject has served as the outer test subject once.

If the dataset contains 30 subjects, the outer evaluation has 30 folds. In each fold, one subject is tested and the remaining 29 subjects are available for training and inner validation.

The held-out outer subject must not influence:

- model selection;
- sensor-combination selection;
- feature-count selection;
- recursive feature elimination;
- preprocessing parameters;
- hyperparameter selection;
- threshold selection;
- any other data-dependent modelling decision.

Predictions from all outer folds are pooled to calculate the final accuracy, balanced accuracy, precision, recall, F1 scores, Matthews correlation coefficient, classification report, and confusion matrix.

These outer-LOSO results are the primary unbiased performance estimate.

## The inner cross-validation loop

The inner loop selects the best configuration for the current outer training set.

In this project, the inner loop uses five-fold `StratifiedGroupKFold`, not LOSO. It divides the outer training subjects into five grouped folds while attempting to preserve the class distribution across folds.

For each candidate configuration, the inner loop repeatedly:

1. Uses four inner folds for fitting.
2. Uses the remaining inner fold for validation.
3. Fits all data-dependent preprocessing and feature-selection steps using only the inner training folds.
4. Evaluates the fitted candidate on the inner validation fold.
5. Repeats until every inner fold has been used for validation.
6. Averages the validation scores across the five inner folds.

The candidate configurations include combinations of:

- sensor groups;
- feature-set sizes;
- machine-learning models.

The configuration with the best mean inner score is selected for that outer fold. In this project, macro F1 is the primary selection metric.

After selection, the winning configuration is fitted again using the complete outer training set. It is then evaluated once on the held-out outer subject.

## Data flow through one outer fold

For a dataset with 30 subjects, one outer fold can be represented as follows:

```text
All 30 subjects
|
+-- 1 held-out subject
|   +-- Outer test set
|       +-- Used only for the final prediction in this fold
|
+-- 29 remaining subjects
    +-- Outer training set
        |
        +-- Inner five-fold grouped cross-validation
        |   +-- Compare sensors, feature counts, and models
        |   +-- Select the best configuration
        |
        +-- Refit the selected configuration on all 29 subjects
            +-- Predict the held-out outer subject
```

The procedure is repeated 30 times so that each subject is tested once.

## Main difference between the two loops

| Aspect | Inner cross-validation | Outer LOSO |
|---|---|---|
| Primary purpose | Select a configuration | Estimate final generalization performance |
| Data used | Only the current outer training subjects | One completely held-out subject per fold |
| Splitting method in this project | Five-fold `StratifiedGroupKFold` | Leave-One-Subject-Out |
| Subject grouping | Subjects remain intact | The entire test subject is isolated |
| Model comparison | Yes | No selection should be performed using outer test results |
| Feature selection | Fitted within inner training partitions | The selected procedure is evaluated on the outer subject |
| Meaning of score | Selection diagnostic | Unbiased test-performance estimate |
| Appropriate for headline reporting | No | Yes |

## Why the inner score is not the final performance score

The inner score is used to choose among many alternatives. Because the best-performing candidate is selected from a collection of candidates, its inner score can be optimistic. Some candidates may perform well partly because of random variation in the inner folds.

The outer subject corrects for this selection effect. It was not involved in deciding which candidate won, so performance on that subject evaluates the entire selection procedure rather than only the fitted model.

This distinction explains the wording used in the plots:

- Inner-CV plots are labelled as selection diagnostics and not final test performance.
- Outer-LOSO plots describe unbiased predictions for held-out subjects.

An inner-CV score may be useful for understanding which configurations tend to be preferred, but it should not replace the outer-LOSO result when reporting expected performance on unseen subjects.

## Feature selection and leakage prevention

Feature selection must be repeated inside the relevant training partition. It must not be performed once on the complete dataset before cross-validation.

Within an inner fold, the correct sequence is:

1. Fit preprocessing using the inner training subjects.
2. Fit feature selection using the inner training subjects.
3. Fit the classifier using the selected inner-training features.
4. Apply the fitted transformations to the inner validation subjects.
5. Evaluate the predictions.

After the inner loop selects a configuration, the same sequence is fitted on the complete outer training set and applied to the held-out outer subject.

This ensures that neither the values nor labels of validation and test subjects influence feature selection.

## Why the selected configuration can change across outer folds

Each outer fold has a slightly different training population because a different subject is removed. The inner loop may therefore select a different model, sensor combination, or feature count in different outer folds.

This is expected. It reflects the stability of the model-selection process. The selection-frequency and feature-stability plots summarize how often particular choices were selected across the outer folds.

A configuration selected frequently is relatively stable across changes in the training subjects. A configuration selected rarely may be more dependent on the particular subjects included in an outer training set.

Selection frequency is descriptive and should not be confused with predictive performance.

## How to interpret the generated files

### Final outer-LOSO outputs

These files describe performance on subjects that were excluded from configuration selection and model fitting:

- `unbiased_final_performance.csv`
- `outer_loso_predictions.csv`
- `final_classification_report.txt`
- `final_per_class_metrics.csv`
- `final_outer_loso_all_metrics.png`
- `final_outer_loso_per_class_metrics.png`
- `final_outer_loso_confusion_matrix.png`
- `outer_loso_metric_confidence_intervals.png`
- `outer_loso_performance_by_selected_configuration.png`
- `outer_loso_model_feature_accuracy_heatmap.png`
- `outer_loso_model_sensor_accuracy_heatmap.png`

These are the appropriate outputs for reporting final model performance.

The grouped configuration-performance plot is descriptive and should be interpreted
with its displayed number of contributing subjects. It does not replace the pooled
outer-LOSO metrics.

### Inner-CV diagnostic outputs

These files summarize the model-selection process inside the outer training folds:

- `inner_cv_candidate_scores.csv`
- `inner_cv_mean_candidate_scores_for_plots.csv`
- heatmaps of inner-CV scores;
- best-model comparison plots;
- model-ranking plots;
- feature-set comparison plots;
- sensor-by-feature-set matrices.

These outputs explain why configurations were selected. They do not represent independent test performance.

### Selection-stability outputs

These files describe how the selected configuration varies across outer folds:

- `configuration_selected_in_each_outer_fold.csv`
- `selected_feature_stability.csv`
- model, sensor, and feature-set selection-frequency plots;
- the model-by-sensor selection-frequency matrix;
- the feature-selection stability plot.

These outputs help assess the consistency of the selection procedure.

## What should be reported

The primary reported performance should come from the pooled outer-LOSO predictions. Relevant results include:

- accuracy;
- balanced accuracy;
- macro F1;
- per-class precision, recall, and F1;
- Matthews correlation coefficient;
- the outer-LOSO confusion matrix.

The validation method should be described explicitly. For example:

> Performance was evaluated using nested subject-level cross-validation. The outer loop used leave-one-subject-out validation to estimate generalization to unseen subjects. Within each outer training set, five-fold stratified grouped cross-validation selected the sensor combination, feature count, and classifier using macro F1. All preprocessing and recursive feature elimination steps were fitted within their respective training partitions.

Inner-CV results can be presented as supporting analyses of model selection and stability, but they should not be described as final test results.

## Summary

The inner and outer loops answer different questions:

- **Inner cross-validation:** Which configuration should be selected using the currently available training subjects?
- **Outer LOSO:** How well does the complete selection and training procedure perform on a subject it has never seen?

Keeping these roles separate prevents information leakage and produces a more credible estimate of performance on future unseen subjects.
