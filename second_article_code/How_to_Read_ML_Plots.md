# How to Read the Nested Cross-Validation Plots

## Purpose of this guide

The figures in `outputs/ml_results_nested_subject_cv` describe two different parts of the analysis:

1. Final performance on held-out subjects.
2. Diagnostic information about the model-selection process.

These two groups must be interpreted differently. Final outer-LOSO plots estimate performance on unseen subjects. Inner-CV plots explain how configurations behaved during selection, but they are not independent test results.

This guide covers all 35 plots in the `plots` directory and the final confusion matrix in the `confusion_matrices` directory.

## Recommended reading order

Read the plots in this order:

1. `final_outer_loso_all_metrics.png`
2. `outer_loso_metric_confidence_intervals.png`
3. `final_outer_loso_per_class_metrics.png`
4. `final_outer_loso_confusion_matrix.png`
5. The three outer-LOSO configuration summary plots
6. The three `best_model_inner_cv_*.png` plots
7. The three `sensor_feature_matrix_inner_cv_*.png` plots
8. The twelve `heatmap_inner_cv_*.png` plots
9. The three `model_ranking_inner_cv_*.png` plots
10. The three `feature_set_comparison_inner_cv_*.png` plots
11. The four selection-frequency plots
12. `selected_feature_stability_top30.png`

This order starts with the unbiased final result and then moves into increasingly detailed diagnostic information.

## Metrics used in the plots

### Accuracy

Accuracy is the proportion of all predictions that are correct.

```text
Accuracy = correct predictions / all predictions
```

Accuracy is easy to understand, but it can be dominated by the largest class. A high accuracy does not guarantee that every class is recognized well.

### Balanced accuracy

Balanced accuracy is the average recall across the classes. Each class contributes equally, regardless of how many samples it contains.

Balanced accuracy is particularly useful when the classes are unequal in size. A difference between accuracy and balanced accuracy often indicates that performance is stronger for a larger or easier class.

### Macro F1

Macro F1 calculates the F1 score separately for each class and then gives every class equal weight.

It balances precision and recall while preventing the largest class from dominating the result. Macro F1 is the primary configuration-selection metric in this analysis.

### Macro precision

Macro precision is the average of the class-specific precision values. It describes how reliable the model's predictions are across all classes, giving each class equal weight.

### Macro recall

Macro recall is the average of the class-specific recall values. For multiclass classification, it is equivalent to balanced accuracy when calculated over the same classes.

### Weighted F1

Weighted F1 averages the class-specific F1 scores while weighting each class by its number of observations. Larger classes therefore have more influence than smaller classes.

### Matthews correlation coefficient

The Matthews correlation coefficient, or MCC, summarizes the relationship between true and predicted labels. Its theoretical range is from -1 to 1:

- 1 indicates perfect prediction.
- 0 indicates performance with no useful correlation.
- -1 indicates complete disagreement.

MCC is useful because it considers the full confusion matrix and remains informative when class sizes differ.

## Final outer-LOSO performance plots

These are the most important figures for evaluating model performance. They use predictions made for subjects who were excluded from model selection and training in their corresponding outer fold.

### 1. Overall performance summary

File:

`plots/final_outer_loso_all_metrics.png`

#### How to read it

- The vertical axis lists the evaluation metrics.
- The horizontal axis shows the score.
- Longer bars represent higher values.
- The number beside each bar gives the exact score.
- Metrics are ordered by value to make their relative levels easier to see.

#### What to examine

Compare accuracy with balanced accuracy and macro F1. If accuracy is noticeably higher, the model may be performing better on the larger or easier class than on the other classes.

Compare macro F1 with weighted F1. A higher weighted F1 suggests that the larger classes are performing better than the smaller classes.

Use MCC as an additional whole-model summary rather than comparing it numerically as if it were identical to accuracy or F1. It has a different mathematical definition and theoretical range.

#### Valid conclusion

This plot summarizes the model-selection procedure's performance on unseen outer subjects.

#### Invalid conclusion

Do not identify the best individual model from this plot. Different outer folds can select different configurations, so this figure evaluates the complete selection procedure rather than one fixed classifier.

### 2. Per-class performance

File:

`plots/final_outer_loso_per_class_metrics.png`

#### How to read it

- The horizontal axis lists the three classes.
- Each class has bars for precision, recall, and F1.
- The vertical axis runs from 0 to 1.
- Exact values appear above the bars.

#### Interpretation of each measure

- Precision asks: when the model predicts this class, how often is it correct?
- Recall asks: of all true observations from this class, how many did the model identify?
- F1 summarizes the balance between precision and recall.

#### What to examine

Look for differences between classes. Similar values indicate consistent performance across classes. A low value for one class identifies a class-specific weakness that an overall score could hide.

Compare precision and recall within a class:

- Lower precision indicates more false-positive predictions for that class.
- Lower recall indicates more missed observations from that class.
- Similar precision and recall indicate a relatively balanced error pattern.

#### Valid conclusion

This plot shows which classes are recognized reliably when subjects are held out.

#### Caution

Always consider class support. A score based on fewer observations is more uncertain and may change more with a small number of errors.

### Outer metric confidence intervals

File:

`plots/outer_loso_metric_confidence_intervals.png`

#### How to read it

- Each row represents one final outer-LOSO metric.
- The point is the metric calculated from all pooled outer predictions.
- The horizontal line is a 95% percentile bootstrap interval.
- The printed text gives the estimate followed by the lower and upper limits.
- Subjects, rather than individual rows, are resampled during bootstrapping.

The interval shows how much the final estimate could vary when the sampled subjects change. Wider intervals indicate greater uncertainty.

The interval is not the range of outer-fold scores. It is a subject-level bootstrap estimate of uncertainty around the pooled metric.

### 3. Final confusion matrix

File:

`confusion_matrices/final_outer_loso_confusion_matrix.png`

#### How to read it

- Rows represent true classes.
- Columns represent predicted classes.
- Diagonal cells are correct predictions.
- Off-diagonal cells are errors.
- Each cell shows the number of predictions and the percentage of its true-class row.
- Darker cells contain more observations.

Read one row at a time. For example, the affected-side row shows how all true affected-side observations were classified. The diagonal entry is the recall for that class. Off-diagonal entries show which other classes received the errors.

#### What to examine

- A strong diagonal indicates good classification.
- Large off-diagonal cells reveal specific class confusions.
- Symmetric errors between two classes indicate that the model has difficulty separating that pair.
- One-directional errors can indicate that one class is systematically being absorbed into another.

#### Important caution about color

The color represents counts, not percentages. A class with more observations can therefore have darker cells. Use the printed row percentages when comparing error rates between classes.

## Outer-LOSO configuration summaries

These figures provide a simpler explanation of which configuration components were selected and how the corresponding held-out predictions performed.

### 1. Performance by selected component

File:

`plots/outer_loso_performance_by_selected_configuration.png`

The three panels group outer predictions according to the model, sensors, and feature set selected in their folds.

- Bar length is pooled outer accuracy for that selected option.
- The printed score gives the exact accuracy.
- `n` is the number of outer subjects whose inner loop selected that option.

This plot is descriptive only. Options were not assigned randomly, and some have very small sample sizes. A bar with `n=1` must not be interpreted as reliable evidence that the option is superior.

### 2. Model and feature-set accuracy heatmap

File:

`plots/outer_loso_model_feature_accuracy_heatmap.png`

- Rows are models selected by inner CV.
- Columns are feature sets selected by inner CV.
- Each populated cell shows held-out accuracy and the number of contributing subjects.
- Darker cells indicate higher held-out accuracy.
- A gray `Not selected` cell means that pairing never won the inner selection process.

This is the simplest plot for answering which selected model and feature-set pairings produced the strongest observed outer accuracy.

The plot is not a complete head-to-head comparison. It contains only pairings that were selected in at least one outer fold. A high value based on one subject is much less reliable than a similar value based on many subjects.

### 3. Model and sensor accuracy heatmap

File:

`plots/outer_loso_model_sensor_accuracy_heatmap.png`

- Rows are models selected by inner CV.
- Columns are sensor combinations selected by inner CV.
- Each populated cell shows held-out accuracy and the number of contributing subjects.
- Gray cells identify combinations that were never selected.

Use this plot to understand whether the observed outer performance is associated with particular model and sensor pairings. Read the displayed sample size before comparing cells.

## Best-model plots

Files:

- `plots/best_model_inner_cv_accuracy.png`
- `plots/best_model_inner_cv_balanced_accuracy.png`
- `plots/best_model_inner_cv_macro_f1.png`

Each file uses the metric named in its filename.

### How to read them

Each figure contains four panels:

- All features
- Top 5 RFE features
- Top 10 RFE features
- Top 15 RFE features

Within each panel:

- The vertical axis lists sensor combinations.
- The horizontal position shows the best mean inner-CV score for that sensor and feature-set pair.
- The printed number is the exact score.
- Point color identifies the model that achieved that score.
- The model-color mapping is shown in the legend.

The horizontal scale is shared across all four panels within the same figure. Points can therefore be compared directly between panels in that figure.

### What these plots answer

They answer:

> For each sensor and feature-set combination, which model obtained the highest mean inner-CV score, and what was that score?

### Useful comparisons

- Compare points within a feature-set panel to assess sensor combinations.
- Follow a sensor row across panels to assess whether feature reduction helped.
- Compare point colors to see whether the best model changes across conditions.
- Compare the accuracy, balanced-accuracy, and macro-F1 versions to see whether the preferred configuration depends on the metric.

### Caution

These are inner-CV selection scores averaged across outer training folds. They are not final test scores. A configuration with the highest point should not be reported as having achieved that performance on unseen subjects.

## Sensor-by-feature-set matrices

Files:

- `plots/sensor_feature_matrix_inner_cv_accuracy.png`
- `plots/sensor_feature_matrix_inner_cv_balanced_accuracy.png`
- `plots/sensor_feature_matrix_inner_cv_macro_f1.png`

### How to read them

- Rows represent sensor combinations.
- Columns represent feature sets.
- Each cell contains the score of the best model for that sensor and feature-set combination.
- The color bar maps cell color to score.
- Darker blue generally represents a higher score.

These matrices contain the same type of best-candidate score shown in the best-model plots, but present the values in a compact table for pattern recognition.

### What to examine

- Read across a row to see how feature reduction affects one sensor combination.
- Read down a column to compare sensor combinations under the same feature set.
- Look for consistently strong rows rather than focusing only on a single dark cell.
- Look for feature-set columns that perform consistently across sensors.

### Important color-scale caution

The color scale is adapted to the values in each figure. Always read the color bar. Do not compare color darkness across separate metric figures without comparing their numerical values and color-bar ranges.

### Model identity

These matrices show the score of the best model but not its identity. Use the corresponding `best_model_inner_cv_*.png` plot to identify which model produced each cell.

## Detailed model-by-sensor heatmaps

There are twelve heatmaps. Four feature sets are produced for each of three metrics.

### Accuracy heatmaps

- `plots/heatmap_inner_cv_accuracy_all_features.png`
- `plots/heatmap_inner_cv_accuracy_top_5_rfe_features.png`
- `plots/heatmap_inner_cv_accuracy_top_10_rfe_features.png`
- `plots/heatmap_inner_cv_accuracy_top_15_rfe_features.png`

### Balanced-accuracy heatmaps

- `plots/heatmap_inner_cv_balanced_accuracy_all_features.png`
- `plots/heatmap_inner_cv_balanced_accuracy_top_5_rfe_features.png`
- `plots/heatmap_inner_cv_balanced_accuracy_top_10_rfe_features.png`
- `plots/heatmap_inner_cv_balanced_accuracy_top_15_rfe_features.png`

### Macro-F1 heatmaps

- `plots/heatmap_inner_cv_macro_f1_all_features.png`
- `plots/heatmap_inner_cv_macro_f1_top_5_rfe_features.png`
- `plots/heatmap_inner_cv_macro_f1_top_10_rfe_features.png`
- `plots/heatmap_inner_cv_macro_f1_top_15_rfe_features.png`

### How to read them

- Rows represent models.
- Columns represent sensor combinations.
- Each heatmap is restricted to one feature set and one metric.
- Each cell shows the mean inner-CV score.
- Cell color represents the same value according to the color bar.

### What to examine

- Read across a row to see how one model responds to different sensor inputs.
- Read down a column to compare models using the same sensor combination.
- Look for strong rows to identify models that are robust across sensors.
- Look for strong columns to identify sensor combinations that work across several models.
- Compare the four feature-set heatmaps numerically to assess the effect of feature reduction.

### Important color-scale caution

Each heatmap uses an adaptive color range to reveal differences within that figure. A dark cell in one heatmap is not automatically better than a lighter cell in another heatmap. Use the printed values or color bars for cross-figure comparisons.

### Why these plots differ from the best-model plots

The detailed heatmaps show every evaluated model for a given sensor and feature set. The best-model plots retain only the highest-scoring model for each sensor and feature-set pair.

## Average model-ranking plots

Files:

- `plots/model_ranking_inner_cv_accuracy.png`
- `plots/model_ranking_inner_cv_balanced_accuracy.png`
- `plots/model_ranking_inner_cv_macro_f1.png`

### How to read them

- Each row represents one model.
- The horizontal position gives its mean score across all evaluated sensor and feature-set combinations.
- The printed number is the exact mean.
- Point colors use the same model-color mapping as the best-model plots.
- Models are ordered by their mean score.

### What these plots answer

They answer:

> Which models perform strongly on average across the full range of sensor and feature-set conditions?

This is different from asking which model produces the single best result.

### What to examine

- A high mean suggests robust performance across many configurations.
- A model with a lower mean could still be best for a specific sensor and feature set.
- Small numerical differences should not automatically be treated as meaningful without uncertainty estimates or statistical comparison.

### Caution

These are aggregated inner-CV diagnostics. They do not establish that the top-ranked model has superior outer-subject performance.

## Feature-set comparison plots

Files:

- `plots/feature_set_comparison_inner_cv_accuracy.png`
- `plots/feature_set_comparison_inner_cv_balanced_accuracy.png`
- `plots/feature_set_comparison_inner_cv_macro_f1.png`

### How to read them

- Each row represents a feature set.
- The horizontal position is the mean best-model score across sensor combinations.
- The point color identifies the feature set.
- The exact score is printed beside the point.

For each sensor and feature-set pair, the best model is selected first. Those best-model scores are then averaged across sensors.

### What these plots answer

They answer:

> After allowing each sensor and feature-set pair to use its best model, which feature-set size performs best on average across sensors?

### Caution

This comparison combines model selection and averaging. It does not show whether the same features were selected in every outer fold, and it does not prove that one feature-set size will be best for a new independent dataset.

## Selection-frequency plots

These plots describe which configurations were selected as the winner in the outer folds. Selection was based on the inner macro-F1 criterion.

### 1. Model selection frequency

File:

`plots/selection_frequency_models.png`

- Each bar represents a model.
- Bar length and the printed value show the number of outer folds in which that model was selected.
- Model colors are consistent with the other figures.

A frequently selected model is stable under changes to the outer training population. Frequency does not directly measure predictive performance.

### 2. Sensor selection frequency

File:

`plots/selection_frequency_sensors.png`

- Each bar represents a sensor combination.
- The value shows how many outer folds selected that sensor combination.

This plot indicates whether the selection process repeatedly favors a particular sensor set.

### 3. Feature-set selection frequency

File:

`plots/selection_frequency_feature_sets.png`

- Each bar represents a feature-set size.
- The value shows how many outer folds selected that feature set.

This helps determine whether the selection procedure consistently favors all features or a reduced feature set.

### 4. Model-by-sensor selection matrix

File:

`plots/selection_frequency_model_sensor_matrix.png`

- Rows represent models.
- Columns represent sensor combinations.
- Each cell counts how many outer folds selected that exact model and sensor pairing.
- Darker cells represent more selections.

This matrix reveals interactions hidden by the separate frequency charts. For example, a model may be selected frequently only when paired with one particular sensor combination.

### Frequency-plot cautions

- Counts should sum to the number of outer folds in each single-category frequency plot.
- Frequent selection indicates stability, not necessarily a large performance advantage.
- A rare selection is not necessarily poor. It may be useful for a particular training population.
- These plots summarize configuration choices, not final prediction correctness.

## Feature-selection stability plot

File:

`plots/selected_feature_stability_top30.png`

### How to read it

- The vertical axis lists the 30 most frequently retained features.
- The horizontal axis shows the number of outer folds whose winning configuration contained each feature.
- The printed number gives the exact count.
- Color reinforces frequency, with brighter colors representing larger counts.

### What this plot answers

It answers:

> Which individual features appear most consistently in the configurations selected across outer folds?

A frequently retained feature is stable across changes in the outer training subjects.

### Important cautions

- Selection frequency is not the same as feature importance.
- It does not show whether increasing a feature's value raises or lowers a class probability.
- Correlated features can substitute for one another, causing individually useful features to have lower frequencies.
- A feature can only be selected when its sensor is included in the winning sensor combination.
- Feature-set size affects selection opportunity. A configuration using more features naturally includes more variables.
- The plot does not support causal conclusions.

## Comparing the three main metrics

When reading accuracy, balanced accuracy, and macro F1 together:

1. Start with accuracy to understand the overall correct-prediction rate.
2. Check balanced accuracy to determine whether recall is consistent across classes.
3. Check macro F1 to assess the balance of precision and recall across classes.

If all three are similar, performance is likely reasonably balanced. If accuracy is higher than balanced accuracy and macro F1, the result may be driven partly by stronger performance on the largest or easiest class.

Configuration rankings can also change between metrics. This is expected because each metric rewards different behavior. The primary configuration-selection metric for this project is macro F1, so the macro-F1 diagnostic plots are the most directly connected to the selection decisions.

## Common interpretation mistakes

### Treating inner-CV scores as final test scores

Inner-CV scores were used during configuration selection. Report outer-LOSO scores as final performance.

### Selecting a model after viewing outer results

The outer results estimate the complete predefined procedure. Choosing a new model because it looks best on outer results would reuse the test information and weaken the unbiased interpretation.

### Comparing heatmap colors without reading color bars

Heatmap scales adapt to each figure. Compare printed numbers and color-bar ranges, not color alone.

### Assuming small differences are meaningful

A difference of a few thousandths may reflect sampling variation. The plots do not by themselves provide confidence intervals or formal significance tests.

### Equating selection frequency with performance

Frequency measures how often a configuration wins, not how large its advantage is or how accurate its predictions are.

### Ignoring class-specific results

A good overall score can coexist with poor performance for one class. Always examine the per-class plot and confusion matrix.

### Interpreting feature frequency causally

Frequently selected features are stable inputs to the prediction procedure. They are not necessarily causal clinical factors.

## A concise reporting workflow

For a report, thesis, or presentation:

1. Report the values from `final_outer_loso_all_metrics.png`.
2. Discuss class-specific behavior using `final_outer_loso_per_class_metrics.png`.
3. Describe the main error pattern using the final confusion matrix.
4. Use the macro-F1 inner-CV plots to explain configuration selection.
5. Use selection-frequency plots to discuss stability across outer folds.
6. Use the feature-stability plot as a descriptive analysis, with the stated limitations.

Always state that outer LOSO provides the final performance estimate and that inner-CV figures are model-selection diagnostics.
