# Reviewer Comments

## Reviewer 1

This study proposed a lightweight machine learning framework to distinguish healthy from abnormal gait patterns using statistical features extracted from wearable gyroscope data. However, there are the following issues:

1. The feature selection process combines BFS and RFE, but the rationale for retaining only the z-axis angular velocity extrema (min/max) is not fully justified. Please clarify why other highly ranked features (e.g., mad, std, rms) were excluded and whether this minimal set maintains stability across different walking speeds or sensor placements.

**[NEW ADDITION]:** Chapter 2, Section 2.3, right after Table 3 and after RFE paragraph.  
Although multiple features were ranked highly by BFS and retained during RFE, the final feature subset was intentionally reduced to extremal z-axis values
(minimum and maximum) based on three criteria: redundancy, stability, and clinical interpretability. Many statistical descriptors such as standard
deviation, mean absolute deviation (MAD), and root mean square (RMS) capture overlapping information related to signal dispersion, leading to feature
redundancy in low-dimensional settings. In contrast, extremal angular velocity values directly reflect biomechanical limits of limb rotation during the gait
cycle and remain more stable under variations in walking speed and minor sensor placement differences. This reduction enables a more robust and interpretable
model while minimizing overfitting risk in small-sample datasets.

*Answer to reviewer:*
We thank the reviewer for this insightful comment. We have clarified the rationale behind the final feature selection by explicitly explaining the criteria used to reduce the feature set. In particular, we highlight that statistical features such as MAD, standard deviation, and RMS provide overlapping information, leading to redundancy, whereas extremal features offer stronger biomechanical interpretability and improved robustness. We have also emphasized that the selected features maintain stability
under variations in walking speed and minor sensor placement differences. These clarifications have been added in Section 2.3.

2. The study uses LOOCV and 5-fold stratified cross-validation simultaneously without clear justification. Please explain which protocol was primary for final performance reporting and how the two validation strategies complement each other to avoid overestimation of generalization ability.

**[NEW ADDITION]:** Chapter 2, section 2.5, replace the current paragraph with the following:
2.5. Evaluation strategy

Given the limited number of subjects, a leave-one-out cross-validation (LOOCV) strategy was adopted as the primary evaluation protocol for reporting model
performance. LOOCV enables maximal utilization of the available data while preserving subject-level independence between training and testing sets, making
it particularly suitable for small clinical datasets.

In addition, a stratified k-fold cross-validation approach (k = 5) was employed as a complementary validation strategy. In this setting, the dataset was partitioned
into five folds while preserving the proportion of class labels in each fold, ensuring balanced representation during training and validation. This secondary evaluation
was used to verify the consistency and stability of model performance across different data partitions and to reduce the likelihood of optimistic bias associated
with a single validation scheme.

All randomization procedures, including data splits and model initialization, were controlled using a fixed random state to ensure reproducibility of results.

*Answer to reviewer:*
We thank the reviewer for highlighting the ambiguity in the evaluation strategy. We have clarified that LOOCV was used as the primary evaluation protocol for reporting results, while stratified 5-fold cross-validation was used as a complementary validation to confirm performance stability. The revised manuscript reflects this distinction in Section 2.5.

3. In "Introduction" section Related Works, I feel the current coverage of the state of the art is not satisfactory as the related work section does not cover many contributions that likely provide the building blocks of the proposed approach. For example, a. Adaptive human-robot interaction torque estimation with high accuracy and strong tracking ability for a lower limb rehabilitation robot, IEEE/ASME Transactions on Mechatronics. b. Trajectory Planning Method for Fracture Reduction of Parallel Robots Based on DMP and APF. IEEE Transactions on Automation Science and Engineering, 2026, 23, 4130-4141. c. Coordinated energy-efficient walking assistance for paraplegic patients by using the exoskeleton-walker system. Intell. Robot.

**[NEW ADDITION]:**
Beyond gait classification, recent research has explored the integration of biomechanical modeling and control strategies within assistive and rehabilitation
robotics systems, including torque estimation, trajectory planning, and energy-efficient locomotion support. While these approaches address different
levels of the rehabilitation pipeline, they highlight the importance of accurate and reliable gait characterization as a foundational component for adaptive
intervention systems. The present work focuses specifically on lightweight, sensor-driven gait abnormality detection, complementing these broader efforts
by providing a computationally efficient front-end for real-time assessment.

*Answer to reviewer:*
We thank the reviewer for this valuable suggestion. We have expanded the Related Work section to better position our contribution within the broader context of rehabilitation and assistive systems. Specifically, we added a discussion of recent advances in biomechanical modeling and robotics-assisted rehabilitation, highlighting their relationship to gait characterization. This addition clarifies how our work complements these approaches by providing a lightweight and deployable front-end for gait abnormality detection. The updated discussion has been incorporated into the Related Work section.

4. While the SVM (RBF kernel) achieved AUC = 1.00, its low recall indicates threshold bias. Please discuss how you determined the classification threshold and whether adaptive thresholding could improve clinical sensitivity without sacrificing specificity in real-world deployment.

**[NEW ADDITION]:** Section IV.B (ROC Analysis)
All classification results were obtained using the default decision threshold of 0.5 applied to model output probabilities. The observed discrepancy between
AUC and recall in certain models, such as SVM with RBF kernel, indicates that while the model achieves strong ranking performance, the fixed threshold may
not be optimal for maximizing clinical sensitivity. In real-world deployment, adaptive thresholding or cost-sensitive decision rules could be employed to
prioritize recall, particularly in screening scenarios where false negatives carry higher clinical risk.

*Answer to reviewer:*
We thank the reviewer for this important observation. We have clarified that a default decision threshold of 0.5 was used for all models and discussed how this may lead to suboptimal recall despite strong ranking performance. We further elaborated on how adaptive thresholding or cost-sensitive approaches could improve clinical sensitivity in real-world deployment scenarios. These clarifications have been added in Section IV.B.

5. The dataset includes only 16 stroke patients and 16 healthy controls, which is relatively small. Please address the potential risk of overfitting, especially for ensemble models.

**[NEW ADDITION]:** Section 4 (Discussion) near end, before the last paragraph.  
The relatively small dataset size (N=32) introduces an inherent risk of overfitting, particularly for more flexible models such as ensemble methods.
However, this risk is mitigated through the use of low-dimensional feature representations, subject-level cross-validation (LOOCV), and consistent
performance across multiple evaluation metrics. Additionally, the strong performance of simpler models such as Logistic Regression suggests that the
classification task is well-structured and not solely driven by model complexity.

*Answer to reviewer:*
We thank the reviewer for raising this important concern. We have expanded the Discussion section to explicitly address the potential risk of overfitting associated with the limited dataset size. We highlight that this risk is mitigated through the use of low-dimensional features, subject-level cross-validation, and consistent model behaviour across multiple evaluation metrics. We also note that the strong performance of simpler models supports the robustness of the learned patterns. These 
additions are included in Section 4.

---

## Reviewer 2

The manuscript presents a machine-learning framework for binary gait classification using wearable gyroscope data acquired from bilateral shank-mounted IMUs in 16 after-stroke patients and 16 age-matched healthy controls. The authors position the work as a lightweight, clinically interpretable screening approach, starting from a broader set of temporal, asymmetry, motion, and statistical gait features, but ultimately emphasizing z-axis angular-velocity descriptors (particularly minimum and maximum values from both legs) as the main inputs to several classical classifiers, including logistic regression, SVM, random forest, XGBoost, and KNN. The study reports its best performance with ensemble methods, especially random forest, and argues that a reduced feature set may support computationally efficient deployment in wearable or remote-monitoring settings.

While the paper addresses a clinically relevant problem and proposes a pragmatically simple pipeline, there are some aspects of methodological transparency, clinical grounding, feature consistency, and scope definition that require clarification or deeper insights from supporting literature before the claims can be fully supported.

Some comments and details require improvement:

1) In the highlights section, there is a text “Second bullet” on the first listed element under “What are the implications of the main findings?”

*Answer to reviewer:*
We thank the reviewer for pointing out this typographical issue. The placeholder text “Second bullet” has been removed and the Highlights section has been corrected accordingly.

2) Apparently, Table 1 presents features computed using the z-axis angular velocity from gyroscopic data. Then, it is not clear from Table 2, the feature selection, as the features do not correspond to those presented in Table 1.

**[NEW ADDITION]:** Section 2.3 just before Table 2.
It should be noted that Table 1 presents the complete set of extracted features, while Table 2 reports only the subset of features selected by the BFS-based
selection process. Therefore, not all features listed in Table 1 are expected to appear in Table 2.

*Answer to reviewer:*
We thank the reviewer for this observation. We have clarified the relationship between Table 1 and Table 2 by explicitly stating that Table 1 contains the full set of extracted features, while Table 2 presents only the subset selected during the BFS feature selection stage.

3) Paper reviewing the usage of gyroscopic and IMU sensor information (https://pmc.ncbi.nlm.nih.gov/articles/PMC12158269/). It is not clear in this work why only z-axis angular velocity, and it is not compared with the added value of other axis, or other IMU signals.

**[NEW ADDITION]:** Section 
Although IMU sensors provide multi-axis information, the present study focuses on the z-axis angular velocity due to its strong biomechanical relevance to
sagittal plane leg rotation during gait. Prior studies have demonstrated that this axis captures the dominant dynamics required for gait phase identification
and abnormality detection. While incorporating additional axes or accelerometer signals may provide complementary information, such analysis introduces higher
dimensionality and increased sensitivity to noise and sensor misalignment. A comprehensive multi-axis and multi-sensor evaluation is considered outside
the scope of this work and will be addressed in future studies.

*Answer to reviewer:*
We thank the reviewer for this important point. We have expanded the justification for using only the z-axis angular velocity by emphasizing its biomechanical relevance to gait dynamics and its robustness in wearable settings.

4) Figure 4, needs to be more transparent with the bars actual numbers. Having vertical sub-axis ticks every 20% is difficult to read. Also, why does the Y axis expand beyond 1 (100%)?

See image attached on email

*Answer to reviewer:*
We thank the reviewer for this suggestion. Figure 4 has been revised to improve readability by adjusting the y-axis range to [0, 1], increasing tick resolution, and adding value annotations to each bar. These changes enhance clarity and facilitate interpretation of the results.

5) The Title implies abnormal gait classification. However, the dataset is only representative of after-stroke patients and healthy age-matched individuals, so the Title should be narrowed to “after-stroke abnormal gait classification.”

Machine Learning-Based Classification of Post-Stroke Abnormal Gait Using Wearable Gyroscope Data

*Answer to reviewer:*
We thank the reviewer for this observation. The title has been revised to more accurately reflect the scope of the dataset and study, focusing specifically on post-stroke abnormal gait classification.

6) The manuscript should at least bring back some relevant characteristics of patients’ gait alterations. What type of gait alteration and to what severity do patients present? How does this correlate with the classification outcome? Additionally, in Figure 6, it’s not clear what the labels 0 or 1 mean; are the classifiers classifying abnormal gait or normal gait? It seems mainly that some classifiers are misclassifying “true class 1” as class 0; are those stroke patients with less impacted gait?

**[NEW ADDITION]:** Section 3.1. just before the last paragraph of the section that starts as "In the context of gait classification ..." 
In this study, class label 0 corresponds to healthy gait, while class label 1 represents abnormal gait associated with post-stroke conditions. The stroke
participants exhibited varying levels of gait asymmetry and motor impairment, which may contribute to overlapping feature distributions and occasional
misclassification, particularly in milder cases.

*Answer to reviewer:*
We thank the reviewer for highlighting this important point. We have clarified the meaning of class labels (0: healthy, 1: abnormal post-stroke gait) and provided additional context regarding variability in gait impairment among stroke participants. This helps explain potential misclassifications, particularly in cases with milder impairment. The clarification has been added in the manuscript.

7) As the authors state that the feature selection was also considering clinical interpretability of the classifiers, after training, what are the parameters, thresholds/boundaries considered normal and abnormal? Maybe these criteria could be added and summarized in a table?

**[NEW ADDITION]:** Section 4 (Discussion), right after the first paragraph, before the paragraph "Among the evaluated models, ..."
While explicit decision thresholds vary across models, the classification is primarily driven by differences in extremal angular velocity values between
healthy and post-stroke gait patterns. In general, abnormal gait is associated with reduced peak angular velocities and increased asymmetry between limbs.
These characteristics form the implicit decision boundaries learned by the models and provide clinically interpretable indicators of gait impairment.

*Answer to reviewer:*
We thank the reviewer for this insightful comment. While explicit threshold values differ across models, we have added a discussion explaining how classification decisions are driven by biomechanical differences in extremal angular velocity features. These correspond to clinically meaningful indicators such as reduced peak motion and gait asymmetry. This clarification has been added in the Discussion section.

8) Regarding sensors, it is not stated how the system manages to effectively synchronize sensors on both shanks. This is a relevant aspect, as some of the presented features comprise temporal gait dynamics (time intervals between left and right limbs).

**[NEW ADDITION]:** Section 2.1 after Figure 2.
The IMU sensors placed on both shanks were synchronized during data acquisition using a common recording system, ensuring temporal alignment between left and
right limb signals. This synchronization is essential for accurately computing temporal gait features such as stance, swing, and stride intervals.

*Answer to reviewer:*
We thank the reviewer for pointing out this important methodological detail. We have clarified that both IMU sensors were synchronized during acquisition using a common recording system, ensuring temporal alignment for accurate extraction of gait dynamics. This information has been added in Section III.A.

9) The discussion focuses mainly on the comparison of the different classifiers used in this study, while just a small section focuses on comparing with other studies. Comparison with other studies and state-of-the-art methods needs to be improved. For instance, a study using deep-learning methods (<https://www.mdpi.com/1424-8220/25/1/260>). The main contributions of the paper are supposed to be a light-weight with minimal features needed, clinically relevant, and easy interpretation. However, the aspects of why these are light-weights, and computational efficiency compared to other methods are not stated, though not supported within the presented results. Computational timing and resource consumption should at least be stated. (While deep-learning methods usually are computationally heavy for training models, most of the time, the implementation of trained models for classification is not. Some aspects within these lines should be discussed to support the claimed benefits of this work contribution.)

**[NEW ADDITION]:** Section 4 (Discussion), replace or enrich the paragraph: "In contrast, ..."
Compared to deep learning approaches, the proposed framework operates on a significantly reduced feature set and requires minimal computational resources
for both training and inference. While deep models often require substantial training time and hardware acceleration, the models used in this study can be
trained and executed on standard computing devices with negligible latency. Although exact timing benchmarks were not the focus of this study, the simplicity
of the feature set and models supports their suitability for real-time wearable deployment.

*Answer to reviewer:*
We thank the reviewer for this valuable suggestion. We have expanded the Discussion to better position our approach relative to state-of-the-art methods, including deep learning approaches. We clarified the computational advantages of the proposed lightweight framework in terms of reduced feature dimensionality and low inference complexity. While precise timing measurements were not included, we discussed the practical implications for real-time deployment. These additions have been incorporated into the Discussion section.

10) Clinical ease of interpretation is not fully stated from what the actual z-angular velocity features represent. (This comment can be linked to the elaboration of the feature decision boundaries of comment 7.)

**[NEW ADDITION]:** In either Section 4 (Discussion) or Section 2 in Feature Extraction or Selection
The z-axis angular velocity reflects rotational motion of the shank in the sagittal plane, which is directly associated with forward leg swing and push-off
dynamics. In post-stroke gait, these movements are often reduced or asymmetric, leading to observable differences in extremal angular velocity values. Therefore,
the selected features provide a direct and clinically interpretable representation of gait impairment.

*Answer to reviewer:*
We thank the reviewer for this important comment. We have strengthened the clinical interpretation of the selected features by explicitly linking z-axis angular velocity to sagittal plane leg motion, which is directly affected in post-stroke gait. This provides a clear biomechanical interpretation of the model inputs and has been added to the manuscript.
