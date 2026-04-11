# Reviewer Comments

## Reviewer 1

This study proposed a lightweight machine learning framework to distinguish healthy from abnormal gait patterns using statistical features extracted from wearable gyroscope data. However, there are the following issues:

1. The feature selection process combines BFS and RFE, but the rationale for retaining only the z-axis angular velocity extrema (min/max) is not fully justified. Please clarify why other highly ranked features (e.g., mad, std, rms) were excluded and whether this minimal set maintains stability across different walking speeds or sensor placements.

2. The study uses LOOCV and 5-fold stratified cross-validation simultaneously without clear justification. Please explain which protocol was primary for final performance reporting and how the two validation strategies complement each other to avoid overestimation of generalization ability.

3. In "Introduction" section Related Works, I feel the current coverage of the state of the art is not satisfactory as the related work section does not cover many contributions that likely provide the building blocks of the proposed approach. For example, a. Adaptive human-robot interaction torque estimation with high accuracy and strong tracking ability for a lower limb rehabilitation robot, IEEE/ASME Transactions on Mechatronics. b. Trajectory Planning Method for Fracture Reduction of Parallel Robots Based on DMP and APF. IEEE Transactions on Automation Science and Engineering, 2026, 23, 4130-4141. c. Coordinated energy-efficient walking assistance for paraplegic patients by using the exoskeleton-walker system. Intell. Robot.

4. While the SVM (RBF kernel) achieved AUC = 1.00, its low recall indicates threshold bias. Please discuss how you determined the classification threshold and whether adaptive thresholding could improve clinical sensitivity without sacrificing specificity in real-world deployment.

5、The dataset includes only 16 stroke patients and 16 healthy controls, which is relatively small. Please address the potential risk of overfitting, especially for ensemble models.

## Reviewer 2

The manuscript presents a machine-learning framework for binary gait classification using wearable gyroscope data acquired from bilateral shank-mounted IMUs in 16 after-stroke patients and 16 age-matched healthy controls. The authors position the work as a lightweight, clinically interpretable screening approach, starting from a broader set of temporal, asymmetry, motion, and statistical gait features, but ultimately emphasizing z-axis angular-velocity descriptors (particularly minimum and maximum values from both legs) as the main inputs to several classical classifiers, including logistic regression, SVM, random forest, XGBoost, and KNN. The study reports its best performance with ensemble methods, especially random forest, and argues that a reduced feature set may support computationally efficient deployment in wearable or remote-monitoring settings.

While the paper addresses a clinically relevant problem and proposes a pragmatically simple pipeline, there are some aspects of methodological transparency, clinical grounding, feature consistency, and scope definition that require clarification or deeper insights from supporting literature before the claims can be fully supported.

Some comments and details require improvement:

1) In the highlights section, there is a text “Second bullet” on the first listed element under “What are the implications of the main findings?”

2) Apparently, Table 1 presents features computed using the z-axis angular velocity from gyroscopic data. Then, it is not clear from Table 2, the feature selection, as the features do not correspond to those presented in Table 1.

3) Paper reviewing the usage of gyroscopic and IMU sensor information (https://pmc.ncbi.nlm.nih.gov/articles/PMC12158269/). It is not clear in this work why only z-axis angular velocity, and it is not compared with the added value of other axis, or other IMU signals.

4) Figure 4, needs to be more transparent with the bars actual numbers. Having vertical sub-axis ticks every 20% is difficult to read. Also, why does the Y axis expand beyond 1 (100%)?

5) The Title implies abnormal gait classification. However, the dataset is only representative of after-stroke patients and healthy age-matched individuals, so the Title should be narrowed to “after-stroke abnormal gait classification.”

6) The manuscript should at least bring back some relevant characteristics of patients’ gait alterations. What type of gait alteration and to what severity do patients present? How does this correlate with the classification outcome? Additionally, in Figure 6, it’s not clear what the labels 0 or 1 mean; are the classifiers classifying abnormal gait or normal gait? It seems mainly that some classifiers are misclassifying “true class 1” as class 0; are those stroke patients with less impacted gait?

7) As the authors state that the feature selection was also considering clinical interpretability of the classifiers, after training, what are the parameters, thresholds/boundaries considered normal and abnormal? Maybe these criteria could be added and summarized in a table?

8) Regarding sensors, it is not stated how the system manages to effectively synchronize sensors on both shanks. This is a relevant aspect, as some of the presented features comprise temporal gait dynamics (time intervals between left and right limbs).

9) The discussion focuses mainly on the comparison of the different classifiers used in this study, while just a small section focuses on comparing with other studies. Comparison with other studies and state-of-the-art methods needs to be improved. For instance, a study using deep-learning methods (<https://www.mdpi.com/1424-8220/25/1/260>). The main contributions of the paper are supposed to be a light-weight with minimal features needed, clinically relevant, and easy interpretation. However, the aspects of why these are light-weights, and computational efficiency compared to other methods are not stated, though not supported within the presented results. Computational timing and resource consumption should at least be stated. (While deep-learning methods usually are computationally heavy for training models, most of the time, the implementation of trained models for classification is not. Some aspects within these lines should be discussed to support the claimed benefits of this work contribution.)

10) Clinical ease of interpretation is not fully stated from what the actual z-angular velocity features represent. (This comment can be linked to the elaboration of the feature decision boundaries of comment 7.)