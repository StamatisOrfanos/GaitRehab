# Bibliography

## 1. Machine Learning Based Abnormal Gait Classification with IMU Considering Joint Impairment

This study aimed to develop and optimize an abnormal gait classification algorithm considering joint impairments using inertial measurement units (IMUs) and walkway systems. Based on these simulated gaits, we developed classification models: distinguishing abnormal gait due to joint impairments, identifying specific joint disorders, and a combined model for both tasks.

**Recursive Feature Elimination with Cross-Validation** (RFECV) was used for feature extraction, and models were fine-tuned using **Support Vector Machine** (SVM), **Random Forest** (RF), and **Extreme Gradient Boosting** (XGB). The IMU-based system achieved over 91% accuracy in classifying the three types of gait. In contrast, the walkway system achieved less than 77% accuracy in classifying the three types of gait, primarily due to high misclassification rates between knee and ankle joint impairments.


Stroke patients often exhibit joint stiffness and hemiparetic gait due to lesions in the cerebral hemisphere or ataxic gait and impaired balance due to cerebellar lesions. In another study, IMUs and
a deep neural network algorithm were used to detect abnormal gait and classify four gait patterns in stroke patients

## Experimental Instrumentation and Procedure

The IMU sensor includes 3-axis accelerometers, 3-axis gyroscopes, a walkway system and a Bluetooth 4.0 module. Data were collected at a sampling frequency of 50 Hz and transmitted up to 10 m. The IMU sensor was attached to both lateral thighs and shanks parallel to the sagittal plane.

## Data Preprocessing

### Gait Event Detection

The detection of gait events, such as heel-strikes (HS) and toe-offs (TO), was performed to extract gait parameters by gait cycle. HS and TO events were detected on the sagittal plane angular velocity of the shank IMU using the ‘find_peaks’ function from the SciPy library with prominence of 20 degrees per second and distance of 20 samples.

The period between HS and the next TO was defined as the stance phase, while the period between TO and the next HS was defined as the swing the swing phase. Each period from one HS to the next constituted a complete gait cycle.

### Gait Parameter Extraction

Gait parameters encompassed angular, spatio-temporal, and symmetry parameters. The spatio-temporal parameters, such as cycle time (s),
cadence (steps/min), swing time (s), stance time (s), swing phase, and stance phase, were derived from the intervals between gait events.
The symmetry parameters, such as the temporal symmetry ratio (TSR), were calculated as the ratio of corresponding gait parameters between both legs.

TSR = (Swing_Time_Right / Stance_Time_Right) / (Swing_Time_Left / Stance_Time_Left)

## Feature Selection and Classification

### Feature Selection

Feature selection plays a crucial role in classification tasks by eliminating irrelevant information from models, thereby enhancing efficiency and accuracy. In this study, we systematically evaluated feature importance and identified the optimal subsets using Recursive Feature Elimination with Cross-Validation (RFECV).

In this study, three different methods, support vector machine (SVM), random forest (RF), and extreme gradient boosting (XGB), were used, and the optimal feature set was selected based on the average accuracy across all folds.

### Classification

The SVM configuration used a ‘linear’ kernel, with gamma set to ‘scale’ and the decision function shape set to ‘ovr’. RF combines decision trees to provide accurate predictions and utilizes parallel processing for faster computation. The RF configuration used the ‘gini’ criterion with a minimum sample split of 2. XGB constructs decision trees sequentially to correct errors, ensuring efficient processing even with large datasets. For XGB, the booster was set to ‘gbtree’, eta to 0.3, maximum depth to 6, and sampling method to ‘uniform’; the objective to minimize regression error.

## 2. Gait classification of stroke survivors - An analytical study

Gait of post-stroke patients have been analyzed and classified in this paper. Data
pertaining to gait of both post-stroke patients and healthy people of the same age group were obtained and analyzed. Xsens motion capture system was used for obtaining this data from 40 people belonging to both categories. Advanced modern machine learning techniques such as Logistic Regression, Multilayer Perceptron (MLP), Support Vector Machine (SVM), and Extreme Gradient Boosting (XGBoost) were employed for classifying the data obtained. Accurate spatiotemporal parameters values for gait were obtained. Amongst the different machine learning techniques, XGBoost gave the most accurate classification result of about 96%. The walking patterns of the patients who had undergone a stroke attack were analyzed. severity of abnormal gait patterns was a key factor that was taken into consideration and points were given accordingly. The data presented in this paper can be used to develop diagnostic tools for gait rehabilitation of stroke survivors, to evaluate and estimate their way of walking in order to understand their progress. It can be used to understand the different types of walking disorders post-stroke and thereafter select the right kind of treatment that needs to be implemented for proper recovery.