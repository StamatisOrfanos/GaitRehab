# 📄 Paper Summary

<br><br/>

## 1. Machine Learning Based Abnormal Gait Classification with IMU Considering Joint Impairment

### Participants

* Number: 10 healthy male participants
* Age: 39.4 ± 10.3 years
* Purpose: Simulated abnormal gait using knee and ankle braces

### Experimental Setup

* Devices Used:
  * IMU-based system: 4 IMUs (on lateral thighs and shanks), 3-axis accelerometer & gyroscope, 50 Hz
  * Walkway system: GAITRite® electronic pressure mat
  * Trials: 10 walking trials for each of the 3 conditions:
        1. Normal
        2. Knee joint impairment
        3. Ankle joint impairment

### Features

#### From IMUs

* Angular parameters:
  * Shank sagittal, Thigh sagittal and Knee joint angle

* Spatio-temporal parameters:
  * Cycle time, Swing time, Stance time, Swing/stance phase, Cadence

* Symmetry parameters:
  * Temporal Symmetry Ratio (TSR) = (Swing_Time_Right / Stance_Time_Right) / (Swing_Time_Left / Stance_Time_Left)

### Models

* SVM: Linear kernel
* Random Forest (RF): Gini criterion
* XGBoost (XGB): Booster: `gbtree`, eta = 0.3, max\_depth = 6

Feature selection method:

* Recursive Feature Elimination with Cross-Validation (RFECV)
* Cross-validation: Group 5-fold CV

### 📊 Performance

| Classification Task              | Best Model | Accuracy | F1 Score |
| -------------------------------- | ---------- | -------- | -------- |
| Normal vs Abnormal (IMU)         | RF         | 99%      | 0.99     |
| Normal vs Knee Impairment (IMU)  | SVM/RF     | 100%     | 1.00     |
| Normal vs Ankle Impairment (IMU) | RF         | 97%      | 0.97     |
| Knee vs Ankle Impairment (IMU)   | RF/XGB     | 98%      | 0.98     |
| All 3-Class (Normal/Knee/Ankle)  | RF         | 91%      | 0.91     |
|                                  | Walkway    | 77%      | 0.77     |


### Key Observations

* IMU outperforms the walkway system significantly — especially in distinguishing knee vs ankle impairment.
* Knee sagittal angle (right leg) was the most discriminative feature.
* Symmetry parameters like TSR and SSR increased in joint-impaired gait.
* IMU detected angular asymmetry more effectively than the walkway’s spatio-temporal features.

---

<br><br/>

## 2. Gait Classification of Stroke Survivors – An Analytical Study

### Participants

* Total participants: 80 (40 stroke patients, 40 healthy controls)
* Age range: 43 to 61 years
* Conditions: Stroke patients with no other mental/physical ailments; capable of walking at least 60 meters with crutches if necessary (no wheelchairs)

### Equipment & Procedure

* Device Used:
  * Xsens Motion Capture System (17 IMUs): 3D accelerometers, gyroscopes, magnetometers
  * Sampling Rate: 120 Hz
  * Software: MVN Studio
* Sensor Placement: 17 points on the body (e.g., limbs, torso, head)
* Procedure: Subjects walked counterclockwise in a bounded 850 cm × 540 cm space for 3 cycles

### Features Used (Spatiotemporal Parameters)

1. Step Count
2. Step Length (Euclidean distance between heel strikes)
3. Step Width
4. Step Frequency (steps/sec)
5. Velocity (based on distance and time)
6. Gait Cycle Duration
7. Double Support Phase Duration

### Models Used

* Logistic Regression
* Multilayer Perceptron (MLP)
  * 15 hidden layer nodes
* Support Vector Machine (SVM)
* Extreme Gradient Boosting (XGBoost)

### Performance

| Model                 | Precision (avg) | Recall (avg) | F1 Score (avg) | Accuracy |
| --------------------- | --------------- | ------------ | -------------- | -------- |
| XGBoost               | 0.96            | 0.96         | 0.96           | 96%      |
| Logistic Regression   | 0.92            | 0.92         | 0.92           | 92%      |
| Multilayer Perceptron | 0.88            | 0.88         | 0.88           | 88%      |
| SVM                   | 0.80            | 0.71         | 0.72           | 72%      |

### Key Observations

* Stroke patients had shorter step lengths, lower velocity, higher gait cycle time, and lower step frequency than healthy controls.
* Xsens system with full-body IMUs provided rich spatiotemporal data for ML.
* Study supports the use of ML tools for monitoring rehabilitation progress and detecting gait abnormalities.

---

<br><br/>

## 3. Exploring Unsupervised Feature Extraction of IMU-based Gait Data in Stroke Rehabilitation Using a Variational Autoencoder

### Participants

* Stroke group:
  * Total: 107 people after stroke
  * Age: Mean \~71.5 years (±12.8)
  * Walking aid: Yes/No/Both (32/24/21)
* Healthy controls:
  * Total: 37 (26 adults, 11 elderly)
  * Age: Adults \~42.2 years, Elderly \~84.1 years
* Inclusion: First-ever or recurrent stroke; able to walk 2 minutes at ≥0.05 m/s

### Equipment & Protocol

* Sensors:
  * Two foot-mounted IMUs (triaxial accelerometer ±8g, gyroscope ±500°/s)
  * Sampling Rate: 104 Hz

* Test:
  * 2-minute walk test (2MWT) on 14m path (straight with cones at each end)
  * Performed with and/or without walking aid

### Features Used

* Raw Input:
  * 512 × 6 epochs (5.12 seconds of 3-axis accelerometer + gyroscope data)

* Preprocessing:
  * Offset correction, step detection, sensor fusion, band-pass filtering (0.01–10 Hz), Z-score & Min-Max normalization
  * Segments started in stance phase with 50% overlap

* Latent Features:
  * 12-dimensional latent space learned by VAE (deep convolutional encoder/decoder architecture)

### Model Used

* Variational Autoencoder (VAE):
  * Encoder/decoder: 3-layer convolutional (kernel=3, filters=32–128)
  * Latent dimension: 12
  * Loss: MSE + KL divergence
  * Optimizer: Adam (lr=0.001)
  * Implemented in TensorFlow 2.11

### 📊 Performance & Evaluation

#### Model Performance

* Reconstruction MSE:
  * Stroke data: \~0.004
  * Healthy control data had higher error → model trained only on stroke data

* Latent features test-retest reliability (ICC):
  * 5 of 12 latent features: ICC > 0.75 (good to excellent)

* Statistical significance:
  * 7 latent features showed significant differences between stroke and healthy groups
  * 4 latent features had both reliability and significant group difference

| Feature    | ICC   | p-value | Effect size (Hedges’ g)       |
| ---------- | ----- | ------- | ----------------------------- |
| L1         | 0.847 | <0.01   | 3.00 (better than gait speed) |
| L5         | 0.899 | <0.01   | 1.87                          |
| L0, L2, L7 | >0.75 | varies  | up to 1.24                    |

### Key Insights

* Latent features offer complementary information to gait speed
* L1 was more sensitive to group differences than gait speed itself
* VAE can compress IMU data effectively into a small set of features with minimal loss
* Reconstruction errors for healthy controls were higher → model optimized for stroke gait patterns
* Changes in latent features may reflect diverse rehabilitation strategies (compensation vs. recovery)

---

<br><br/>

## 4. A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors

### Participants

* Subjects: 19 healthy individuals (9 males, 10 females)
* Age: 37.6 ± 13.0 years
* Purpose: Simulated pathological gait for feasibility study

### Gait Types Classified

1. Normal Gait
2. Hemiplegic
3. Equine (Foot Drop)
4. Ataxic
5. Parkinsonian

### Features and Data

* Sensors: 5 IMUs (sternum, left/right pelvis, left/right wrist)
* Sampling Rate: 128 Hz
* Signals Captured: Accelerometer, gyroscope, magnetometer (9-axis)
* Preprocessing:
  * Manual segmentation (removal of static phases)
  * Normalization to \[-1, 1]
  * 1-second windowing with 50% overlap
* No handcrafted features; raw data fed to models

### Model Architectures

Three CNN-based deep learning models:

1. mCNN-1D – Multi-branch 1D CNN (complex)
2. smCNN-1D – Simplified multi-branch 1D CNN
3. sCNN-1D – Sequential 1D CNN

### Classification Protocol and Performance

* Repeated 10x with different splits
* Evaluated IMU component impact (acc, gyro, mag)
* Tested single and combined sensor placements
* Accuracy: 100% for all 5 gait types across all 3 CNN architectures
* Recall: Also near 100%

* Inference Time:
  * smCNN-1D: \~300 ms
  * sCNN-1D: \~100–200 ms depending on configuration (suitable for real-time)
* Optimal Channels: Accelerometer and gyroscope outperformed magnetometer

### Key Takeaways

* Feasibility study using simulated gaits; real patient validation pending
* Demonstrated that deep learning (CNNs) on raw IMU data can robustly classify pathological gait
* Sensor placement and choice (especially wrist and pelvis) significantly influence results
* smCNN-1D and sCNN-1D architectures are more computationally efficient than mCNN-1D

---

<br><br/>

## 5. Detection and Classification of Stroke Gaits by Deep Neural Networks Employing Inertial Measurement Units

### Objective

To develop a deep neural network (DNN) model that:

* Detects whether a gait is healthy or stroke-affected.
* Classifies the stroke gaits into one or more of four abnormalities:
  1. Drop foot (SGwDF)
  2. Circumduction (SGwC)
  3. Hip hiking (SGwHH)
  4. Back knee (SGwBK)

### Methodology

#### Data Collection

* Subjects: 8 stroke patients, 7 healthy individuals
* Sensors: 2 IMUs (on shanks) from APDM OPAL system, 128 Hz
* Data: Angular velocity on the y-axis used to segment gait cycles by mid-swing peaks
* Samples: 2037 stroke gait cycles and 2000 normal gait cycles

#### Deep Neural Network (DNN)

* Architecture:
  * Detection branch:
    * 6 hidden layers (100 neurons each, ReLU)
    * Output: binary (normal vs stroke)
  * Classification branch:
    * 10 hidden layers (100 neurons each, ReLU)
    * Output: 5-label multi-output (multi-label sigmoid classifier)

* Training setup:
  * Optimizer: Adam
  * Loss: Binary Cross Entropy
  * Batch size: 500, Epochs: 60
  * Regularization: Dropout (0.2) in classification layers
  * Validation: 4-fold cross-validation

### Results

| Task                | Accuracy (%) | F1 Score |
| ------------------- | ------------ | -------- |
| Stroke Detection    | 99.35        | 0.9935   |
| Gait Classification | 97.31        | 0.9662   |

### Key Contributions

* First model to classify stroke gaits into clinically meaningful subtypes using IMU and DNN.
* High-performance multi-label classifier for gait abnormalities using a simple yet effective DNN architecture.

---

<br><br/>

## 6. IMU-Based Post-Stroke Gait Data Acquisition and Analysis System

### Objective

To develop an affordable and accurate gait analysis system using IMU sensors for post-stroke rehabilitation, focusing on gait asymmetry detection 
and intervention monitoring.

### Dataset

* Subjects: 20 participants (10 healthy, 10 post-stroke patients).
* Sensors: IMU sensors (accelerometer + gyroscope) placed on shanks, thighs, and waist.
* Sampling Rate: 100 Hz.
* Environment: Controlled clinical setup; subjects walked on a 10 m walkway.
* Ground Truth: Video-based gait analysis for comparison.

### Preprocessing and Features

* Filtering: Butterworth low-pass filter (cutoff: 5 Hz).
* Segmentation: Heel-strike and toe-off events detected using angular velocity signals.
* Extracted Features:
  * Step time, Swing time, Stance time, Cadence, Gait asymmetry index
  * Stride length (estimated using gyroscope integration and calibration)

### Model / Analysis Techniques

* No machine learning model was used.
* Signal processing techniques for event detection and statistical comparison.
* Gait asymmetry index calculated and visualized for assessment.

### Evaluation Metrics

* Accuracy of event detection (compared with video ground truth): \~95%.
* Gait asymmetry index differentiation between healthy and post-stroke groups.

### Results

* The IMU system successfully detected gait events with high accuracy.
* Post-stroke patients exhibited significant asymmetries, especially in stance and swing phases.
* The system differentiated healthy vs. impaired gait effectively using statistical metrics.

---

<br><br/>

## 7. Enhanced gait for individuals with stroke using leg-worn inertial sensors

### Participants

* N = 10 stroke patients
  * Age: 55.5 ± 14.5 years
  * Time since stroke: 27.9 ± 34.4 months
* Compared to 10 healthy controls

#### Features Used

* Gait events derived from inertial sensors on both legs.
* Key gait metrics:
  * Stance time, Swing time, Step time, Step length, Cadence, Symmetry indices
* Angular velocities and accelerations from shank-mounted IMUs were core signals.

#### Model or Method Used

* No ML classifier; rather, the focus was on real-time feedback system using a wearable controller.
* The system modulated vibrotactile feedback based on gait symmetry.

#### Purpose

* Aid rehabilitation via real-time feedback — not classification.
* Detect gait phases in real-time and give haptic cues to improve symmetry.

#### Performance/Results

* Improvement in temporal symmetry ratio (TSR)
* Not a classification paper — hence no model accuracy reported, but significant clinical impact was observed in pilot tests.
