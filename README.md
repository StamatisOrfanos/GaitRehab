# GaitRehab_Models

This projet focuses on gait abnormalities following a stroke, particularly hemiplegic gait, and how rehabilitation can help recover motor function.
The goal is to analyze and understand muscle performance and gait cycle using sensor data such as gyroscope and EMG (electromyography).
This includes detecting abnormalities through parameters like stride time, stance time, and swing time.

It highlights:

1. Causes of abnormal gait (e.g., muscle weakness, abnormal tone).
2. Phases of gait (stance, swing, stride, step).
3. Gait analysis using sensor signals like gyroscope angular velocity.
4. Signal processing using a low-pass Butterworth filter (35 Hz).
5. Calculation of gait metrics.
6. Threshold-based alerts for asymmetry (e.g., if stance % difference > 10%).

## Key Definitions and Concepts

### Stroke & Hemiplegic Gait

**Hemiplegia**: Paralysis on one side of the body (arm, leg, trunk).

**Hemiplegic gait**: Asymmetric walking pattern due to stroke-induced paralysis, often slower with impaired balance.

### Gait Cycle Terminology

1. Gait cycle: One complete sequence of walking, starting and ending with the same foot touching the ground.
2. Stance phase: ~60% of the gait cycle; foot is on the ground, body weight is supported.
3. Swing phase: ~40% of the gait cycle; foot moves forward, not in contact with the ground.
4. Stride: Distance or time between two consecutive initial contacts of the same foot.

![Gait Cycle Phases](./data/Information/phase.jpg)

**Step**: Distance or time between initial contacts of alternating feet.

### Signal Analysis

**Gyroscope signal**: Measures angular velocity (z-axis used); processed to extract gait events.

**Butterworth filter**: A type of signal filter that smooths data; 2nd-order, 35 Hz cutoff used.

### Key events from signal

1. Stride time: Peak to peak in gyroscope signal.
2. Stance time: From zero crossing to minimum point (before peak).
3. Swing time: From minimum point (before peak) to next zero crossing.

### Gait Metrics

1. **% Stance** and **% Swing**: Percentage of time in stance or swing phase relative to full gait cycle.

1. **% Difference of stance/swing**: Asymmetry measure between limbs. If difference > 10%, trigger alert.

### EMG (Electromyography)

Used to monitor muscle activity during gait to assess coordination and timing.

### Neuroplasticity in Rehab

Motor recovery relies on:

1. Task-specific, progressive training.
2. Brain reorganization (neuroplasticity).
3. Formation of stronger neural connections.

---

## Data Sources

1. **IMU Accelerometer Data**: provides us a body motion information (e.g., trunk or limb acceleration) over time, which is
useful for detecting gait events like heel strikes, toe-offs, stride time, and cadence.

2. **Vicon Analog Data**: provides High-frequency (1000Hz) force and moment data from force plates. EMG data (16 muscles) to study muscle
activation patterns during walking. Foot switches for detecting precise gait events (stance/swing phases, foot contact).

3. **Video**: provides a visual reference for gait phase annotation and validation. Potential use for labeling, annotating abnormal gaits, or validating model outputs.

---

### Possible Applications

#### Gait Event Detection

| **Goal** | **Tools** |
| ---------------------------------- | ------------------------------------------------------- |
| Detect heel strikes / toe-offs     | FootSwitch1, FootSwitch2, vertical GRF (Fz), IMU z-axis |
| Compute gait phases (stance/swing) | From timing of events |
| Calculate stride/step/cadence      | Time between events or peaks in IMU/Vicon |

#### Spatiotemporal Gait Metrics

- **Stride time** = time between heel strikes of the same foot.
- **Step time** = time between alternating foot contacts.
- **Stance/swing time** = from force plates or foot switches.
- **Cadence** = steps per minute.
- **Symmetry/asymmetry index** = compare left/right step timing

#### Muscle Activation Analysis (EMG)

| **Goal** | **Tools** |
| ---------------------------------- | ------------------------------------------------------- |
| Muscle recruitment timing            | When each muscle starts/stops firing relative to gait cycle|
| EMG envelopes                        | Smoothed EMG signals using RMS or filters |
| Compare affected vs. unaffected side | Hemiplegic patterns show asymmetric activations |
| Activation correlation with events   | E.g., tibialis anterior during swing phase |

#### Combined Sensor Fusion

- Synchronize IMU and Vicon using timestamps or signal events.
- Use IMU + EMG + Vicon for multi-modal gait model.
- Train models to detect gait phases using only IMU/EMG and validate against Vicon ground truth.

#### Alert System (SOS)

- Implement detection of gait asymmetries or abnormal %stance differences.
- Trigger alerts if left/right stance % differ >10%.
- Use IMU + EMG to detect changes in walking patterns (e.g., fatigue, recovery)

#### AI/ML Opportunities

1. Train a model to classify hemiplegic vs. normal gait from IMU or EMG, using time-series models (e.g., LSTM, 1D CNN, Transformer).
    - Create features from: IMU signals (stride stats, frequency content), EMG bursts and timing, Vicon ground truth labels
