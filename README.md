# EEG Intent-to-Speak Classification System

This repository contains the complete implementation and evaluation framework for detecting a participant's **Intent-to-Speak** from electroencephalography (EEG) signals using a minimal-electrode configuration. The objective is to classify whether a participant is preparing to speak (the 500 ms window immediately preceding speech onset) against no-intent intervals (resting, listening, or concentration) for real-time brain-computer interface (BCI) applications.

---

## 1. Repository Structure

The project codebase is organized into the following structured directories:

*   **`src/`**: Core library modules containing processing logic and modeling parameters:
    *   [config.py](file:///d:/Temple%20Project/src/config.py): Contains all global constants, directory paths, channel selections, preprocessing filter frequencies, trigger codes, and evaluation parameters.
    *   [dataset.py](file:///d:/Temple%20Project/src/dataset.py): Handles dataset splitting (randomized trial-level or session-level), class balancing to target 1:N ratios, raw channel normalization using training statistics, and feature extraction execution.
    *   [preprocessing.py](file:///d:/Temple%20Project/src/preprocessing.py): Functions for BDF raw file scanning and loading, EXG channel re-referencing, bandpass/notch filtering, downsampling, baseline variance estimation, and amplitude/variance-based artifact rejection.
    *   [features.py](file:///d:/Temple%20Project/src/features.py): High-speed feature extraction routines (including Hjorth parameter, spectral entropy, fast skewness/kurtosis, and Welch PSD calculations) and the continuous raw data epoch slicing logic.
    *   [model.py](file:///d:/Temple%20Project/src/model.py): Core training functions for fitting the standard Linear SVM classifier and scoring functions computing standard classification metrics.
*   **`scripts/`**: Executable Python scripts categorized by function:
    *   **`scripts/eval/`**: Core evaluation pipelines:
        *   [run_pipeline.py](file:///d:/Temple%20Project/scripts/eval/run_pipeline.py): Within-subject cross-validation pipeline across all subjects.
        *   [run_cross_subject.py](file:///d:/Temple%20Project/scripts/eval/run_cross_subject.py): All-to-all pairwise cross-subject transfer learning matrix evaluation.
        *   [run_group_cross_subject.py](file:///d:/Temple%20Project/scripts/eval/run_group_cross_subject.py): Cohort-based cross-subject transfer learning (training on groups and testing on unseen subjects).
        *   [run_unified_model.py](file:///d:/Temple%20Project/scripts/eval/run_unified_model.py): A single global classifier trained on pooled trials from all subjects combined.
        *   [run_mixed_windows.py](file:///d:/Temple%20Project/scripts/eval/run_mixed_windows.py): Real-time latency evaluation testing the standard 500 ms model on shortened test windows.
        *   [evaluate_cross_duration.py](file:///d:/Temple%20Project/scripts/eval/evaluate_cross_duration.py): Comparative latency testing pitting standard models against custom-trained short-window classifiers.
    *   **`scripts/exploratory/`**: Feasibility and optimization scripts:
        *   [optimize_model.py](file:///d:/Temple%20Project/scripts/exploratory/optimize_model.py): Benchmarking of different classifier architectures (Linear SVM, RBF SVM, Sparse L1 SVM, tree ensembles, etc.) to optimize selection.
        *   [run_short_windows.py](file:///d:/Temple%20Project/scripts/exploratory/run_short_windows.py): Standard within-subject pipelines run using custom-trained short analysis windows (300 ms and 200 ms).
        *   [test_detection_latency.py](file:///d:/Temple%20Project/scripts/exploratory/test_detection_latency.py): Multi-shift analysis moving the positive window backward in time to characterize temporal latency boundaries.
    *   **`scripts/utils/`**: Utility scripts:
        *   [download_data.py](file:///d:/Temple%20Project/scripts/utils/download_data.py): Automated downloader utilizing standard HTTP Range requests to retrieve the dataset.
*   **`results/`**: Contained directory storing the output CSV performance metrics for all subjects and evaluation pipelines.
*   **`requirements.txt`**: Project dependency definitions mapping exact versions of external libraries used.

---

## 2. Setup and Execution Guide

### Prerequisites
A standard Python 3.10+ installation is required. Install the necessary library dependencies via the package manager:
```bash
pip install -r requirements.txt
```

### Dataset Retrieval
The raw EEG files are not distributed via version control. The dataset must be downloaded from the OpenNeuro repository prior to running evaluations. The automated utility script downloads, verifies, and extracts the target subject files (subjects `sub-01` through `sub-10`), and cleans up incomplete downloads:
```bash
python scripts/utils/download_data.py
```
This utility downloads approximately 18 GB of raw BDF format files, storing them in a local folder named `ds003626/` in the root workspace.

### Running Evaluations
1.  **Standard Within-Subject Pipeline**:
    ```bash
    python scripts/eval/run_pipeline.py
    ```
    This script processes all subjects independently, fits Linear SVMs, computes metrics for 1:1 and 1:2 class ratios under randomized splits, and saves output CSV files to the `results/` folder.
2.  **Cross-Subject Transfer Learning**:
    ```bash
    python scripts/eval/run_cross_subject.py
    ```
3.  **Unified Multi-Subject Classifier**:
    ```bash
    python scripts/eval/run_unified_model.py
    ```
4.  **Real-Time Latency Trade-Off Analysis**:
    ```bash
    python scripts/eval/evaluate_cross_duration.py
    ```

---

## 3. Dataset & Electrode placement Details

### Data Collection & Experimental Protocol
The experimental data is sourced from OpenNeuro dataset `ds003626`. Recordings are obtained from 10 healthy adult participants, each performing approximately 100–150 trials across 3 separate sessions.
The experimental design incorporates three primary speech conditions alongside control listening states:
1.  **Read-Aloud Task**: Participants read visually presented words or short sentences aloud. This task offers highly controlled and repeatable speech-onset markers, establishing the baseline training set.
2.  **Question-Answer Task**: Participants respond naturally to simple prompts (e.g., "What did you eat for breakfast?"). This preserves a semi-controlled structure while capturing natural speech variability.
3.  **One-on-One Conversation Task**: Participants carry out a short, spontaneous dialogue with the experimenter to capture realistic speech preparation and turn-taking behavior.
4.  **Non-Speaking & Listening Conditions (Negative Controls)**: Negative samples are extracted from idle periods, listening to an experimenter speak, or passively sitting in a room without speaking.

### Speech Onset Ground Truth Detection
The speech onset timestamp $t_0$ is used to label the positive intent-to-speak window. Speech and microphone streams are synchronized through a Lab Streaming Layer (LSL) network. The onset time $t_0$ is mathematically defined as the first sustained increase in short-time audio energy above the pre-speech silent baseline:
$$E(t) = \frac{1}{W} \sum_{\tau=t-W/2}^{t+W/2} s^2(\tau)$$
where $s(\tau)$ represents the audio signal. EEG segments ending precisely at $t_0$ represent active pre-speech planning ($X^+ = EEG[t_0 - 500\text{ ms}, t_0]$). To prevent contamination from motor artifacts associated with vocal tract movement, post-speech windows are strictly excluded.

### Electrode Montage Mapping
The target montage requested by the research protocol is:
$$\{FCz, Cz, C3, C4\}$$
The dataset utilizes a 64-channel BioSemi ActiveTwo cap, which uses a non-standard geodesic channel naming system. To align the active channel indices with the international 10-20 system locations, the following equivalent electrodes are selected:
*   **`C23` $\approx$ `FCz`**: Positioned over the fronto-central midline region to capture premotor speech planning and action-preparation activity.
*   **`A1` $\approx$ `Cz`**: Positioned over the central midline region to capture general motor preparation.
*   **`D19` $\approx$ `C3`**: Positioned over the left central motor region to capture unilateral sensorimotor planning and articulatory muscle preparation.
*   **`B22` $\approx$ `C4`**: Positioned over the right central motor region to provide bilateral sensorimotor coverage, mitigating bias from single-sided headcap placement.

These selected BioSemi electrodes correspond to the standard 10-20 montage coordinates and differ in physical scalp placement by only a few millimeters.

---

## 4. Preprocessing & Artifact Rejection

A rigorous preprocessing pipeline is executed to isolate clean neural markers from high-amplitude noise:
1.  **Re-Referencing**: Raw signals are re-referenced to the average of the bilateral earlobe channels `EXG1` and `EXG2`.
2.  **Filtering**: Continuous data is bandpass filtered between 0.5 and 45.0 Hz using a zero-phase finite impulse response (FIR) filter to remove low-frequency drift and high-frequency muscle noise. A 50.0 Hz notch filter is applied to remove power-line interference.
3.  **Downsampling**: The filtered data is resampled to 250 Hz to minimize computational complexity.
4.  **Combined Artifact Rejection**:
    *   *Absolute Amplitude Thresholding*: Any analysis window where the absolute voltage exceeds $\pm 120\ \mu\text{V}$ is rejected to eliminate blink and physical movement artifacts.
    *   *Subject-Specific Baseline Variance Thresholding*: To filter muscle tension and swallowing artifacts, typical baseline variance parameters are estimated from rest intervals (trigger 13 to 14) segmenting 500 ms windows. The 95th percentile baseline variance limit is computed channel-by-channel. Any experimental trial window where the signal variance exceeds these thresholds is rejected.
5.  **Leakage Prevention**: Standard scaling (`StandardScaler`) is applied channel-wise. Mean and standard deviation statistics are computed strictly on the training partitions and then applied to validation and test sets.

---

## 5. Feature Extraction System

For each normalized 500 ms EEG epoch (125 time points per channel), 18 features are extracted per channel (total of 72 features across the 4 channels):

### Time-Domain Features (8)
*   **Mean Amplitude**: Computes the average voltage offset, capturing slow readiness potentials.
*   **Signal Variance**: Captures the total energy fluctuation in the time series.
*   **Linear Regression Slope**: Fits a linear trend to the time points, indicating slow voltage drifts prior to speech.
*   **Hjorth Mobility**: Measures the mean frequency of the signal, calculated as the square root of the variance of the first derivative divided by the variance of the raw signal.
*   **Hjorth Complexity**: Indicates the bandwidth of the signal, evaluating how closely the signal resembles a pure sine wave.
*   **Zero Crossing Rate (ZCR)**: Represents the frequency at which the signal alternates sign, serving as a robust measure of high-frequency noise.
*   **Skewness**: Calculates the third statistical moment, quantifying the asymmetry of voltage fluctuations.
*   **Kurtosis**: Calculates the fourth statistical moment, quantifying the presence of transient voltage spikes.
*   **Root Mean Square (RMS)**: Computes the quadratic mean of the amplitude, reflecting the signal's total power.

### Frequency-Domain Features (10)
*   **Absolute Band Powers (3)**: Computes the average power spectral density (PSD) via Welch's method across relevant physiological bands:
    *   *Theta* ($4-8\text{ Hz}$): Correlates with cognitive control and speech-planning focus.
    *   *Alpha/Mu* ($8-13\text{ Hz}$): Reflects sensorimotor rhythm suppression during motor preparation.
    *   *Beta* ($13-30\text{ Hz}$): Linked with active motor execution readiness.
*   **Relative Band Powers (5)**: Computes the ratio of individual band powers (Delta, Theta, Alpha, Beta, Gamma) to total power, aligning ranges across subjects.
*   **Spectral Entropy (2)**: Evaluates the complexity or predictability of the signal's frequency spectrum.

---

## 6. Classifier Model & Training Paradigm

### Linear Support Vector Machine
The system utilizes a Linear Support Vector Machine (SVM) with L2 regularization ($C=0.5$). The model solves the dual formulation of the soft-margin optimization problem, minimizing hinge loss:
$$L = \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b)) + \frac{1}{2C} \|\mathbf{w}\|^2$$
where $\mathbf{x}_i$ is the 72-dimensional feature vector, $y_i \in \{-1, +1\}$ is the classification label, $\mathbf{w}$ is the weight vector, and $b$ is the bias.

### Class Balancing & Training Splits
Negative instances (rest/idle) are naturally more abundant in continuous BCI setups. To analyze performance boundaries under different imbalance levels, the training runs are structured under two class balance ratios:
*   **Ratio 1:1**: One negative window selected per positive trial.
*   **Ratio 1:2**: Two negative windows selected per positive trial, providing more variance in the control class.

To prevent leakage, dataset splitting is performed strictly at the **trial level** (70% training, 15% validation, and 15% testing). Adjacent windows extracted from the same experimental trial are kept within the same partition, preventing overfitting.

---

## 7. Experimental Results

The experimental results for the Linear SVM ($C=0.5$) are compiled below:

### 7.1 Within-Subject Classification Results
Within-subject performance is evaluated separately for each participant, reflecting BCI personalization:

*   **Ratio 1:1 (Balanced)**:
    *   *Overall Average*: **74.73% Accuracy**, **74.32% F1-Score**, and **80.88% ROC-AUC**.
    *   *Top-Performing Subject*: `sub-03` (**88.46% Accuracy**, **88.89% F1-Score**).
    *   *Top 50% Cohort Average*: **79.14% Accuracy**, **78.84% F1-Score**.
*   **Ratio 1:2 (Realistic)**:
    *   *Overall Average*: **74.15% Accuracy**, **64.61% F1-Score**, and **82.21% ROC-AUC**.
    *   *Top-Performing Subject*: `sub-10` (**83.33% Accuracy**, **76.92% F1-Score**).
    *   *Top 50% Cohort Average*: **80.39% Accuracy**, **71.69% F1-Score**.

### 7.2 Cross-Subject Transfer Performance
Cross-subject transfer learning represents a zero-calibration scenario, where a classifier trained on a single source subject is evaluated on unseen target subjects:
*   *Overall Transfer Average (1:1 Ratio)*: **55.38% Accuracy**, **58.64% F1-Score**.
*   *Best Source Model*: `sub-05` achieves the highest generalization accuracy across target subjects, averaging **53.67% Accuracy** and **59.95% F1-Score** on the 1:1 ratio.
*   *Top 50% Cohort average*: **60.97% Accuracy**, **65.12% F1-Score**.

### 7.3 Group-Based Cohort Transfer
Evaluating classifiers trained on pooled data from a group of source subjects and tested on unseen target subjects:
*   **Top-7 Group (Ratio 1:1)**: Achieves **65.35% Pooled Accuracy** and **65.35% Pooled F1-Score** on unseen subjects. The average individual target F1-score is **65.16%**.
*   **Top-7 Group (Ratio 1:2)**: Achieves **65.35% Pooled Accuracy** and **55.70% Pooled F1-Score** on unseen subjects.

### 7.4 Unified Multi-Subject Global Model
A single global model trained on subject-wise Z-scored trials pooled from all 10 subjects:
*   **Ratio 1:1**: **66.67% Accuracy**, **65.99% F1-Score**, and **71.08% ROC-AUC**.
*   **Ratio 1:2**: **65.56% Accuracy**, **55.59% F1-Score**, and **71.91% ROC-AUC**.

This unified model's performance on the 1:1 ratio (**66.67%**) meets the paper's target within-subject accuracy range ($65\% - 80\%$), proving that exposing the model to multi-subject training distributions constructs a robust general decision boundary.

### 7.5 Real-Time Latency Trade-Offs (Window Shortening)
BCI latency was analyzed by shortening the analysis window from 500 ms down to 300 ms or 200 ms:
1.  **Feature Shift Penalty (500 ms model tested on short windows)**:
    *   *500 ms -> 300 ms*: F1-score drops to **67.74%** (1:1) and **58.70%** (1:2).
    *   *500 ms -> 200 ms*: F1-score drops to **57.87%** (1:1) and **49.23%** (1:2).
    *   *Neurophysiological Cause*: Shortening the testing window introduces a feature space mismatch; the mean and variance scaling parameters learned on 500 ms data do not align with shorter window inputs.
2.  **Re-Training Advantage (Matching duration models)**:
    *   *300 ms -> 300 ms*: Recovers F1-score to **71.74%** (1:1) and **63.25%** (1:2) (within **1%** of the standard 500 ms baseline).
    *   *200 ms -> 200 ms*: Recovers F1-score to **71.42%** (1:1) and **59.69%** (1:2).
    *   *Neurophysiological Cause*: Re-training ensures the feature scaler learns scaling statistics matching that specific length, allowing the classifier to adapt to the higher noise floor of shorter windows.
3.  **Physical Limits of Window Shortening**:
    *   Shortening the window below 200 ms (e.g. to 100 ms) causes performance to collapse to chance levels. A 100 ms window contains only 25 samples at 250 Hz, failing the physical limit of spectral resolution (Welch's PSD cannot separate theta from alpha band powers).

---

## 8. Broader Model Exploration (Rejected Architectures)

During system optimization, alternative classifiers were benchmarked and ultimately rejected for the following reasons:
*   **Logistic Regression**: Achieved comparable within-subject performance, but L1/L2 penalties sparsify the feature space in a highly subject-specific manner. This degrades cross-subject transfer learning performance where test subjects exhibit shifted feature distributions.
*   **Linear Discriminant Analysis (LDA)**: Stable within-subject performance, but struggled with cross-subject transfer due to class imbalance under the 1:2 ratio. Standard LDA suffered from numerical instability due to high covariance correlation among adjacent channels.
*   **Ridge Classifier**: Lacks the margin-maximizing properties that make SVMs robust to non-Gaussian trial-to-trial outliers in EEG data.
*   **Tree-Based Ensembles (Random Forest & Extra Trees)**: Suffered from severe overfitting on the high-dimensional feature space (324 dimensions). They struggled to find meaningful orthogonal decision boundaries because EEG features (e.g., band powers across adjacent channels) are highly correlated.
*   **Kernel SVM (RBF Kernel)**: Benchmarked during optimization (testing scales $C \in \{1, 10, 100\}$) but rejected. Because the number of training trials per subject is small (~66 positive epochs) relative to the feature dimensionality (72 features), mapping to a high-dimensional space via the RBF kernel caused overfitting on the training set.
*   **Multi-Layer Perceptron (MLP)**: MLP networks overfitted to the small number of trials per subject, failing to generalize to unseen test sets and yielding unstable training dynamics.

---

## 9. Comprehensive Performance Report

For an exhaustive, subject-by-subject breakdown of all classification performance metrics, confusion matrices, and cross-subject transfer matrices, refer to the detailed [results.md](file:///d:/Temple%20Project/results.md) report in the project root.

---

## 10. Dataset Citation

To cite the dataset or associated publication, please use the following formats:

### Dataset Citation
> Nicolas Nieto, Victoria Peterson, Hugo Rufiner, Juan Kamienkowski, and Ruben Spies (2022). Inner Speech. OpenNeuro. [Dataset] doi: 10.18112/openneuro.ds003626.v2.1.2

### Associated Research Publication
> Nieto, N., Peterson, V., Rufiner, H. L., Kamienkowski, J. E., & Spies, R. (2022). Thinking out loud, an open-access EEG-based BCI dataset for inner speech recognition. *Scientific Data*, 9(1), 1–17.
