# Evaluation of Python Toolkits for ECG Simulation, Signal Processing, Augmentation, and Empirical Seizure Characterization

A systematic methodological investigation into Python biosignal toolkits (**NeuroKit2**, **physioKIT**, **torch_ecg**, **BioSPPy**, **WFDB**, and **SciPy**), progressing from synthetic waveform simulation to real clinical seizure ECG characterization and empirically calibrated parametric synthesis.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Concepts and Terminology](#2-key-concepts-and-terminology)
3. [Experiment-by-Experiment Explanation](#3-experiment-by-experiment-explanation)
   - [Experiment 01 — Systematic Generation Parameter Sweeps](#experiment-01--systematic-generation-parameter-sweeps-neurokit2--physiokit)
   - [Experiment 02 — physioKIT Pathological Presets & Clinical Terminology Audit](#experiment-02--physiokit-pathological-presets--clinical-terminology-audit)
   - [Experiment 03 — Dynamic Time-Varying Proof-of-Concept Seizure Simulation](#experiment-03--dynamic-time-varying-proof-of-concept-seizure-simulation)
   - [Experiment 04 — Native torch_ecg Augmentation Audit & Task-Dependent Risk Assessment](#experiment-04--native-torch_ecg-augmentation-audit--task-dependent-risk-assessment)
   - [Experiment 05 — Signal Acquisition & Data Representation Audit](#experiment-05--signal-acquisition--data-representation-audit)
   - [Experiment 06 — Quantitative Noise Filtering Benchmark across Target SNRs](#experiment-06--quantitative-noise-filtering-benchmark-across-target-snrs)
   - [Experiment 07 — R-Peak Detection Performance Benchmark](#experiment-07--r-peak-detection-performance-benchmark-10-mit-bih-records)
   - [Experiment 08 — Real Epileptic Seizure & Arrhythmia Characterization Pipeline](#experiment-08--real-epileptic-seizure--cardiac-arrhythmia-empirical-characterization-pipeline)
   - [Experiment 09 — Empirically Calibrated Parametric Synthetic ECG Generation Engine](#experiment-09--empirically-calibrated-parametric-synthetic-ecg-generation-engine)
4. [Experiments 10 and 11 — STILL IN PROGRESS](#4-experiments-10-and-11--still-in-progress)
   - [Experiment 10 — Quantitative Real vs. Synthetic Feature Validation](#experiment-10--quantitative-real-vs-synthetic-feature-validation-in-progress)
   - [Experiment 11 — Downstream AI Seizure Detection Evaluation Scaffold](#experiment-11--downstream-ai-seizure-detection-evaluation-scaffold-in-progress)
5. [Current Research Pipeline](#5-current-research-pipeline)
6. [Relationship to Deep Generative Models](#6-relationship-to-deep-generative-models)
7. [Presentation-Friendly Summary](#7-presentation-friendly-summary)
   - [30-Second Explanation](#30-second-explanation)
   - [1-Minute Explanation](#1-minute-explanation)
   - [Main Research Question](#main-research-question)
   - [Main Contribution So Far](#main-contribution-so-far)
   - [Biggest Current Limitation](#biggest-current-limitation)
   - [Logical Next Step](#logical-next-step)
8. [Repository Structure & Execution](#8-repository-structure--execution)

---

# 1. Project Overview

### Project Evolution and Progression

This project explores how computational tools can generate, process, augment, and characterize electrocardiogram (**ECG**) signals, with a specific focus on capturing cardiovascular shifts associated with epileptic seizures. The research evolved through eleven sequential experimental steps:

1. **Initial ECG simulation and toolkit exploration:** Surveying the biosignal software ecosystem to identify which libraries natively synthesize ECG waveforms versus those designed solely for digital signal processing (**DSP**) or database access.
2. **Generation parameter investigation:** Quantitatively testing how synthetic generators respond to adjustments in heart rate (**HR**), heart rate variability (**HRV**), sampling frequency ($f_s$), duration, and noise.
3. **Pathological waveform exploration:** Benchmarking preset-based generators to evaluate synthetic cardiac abnormalities and establishing precise clinical terminology (distinguishing arrhythmias from structural, ischemic, and conduction abnormalities).
4. **Seizure-related proof-of-concept simulation:** Constructing a multi-stage synthetic episode that models the autonomic tachycardia surge and muscle tremor artifact observed during epileptic seizures using piecewise rule-based scripts.
5. **Augmentation audit:** Evaluating native deep-learning transformation techniques (tensor-based baseline wander, amplitude scaling, polarity inversion, masking, and CutMix) and categorizing their diagnostic risks across distinct downstream tasks.
6. **Real ECG acquisition and signal processing:** Comparing the data structures, physical calibration units (arbitrary units vs. physical millivolts), and input/output characteristics of synthetic generators against clinical PhysioNet data.
7. **R-peak / RR / HRV analysis:** Benchmarking five automated peak detection algorithms on clinical records to establish standard R-peak extraction accuracy and verify beat-to-beat timing fidelity.
8. **Real seizure ECG characterization:** Harvesting genuine clinical recordings from the PhysioNet Post-Ictal Heart Rate Oscillations in Partial Epilepsy Database (**SZDB**) across confirmed seizure phases (interictal, preictal, ictal, postictal) and an arrhythmia control from the MIT-BIH Arrhythmia Database (**MITDB**) to extract empirical physiological feature distributions without synthetic assumptions.
9. **Empirically informed synthetic ECG generation:** Closing the loop by building a parametric, optimization-calibrated generator that matches synthetic signals to real empirical physiological targets (mean heart rate, heart rate variability, and spectral noise bands).
10. **Quantitative validation (still in progress):** Comparing real and synthetic feature distributions using statistical distance metrics (Wasserstein distance, Kolmogorov-Smirnov tests, power spectral density mean squared error, and template correlation).
11. **Downstream evaluation (still in progress):** Scaffolding a machine learning classification setup to evaluate whether synthetic augmentation impacts model performance on held-out real patient recordings, while examining critical methodological vulnerabilities like data leakage.

```
Initial Simulation & Toolkit Exploration (Exp 01)
  ↓
Generation Parameter Investigation (Exp 01)
  ↓
Pathological Waveform Exploration (Exp 02)
  ↓
Seizure-Related Proof-of-Concept Simulation (Exp 03)
  ↓
Augmentation Capabilities & Diagnostic Risk Audit (Exp 04)
  ↓
Real ECG Acquisition & Signal Processing Audit (Exp 05)
  ↓
Noise Filtering & Quantitative SNR Benchmark (Exp 06)
  ↓
R-Peak Detection & Beat-Timing Benchmark (Exp 07)
  ↓
Real Seizure ECG Empirical Feature Extraction (Exp 08)
  ↓
Empirically Informed Parametric ECG Generation (Exp 09)
  ↓
Quantitative Real-vs-Synthetic Fidelity Validation (Exp 10) [🚧 IN PROGRESS]
  ↓
Downstream Utility Evaluation & Leakage Audit (Exp 11) [🚧 IN PROGRESS]
```

### Parametric Modeling vs. Deep Generative Networks

A central architectural aspect of this repository is that **the current synthetic generation engine is a PARAMETRIC / MODEL-BASED approach. It is NOT a Variational Autoencoder (VAE), Generative Adversarial Network (GAN), or Diffusion Model.**

- **Parametric / Model-Based Generation:** Synthesizes waveforms by integrating differential equations derived from biophysical or kinematic principles (specifically the McSharry et al. **ECGSYN** ordinary differential equation system). The generator adjusts explicit scalar parameters—such as target heart rate, rhythm variability standard deviation, and additive noise variance—to minimize the discrepancy between synthetic outputs and empirically extracted target features.
- **Deep Generative Modeling (Not Implemented Here):** Trains multi-layer neural networks (e.g., latent diffusion models, autoencoding transformers, or adversarial generator-discriminator pairs) to implicitly approximate the high-dimensional probability distribution of raw ECG voltage vectors directly from thousands of training exemplars.

### Methodological Groupings

The repository is divided into two distinct research phases:

- **Experiments 01–07 (Methodological & Exploratory Foundation):** These experiments establish baseline software capabilities, measure algorithm execution speeds, verify digital filtering gains, audit tensor augmentation risks, and evaluate QRS detector accuracy across six Python packages. They represent software and digital signal processing benchmarks.
- **Experiments 08–09 (Real-Data-Informed Research Direction):** These experiments transition away from arbitrary, hand-crafted simulation heuristics. They compute empirical distributions from genuine clinical epilepsy recordings and use numerical optimization to calibrate parametric dynamical simulators directly to real-world physiological targets.
- **Experiments 10–11 (Exploratory Scaffolding — Still in Progress):** These scripts explore quantitative validation metrics and downstream classification pipelines. They are active research prototypes and contain identified methodological constraints that preclude drawing definitive scientific or clinical conclusions.

---

# 2. Key Concepts and Terminology

The following glossary defines the technical abbreviations, signal processing concepts, and clinical terms used throughout this codebase. For each term, the full phrase is expanded on first use, its physiological or mathematical definition is given, and its specific relevance to this repository is explained.

### Biosignal & Signal Processing Terminology

- **ECG — Electrocardiogram:** A non-invasive recording of the electrical potentials generated by the depolarization and repolarization of the cardiac myocardium, measured at the skin surface over time. In this project, ECG is the primary signal modality analyzed for autonomic responses to neurological events.
- **HR — Heart Rate:** The frequency of cardiac ventricular contractions, measured as the number of heartbeats occurring within a standardized interval. In this repository, HR serves as the primary macro-indicator of autonomic nervous system state changes.
- **BPM — Beats Per Minute:** The standard physiological measurement unit for heart rate, calculated either instantaneously from single beat intervals as $60 / \text{RR}_{\text{seconds}}$ or as an aggregate average across a temporal window.
- **R-peak — R-wave Peak:** The prominent upward deflection of the QRS complex, corresponding to the electrical depolarization of the main ventricular muscle mass. Accurate detection of R-peaks is the computational foundation of every rhythm metric in this project.
- **RR interval — R-to-R Peak Interval:** The elapsed time between two consecutive, valid R-wave peaks (expressed in seconds or milliseconds). Sequences of RR intervals form the discrete time series from which all heart rate variability metrics are derived.
- **HRV — Heart Rate Variability:** The physiological phenomenon of beat-to-beat fluctuations in consecutive cardiac cycle durations. HRV reflects the continuous, dynamic interplay between the sympathetic (acceleratory) and parasympathetic/vagal (deceleratory) branches of the autonomic nervous system.
- **SDNN — Standard Deviation of Normal-to-Normal Intervals:** The standard deviation of all physiologically valid RR intervals across a defined time window:
  $$\text{SDNN} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (\text{RR}_i - \overline{\text{RR}})^2}$$
  SDNN measures total autonomic variability. In Experiment 09, SDNN is the primary empirical target used to tune the simulator's rhythm variability parameter.
- **RMSSD — Root Mean Square of Successive Differences:** The square root of the mean squared differences between adjacent RR intervals:
  $$\text{RMSSD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N-1} (\text{RR}_{i+1} - \text{RR}_i)^2}$$
  RMSSD captures high-frequency, beat-to-beat variations reflecting parasympathetic (vagal) cardiac modulation.
- **QRS complex — Ventricular Depolarization Waveform:** The central, dominant morphological feature of an ECG beat, consisting of the downward Q-wave, tall upward R-wave, and downward S-wave. Its duration and shape indicate how electrical activation spreads through the ventricular conduction network.
- **PR interval — Atrioventricular Conduction Interval:** The time elapsed from the beginning of atrial depolarization (P-wave onset) to the onset of ventricular depolarization (Q-wave onset). In this codebase, PR interval extraction was attempted via wavelet delineation, but it was found to yield missing values across low-amplitude recordings and was not used for simulator calibration.
- **QT interval — Total Ventricular Depolarization and Repolarization Interval:** The time elapsed from the start of the Q-wave to the point where the T-wave returns to the isoelectric baseline. It measures the total duration of electrical activation and recovery in the ventricles.
- **QTc — Corrected QT Interval:** The QT interval normalized for heart rate. Because ventricular repolarization naturally shortens as heart rate increases, clinical interpretation requires mathematical rate-correction. This repository implements Bazett's formula:
  $$\text{QTc} = \frac{\text{QT}}{\sqrt{\text{RR}_{\text{seconds}}}}$$
- **ST segment — Isoelectric Ventricular Plateau:** The interval between the end of ventricular depolarization (S-wave or J-point) and the beginning of ventricular repolarization (T-wave). ST-segment elevation or depression indicates myocardial ischemia or injury.
- **PSD — Power Spectral Density:** A frequency-domain representation showing how the total power (variance) of a time-series signal is distributed across individual frequency components. In Experiments 08 and 09, PSD is estimated via Welch's average periodogram method to quantify baseline wander and high-frequency noise bands.
- **SNR — Signal-to-Noise Ratio:** The ratio of true signal power to background noise power, conventionally expressed on a logarithmic decibel scale:
  $$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{\sigma_{\text{signal}}^2}{\sigma_{\text{noise}}^2}\right)$$
- **$\Delta\text{SNR}$ — Change in Signal-to-Noise Ratio:** The quantitative improvement in signal quality achieved by a digital filter, defined as:
  $$\Delta\text{SNR}_{\text{dB}} = \text{SNR}_{\text{output}} - \text{SNR}_{\text{input}}$$
- **EMG — Electromyography:** The electrical potentials generated by skeletal muscle contractions. In this repository, "EMG power" refers specifically to a spectral proxy (high-frequency band energy between $20\text{ Hz}$ and $100\text{ Hz}$) associated with patient movement or seizure tremor, rather than a dedicated physical electromyographic electrode recording.
- **$f_s$ — Sampling Frequency:** The number of discrete voltage samples recorded or generated per second, expressed in Hertz ($\text{Hz}$).
- **Hz — Hertz:** The standard unit of frequency, equal to one cycle or sample per second.
- **mV — Millivolt:** A standard physical unit of electrical potential ($10^{-3}\text{ V}$). Calibrated clinical recordings (such as PhysioNet records) are measured in physical millivolts.
- **a.u. — Arbitrary Units:** A dimensionless scale where numerical amplitudes represent mathematical model states rather than physical voltage measurements. All synthetic signals generated by NeuroKit2 and physioKIT output in arbitrary units.
- **AUROC — Area Under the Receiver Operating Characteristic Curve:** A threshold-independent classification metric measuring the probability that a classifier assigns a higher risk score to a randomly chosen positive case than to a randomly chosen negative case.
- **AUPRC — Area Under the Precision-Recall Curve:** The integral of precision across all recall thresholds, especially informative for severely imbalanced diagnostic datasets where positive events are rare.
- **PPV — Positive Predictive Value (Precision):** The proportion of detected events that are true events:
  $$\text{PPV} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **$F_1$ score — Harmonic Mean of Precision and Recall:** A balanced detection performance index defined as:
  $$F_1 = 2 \cdot \frac{\text{Sensitivity} \cdot \text{PPV}}{\text{Sensitivity} + \text{PPV}}$$
- **MMD — Maximum Mean Discrepancy:** A non-parametric kernel-based statistical metric that measures the distance between two probability distributions in a reproducing kernel Hilbert space.
- **KS — Kolmogorov-Smirnov Test:** A non-parametric statistical hypothesis test that compares the continuous empirical cumulative distribution functions of two samples to test whether they are drawn from the same underlying distribution.
- **WFDB — WaveForm DataBase:** The standardized file format, C library, and Python package developed by PhysioNet for reading, writing, and processing clinical physiological waveforms and annotated event markers.
- **ODE — Ordinary Differential Equation:** A mathematical equation relating a function of a single continuous variable (such as time) to its derivatives. In this project, ODE refers to the dynamic equations governing the state space of the ECGSYN generator.
- **ECGSYN — ECG Dynamical Model Synthesizer:** An ordinary differential equation model introduced by McSharry, Clifford, Tarassenko, and Smith (2003) that generates a realistic synthetic ECG trajectory in 3D state space using angular trajectory equations and Gaussian-shaped wave attractors.

### Clinical & Neurological Terminology

- **Seizure — Epileptic Seizure:** A transient occurrence of signs and symptoms resulting from abnormal, excessive, or synchronous neuronal activity in the brain. Seizures frequently trigger pronounced autonomic shifts, including rapid sinus tachycardia and respiratory irregularity.
- **Interictal:** The baseline period between epileptic seizures, representing normal non-seizure brain activity. In this repository, the interictal state serves as each patient's resting physiological reference point.
- **Preictal:** The time window immediately preceding clinical seizure onset (typically defined here as 1 to 2 minutes prior to electrographic onset). Investigating preictal ECG changes evaluates whether autonomic biomarkers can enable early seizure prediction.
- **Ictal:** The active duration of the clinical or electrographic seizure event, from confirmed onset to cessation. Characterized in the ECG by acute sympathetic acceleration (tachycardia), parasympathetic withdrawal, and somatic muscle contraction tremor.
- **Postictal:** The recovery phase immediately following seizure termination (evaluated here as the first 60 seconds post-offset). Often characterized by rapid heart rate deceleration, transient bradycardia, or profound autonomic instability.
- **Arrhythmia — Cardiac Arrhythmia:** A disruption in the normal rate, rhythm, or sequence of cardiac electrical activation originating from cardiac pathology (e.g., Atrial Fibrillation, Ventricular Tachycardia). In this codebase, arrhythmias serve as "hard negative" controls to ensure seizure detectors respond to neurological autonomic surges rather than intrinsic cardiac conduction defects.
- **Tachycardia:** An abnormally elevated resting heart rate, clinically defined as exceeding $100\text{ BPM}$ in adult humans. Tachycardia is the dominant cardiac manifestation of the sympathetic surge during focal and generalized seizures.
- **ECG Abnormality:** A broad clinical categorization encompassing morphological, conduction, or repolarization deviations from normal sinus rhythm (including bundle branch blocks, hemiblocks, and ST-segment elevations). Not all ECG abnormalities are arrhythmias.

---

# 3. Experiment-by-Experiment Explanation

---

## Experiment 01 — Systematic Generation Parameter Sweeps (NeuroKit2 & physioKIT)

### Research Question
How do the primary synthetic ECG generation functions in **NeuroKit2** (`nk.ecg_simulate`) and **physioKIT** (`pk.ecg.synthesize`) respond to systematic adjustments in heart rate, heart rate variability, additive noise, sampling frequency, duration, and algorithm design? Which parameters are directly comparable, and which represent library-specific features?

### Why This Experiment Was Needed
Before constructing synthetic seizure profiles or augmenting training sets, we needed to know how the available simulation toolkits behave. Running parameter sweeps established whether each generator provides fine-grained, reproducible control over target biophysical characteristics and uncovered library-specific constraints.

### What the Code Does
The script [`Experiments/01_generation_parameter_sweeps.py`](file:///Users/selink/Desktop/ECG/Experiments/01_generation_parameter_sweeps.py) evaluates both libraries through six systematic test sections:
1. **Heart Rate Sweep:** Generates synthetic ECG at target rates of $50$, $70$, $100$, and $150\text{ BPM}$ ($10\text{s}$, $500\text{ Hz}$), runs Pan-Tompkins R-peak detection, and calculates the achieved mean heart rate ($60 / \overline{\text{RR}}$) to determine error.
2. **HRV Sweep:** Varies NeuroKit2's `heart_rate_std` parameter across $[0, 5, 15, 30]\text{ BPM}$ ($30\text{s}$, $500\text{ Hz}$, fixed seed), measures beat-to-beat RR intervals, and records achieved SDNN. It documents that physioKIT does not expose an equivalent API parameter for continuous HRV generation.
3. **Noise Sweep:** Tests nominal noise levels $[0.0, 0.01, 0.05, 0.10]$ in both packages. For NeuroKit2, it computes output signal-to-noise ratio via deterministic clean-reference subtraction. For physioKIT, it measures total signal standard deviation because unseeded phase variations prevent direct sample subtraction.
4. **Sampling Frequency Sweep:** Synthesizes waveforms across $f_s \in [100, 250, 500, 1000]\text{ Hz}$ ($10\text{s}$ duration), verifying sample array lengths and examining morphological fidelity across 3-second windows.
5. **Duration & Scalability Sweep:** Evaluates runtimes for durations of $5$, $10$, and $60\text{ seconds}$ across three timed execution trials to measure generation latency and verify linear scaling.
6. **Simulation Method Comparison:** Benchmarks NeuroKit2's wavelet-based `"simple"` approximation against its 3D ordinary differential equation model `"ecgsyn"`.

### Libraries / Tools
- `neurokit2`: Primary parametric dynamical generator (`ecg_simulate`), signal distortion engine (`signal_distort`), and QRS detector (`ecg_peaks`).
- `physiokit`: Preset-based procedural waveform generator (`pk.ecg.synthesize`).
- `numpy` & `pandas`: Numerical vector operations and structured summary tables.
- `matplotlib`: Multi-panel comparative waveform and metric visualization.

### Important Parameters
- `heart_rate` ($50, 70, 100, 150\text{ BPM}$): Physiologically, this controls the mean cardiac cycle duration ($\overline{\text{RR}} = 60 / \text{HR}$). Computationally, it governs the angular velocity $\omega = 2\pi (\text{HR}/60)$ of the limit cycle trajectory in the ODE integrator or the spacing between concatenated wave segments.
- `heart_rate_std` ($0, 5, 15, 30\text{ BPM}$): Physiologically, represents total heart rate variability standard deviation. Computationally, modulates the internal RR time series generated via respiratory sinus arrhythmia models prior to ODE trajectory integration.
- `noise` / `noise_multiplier` ($0.0$ to $0.10$): Physiologically, simulates baseline wander and electrode contact noise. Computationally, adds scaled Gaussian or colored noise to the generated voltage array.
- `sampling_rate` ($100, 250, 500, 1000\text{ Hz}$): Governs the temporal discretization step $\Delta t = 1 / f_s$. Low values ($100\text{ Hz}$) introduce discretization jitter into sharp QRS peaks; higher values ($1000\text{ Hz}$) preserve sub-millisecond peak timings at increased computational cost.

### Measurements / Metrics
- **Mean RR Interval ($\text{Mean\_RR\_s}$):** The average temporal spacing between detected R-peaks in seconds.
- **Achieved Heart Rate ($\text{Achieved\_HR\_BPM}$):** Calculated as $60 / \text{Mean\_RR\_s}$.
- **Absolute HR Error ($\text{Absolute\_HR\_Error\_BPM}$):** $|\text{Achieved\_HR} - \text{Target\_HR}|$.
- **SDNN ($\text{SDNN\_ms}$):** Standard deviation of normal-to-normal RR intervals in milliseconds, measuring generated beat-to-beat variability.
- **Signal-to-Noise Ratio ($\text{Calculated\_SNR\_dB}$):** Logarithmic power ratio of clean synthetic reference to subtracted additive noise: $10 \log_{10}(\sigma_{\text{clean}}^2 / \sigma_{\text{diff}}^2)$.
- **Median Runtime ($\text{Median\_Runtime\_ms}$):** Median wall-clock duration required to execute the generation function across three runs.

### Outputs
- Figures: `outputs/01a_heart_rate_comparison.png`, `outputs/01b_hrv_control.png`, `outputs/01c_noise_comparison.png`, `outputs/01d_sampling_rate_comparison.png`, `outputs/01e_duration_scalability.png`, `outputs/01f_neurokit_method_comparison.png`, and comprehensive overview `outputs/01_generation_parameter_sweeps.png`.
- Summary CSVs: `outputs/01a_heart_rate_comparison_summary.csv`, `outputs/01b_hrv_control_summary.csv`, `outputs/01c_noise_comparison_summary.csv`, `outputs/01d_sampling_rate_summary.csv`, `outputs/01e_duration_scalability_summary.csv`, `outputs/01f_neurokit_method_summary.csv`, and `outputs/01_parameter_control_capability_matrix.csv`.

### What I Learned
Both toolkits accurately match target mean heart rates ($\text{error} \le 0.14\text{ BPM}$). However, their feature sets differ significantly: NeuroKit2 provides direct, continuous control over rhythm variability (`heart_rate_std`) and generates smooth dynamical trajectories via ODE integration, whereas physioKIT lacks a high-level API for continuous HRV adjustment. Conversely, physioKIT executes faster at long durations because its preset-based synthesis avoids the iterative ODE integration steps required by ECGSYN. At high variability settings (`heart_rate_std = 30`), NeuroKit2's ECGSYN integrator hits numerical bottlenecks under fixed seeds, establishing an operational upper bound for stable simulation.

### Limitations
This experiment evaluates only synthetic models operating under ideal, noise-free or artificially perturbed conditions. It does not measure clinical realism, does not validate synthetic signals against patient recordings, and outputs amplitudes strictly in arbitrary units ($\text{a.u.}$) that cannot be compared to physical millivolts.

### Connection to the Next Experiment
Having established that NeuroKit2 is optimal for continuous parametric heart-rate and HRV sweeps, we turned in Experiment 02 to physioKIT to evaluate its specialized capability: generating discrete pathological rhythm and conduction presets.

---

## Experiment 02 — physioKIT Pathological Presets & Clinical Terminology Audit

### Research Question
Which pathological ECG waveforms can **physioKIT** generate via its native preset library (`pk.ecg.EcgPreset`), how should these waveforms be classified clinically, and what are their limitations when considered for diagnostic training?

### Why This Experiment Was Needed
Machine learning studies frequently misuse clinical terminology, labeling structural, ischemic, or conduction defects as "arrhythmias." This experiment was designed to audit every preset provided by physioKIT, classify each under correct cardiological definitions, and evaluate whether synthetic presets are suitable for diagnostic algorithm pretraining.

### What the Code Does
The script [`Experiments/02_physiokit_arrhythmia_presets.py`](file:///Users/selink/Desktop/ECG/Experiments/02_physiokit_arrhythmia_presets.py) iterates through all eight members of `pk.ecg.EcgPreset`:
1. Synthesizes a 5-second, 12-lead ECG at $500\text{ Hz}$ and baseline $70\text{ BPM}$ with nominal noise ($0.02$).
2. Extracts Lead II (index 1) for visualization and morphological analysis.
3. Maps each preset to its formal clinical category, morphological signature, and classification status (Arrhythmia vs. Conduction/Ischemic Abnormality).
4. Generates a vertical comparison plot and exports an audit summary table.

### Libraries / Tools
- `physiokit`: Procedural synthesis engine (`pk.ecg.synthesize`) and preset enumerations (`pk.ecg.EcgPreset`).
- `matplotlib`: Stacked multi-lead waveform visualization.
- `csv`: Tabular audit export.

### Important Parameters
- `preset` (`pk.ecg.EcgPreset`): Selects the mathematical template for waveform generation:
  - `SR` (Sinus Rhythm): Normal baseline with standard P, QRS, and T intervals.
  - `AFIB` (Atrial Fibrillation): **Cardiac Arrhythmia.** Characterized by chaotic baseline fibrillatory waves, absent discrete P-waves, and irregular R-R interval pacing.
  - `ant_STEMI` (Anterior ST-Elevation Myocardial Infarction): **Ischemic Injury Abnormality (Not an arrhythmia).** Characterized by elevated ST-segments above the baseline, indicating acute coronary artery occlusion.
  - `LAHB` (Left Anterior Hemiblock): **Conduction System Abnormality.** Conduction block in the anterior fascicle of the left bundle branch producing left axis deviation.
  - `LPHB` (Left Posterior Hemiblock): **Conduction System Abnormality.** Conduction block in the posterior fascicle producing right axis deviation.
  - `high_take_off` (Early Repolarization Variant): **Repolarization Abnormality.** Elevated J-point with concave ST-segment elevation, representing a benign or variant repolarization pattern.
  - `LBBB` (Left Bundle Branch Block): **Conduction System Abnormality.** Widened QRS complex ($>120\text{ ms}$) with notched R-waves, reflecting delayed left ventricular activation.
  - `random_morphology`: Stochastic procedural randomization of wave amplitudes and intervals.

### Measurements / Metrics
- **Clinical Category:** Formal diagnostic grouping (Normal Baseline, Cardiac Arrhythmia, Ischemic Pathology, Conduction Abnormality, Repolarization Variant, Stochastic Synthesis).
- **Is Arrhythmia (Yes/No):** Strict clinical binary flag denoting whether the condition involves an intrinsic disturbance in cardiac rhythm timing and pacing.
- **Morphological Signature:** Qualitative description of the hallmark wave deformation (e.g., widened QRS, elevated J-point, absent P-wave).

### Outputs
- Figure: `outputs/02_physiokit_arrhythmia_presets.png` (8 stacked subplots showing Lead II traces).
- Summary CSV: `outputs/02_physiokit_presets_summary.csv` (complete clinical classification and audit table).

### What I Learned
Only **one** of physioKIT's presets—`AFIB`—is a true cardiac arrhythmia. The others (`ant_STEMI`, `LAHB`, `LPHB`, `LBBB`, `high_take_off`) represent structural conduction delays, ischemic repolarization changes, or anatomical axis shifts. Lumping all pathological waveforms under "arrhythmia" is clinically inaccurate. Furthermore, because these presets rely on idealized rule-based mathematical waveforms rather than clinical patient recordings, their utility for pretraining neural networks cannot be assumed without formal downstream evaluation.

### Limitations
Waveforms are deterministic, repetitive mathematical constructs lacking physiological beat-to-beat variability, respiratory modulation, or clinical recording noise. Amplitudes are uncalibrated arbitrary units ($\text{a.u.}$).

### Connection to the Next Experiment
Having evaluated static normal and pathological presets, in Experiment 03 we explored how continuous parametric simulation could model time-varying physiological transitions—specifically the autonomic surge that accompanies an epileptic seizure.

---

## Experiment 03 — Dynamic Time-Varying Proof-of-Concept Seizure Simulation

### Research Question
How can standard synthetic ECG toolkits be adapted to simulate a multi-stage epileptic seizure episode, incorporating dynamic heart rate acceleration, autonomic heart rate variability reduction, and motor tremor artifacts?

### Why This Experiment Was Needed
Epileptic seizures cause dynamic cardiovascular changes—such as profound ictal tachycardia and postictal deceleration—alongside intense electromyographic muscle tremor artifacts. Before developing data-driven generation pipelines, we constructed a 60-second proof-of-concept simulation to examine the mechanics of dynamic parameter transitions, time-varying noise envelopes, and signal continuity.

### What the Code Does
The script [`Experiments/03_dynamic_ictal_surge_simulation.py`](file:///Users/selink/Desktop/ECG/Experiments/03_dynamic_ictal_surge_simulation.py) implements a four-stage, 60-second simulation at $500\text{ Hz}$:
1. **Target HR Trajectory Construction:** Defines a continuous heart rate curve across time:
   - $0\text{--}15\text{ s}$ (Interictal Baseline): Resting sinus rhythm at $70\text{ BPM}$.
   - $15\text{--}30\text{ s}$ (Ictal Surge): Linear acceleration ramping from $70\text{ BPM}$ to $160\text{ BPM}$.
   - $30\text{--}45\text{ s}$ (Peak Seizure / Tachycardia Plateau): Sustained tachycardia at $160\text{ BPM}$.
   - $45\text{--}60\text{ s}$ (Postictal Recovery): Linear deceleration from $160\text{ BPM}$ down to $75\text{ BPM}$.
2. **Piecewise Chunk Generation:** Synthesizes the continuous rhythm in sixty 1-second chunks via `nk.ecg_simulate`, dynamically lowering the `heart_rate_std` parameter during the ictal surge ($15.0 \rightarrow 3.0$) to model autonomic vagal withdrawal.
3. **Boundary Discontinuity Alignment:** Inspects voltage boundaries between adjacent chunks and applies DC-offset shifts to eliminate vertical step discontinuities:
   $$\text{offset}_i = \text{chunk}_{i-1}[-1] - \text{chunk}_i[0], \quad \text{chunk}_i \leftarrow \text{chunk}_i + \text{offset}_i$$
4. **EMG Tremor Artifact Synthesis:** Generates bandpass-filtered Gaussian noise ($20\text{--}200\text{ Hz}$) and modulates it with a smooth half-sine envelope active exclusively during the motor seizure phase ($25\text{--}40\text{ s}$).

### Libraries / Tools
- `neurokit2`: Chunk-based parametric ECG generator (`nk.ecg_simulate`).
- `scipy.signal`: 3rd-order Butterworth bandpass filter (`butter`, `filtfilt`) for tremor noise modeling.
- `numpy`: Dynamic trajectory creation, chunk alignment, and envelope modulation.
- `matplotlib`: Two-panel visualization of the autonomic target trajectory and resulting synthetic waveform.

### Important Parameters
- `hr_curve` ($70 \rightarrow 160 \rightarrow 75\text{ BPM}$): Physiologically, simulates autonomic sympathetic activation followed by gradual postictal recovery. Computationally, provides the target heart-rate parameter for each 1-second segment.
- `std_val` ($15.0$ baseline, $3.0$ ictal surge, $8.0$ recovery): Physiologically, models parasympathetic (vagal) withdrawal during peak sympathetic tone, which suppresses normal beat-to-beat variability.
- `emg_filtered * emg_envelope` ($20\text{--}200\text{ Hz}$, active $25\text{--}40\text{ s}$): Physiologically, approximates generalized tonic-clonic muscle tremors obscuring the surface ECG. Computationally, superimposes bandpass-filtered noise shaped by a squared-sine time envelope:
  $$w(t) = \sin^2\left(\frac{t - 25}{15}\pi\right)$$

### Measurements / Metrics
- **Dynamic Target HR (BPM):** The instantaneous commanded heart rate across the 60-second window.
- **Boundary Voltage Offset (a.u.):** The vertical voltage shift applied to align the first sample of chunk $i$ with the last sample of chunk $i-1$.

### Outputs
- Figure: `outputs/03_dynamic_ictal_surge_simulation.png` (Top panel: target HR profile with phase shading; Bottom panel: continuous simulated ECG with boundary alignment and tremor overlay).

### What I Learned
A time-varying seizure episode can be constructed piecewise from classical simulation tools. However, chunk concatenation introduces two major limitations:
1. Slicing signals into 1-second chunks disrupts dynamical ODE trajectories and requires manual DC-offset adjustments.
2. Most importantly, **all trajectory parameters (the $70 \rightarrow 160\text{ BPM}$ ramp, the $15 \rightarrow 3$ HRV reduction, and the tremor timing) were chosen manually based on heuristic assumptions rather than real patient data.** This highlighted the need for data-driven modeling and motivated the empirical characterization pipeline developed in Experiment 08.

### Limitations
The simulation is purely phenomenological. It lacks genuine neuro-cardiac coupling, models muscle tremor as simple filtered noise rather than physiological motor unit action potentials, uses uncalibrated arbitrary units ($\text{a.u.}$), and was not validated against clinical epilepsy monitoring recordings.

### Connection to the Next Experiment
Before examining real patient data, we explored another common method for expanding ECG datasets: data augmentation using the deep-learning toolkit `torch_ecg`.

---

## Experiment 04 — Native torch_ecg Augmentation Audit & Task-Dependent Risk Assessment

### Research Question
How do native tensor augmentation transforms in **torch_ecg** (`BaselineWanderAugmenter`, `RandomRenormalize`, `RandomFlip`, `RandomMasking`, `CutMix`) alter ECG signals, and how does the safety of each transform depend on the downstream clinical diagnostic task?

### Why This Experiment Was Needed
In computer vision, data augmentation (rotations, crops, flips) is widely used without altering semantic labels. In clinical electrocardiography, however, naive transformations can distort diagnostic features (such as ST-elevation or wave polarity) and invalidate ground-truth labels. We needed to evaluate native `torch_ecg` transformations and systematically assess their clinical risks.

### What the Code Does
The script [`Experiments/04_torch_ecg_augmentations.py`](file:///Users/selink/Desktop/ECG/Experiments/04_torch_ecg_augmentations.py) evaluates five native `torch_ecg.augmenters` classes:
1. Constructs a multi-signal batch of four distinct synthetic ECG signals ($50$, $70$, $120$, and $150\text{ BPM}$, $10\text{s}$, $500\text{ Hz}$) represented as a 3D PyTorch Tensor of shape `(batch_size=4, leads=1, samples=5000)`.
2. Applies each augmenter in a native forward call with probability $p=1.0$:
   - `BaselineWanderAugmenter`: Adds low-frequency sinusoidal drift combined with Gaussian noise.
   - `RandomRenormalize`: Multiplies signal amplitude by a scaling factor ($1.6\times$).
   - `RandomFlip`: Multiplies the entire signal vector by $-1.0$, inverting polarity.
   - `RandomMasking`: Zeroes out random contiguous temporal segments.
   - `CutMix`: Slices and splices segments across different batch elements (e.g., inserting a slice of the $150\text{ BPM}$ signal into the $70\text{ BPM}$ signal).
3. Compares augmented waveforms against the unaugmented baseline and tabulates task-dependent diagnostic risks.

### Libraries / Tools
- `torch_ecg.augmenters`: Deep learning ECG augmentation classes operating on PyTorch Tensors.
- `torch`: Tensor data structures and batch execution.
- `neurokit2`: Baseline multi-rate signal batch generation.
- `matplotlib` & `csv`: Waveform comparison plotting and risk audit export.

### Important Parameters
- `BaselineWanderAugmenter(fs=500, prob=1.0)`: Injects low-frequency baseline drift simulating respiration and patient motion.
- `RandomRenormalize(prob=1.0)`: Rescales signal amplitude while preserving temporal wave timing.
- `RandomFlip(prob=1.0)`: Flips vertical signal polarity, reversing the electrical axis.
- `RandomMasking(fs=500, prob=1.0)`: Zeroes temporal windows to test model robustness to electrode detachment.
- `CutMix(prob=1.0)`: Cuts a temporal segment from one batch signal and pastes it into another, blending labels proportionally.

### Measurements / Metrics
- **Clinical Diagnostic Risk:** Qualitative assessment of which diagnostic interpretations are compromised by the transformation.
- **Task-Dependent Label Preservation:** Explains for which tasks the transform is safe (label-preserving) versus unsafe (label-distorting).

### Outputs
- Figure: `outputs/04_torch_ecg_augmentations.png` (5 subplots showing original vs. transformed signals with highlighted risk callouts).
- Summary CSV: `outputs/04_augmentation_clinical_risk_audit.csv` (detailed audit matrix mapping transforms to task risks).

### What I Learned
Augmentation safety in ECG is strictly **task-dependent**:
- `BaselineWanderAugmenter` is safe for QRS/heart-rate detection, but introduces false positives or negatives in ST-segment ischemic monitoring.
- `RandomRenormalize` preserves rhythm intervals, but alters absolute voltage criteria used to diagnose Left Ventricular Hypertrophy (**LVH**, Sokolow-Lyon index).
- `RandomFlip` preserves beat timing, but inverts P and T waves, mimicking limb-lead reversal or dextrocardia.
- `CutMix` disrupts global rhythm continuity and beat intervals (making it unsuitable for rhythm classification), but can serve as a local feature regularizer in self-supervised representation learning.

### Limitations
Transforms were evaluated using synthetic baselines rather than multi-channel clinical recordings, and were tested with fixed hyperparameters ($p=1.0$) rather than stochastic training regimes.

### Connection to the Next Experiment
Having evaluated synthetic generators and tensor transforms, in Experiment 05 we audited how synthetic data structures and arbitrary units differ from real clinical database acquisitions.

---

## Experiment 05 — Signal Acquisition & Data Representation Audit

### Research Question
Which toolkits natively support signal acquisition (synthetic simulation vs. real database I/O), what data structures do they return, and how do synthetic arbitrary units ($\text{a.u.}$) differ from calibrated physical millivolts ($\text{mV}$)?

### Why This Experiment Was Needed
Different biosignal toolkits serve distinct roles. Conflating uncalibrated model outputs with physical voltage recordings leads to flawed assumptions in downstream feature extraction and filtering. This experiment was designed to audit the exact acquisition mechanisms, data types, and physical units across all six candidate libraries over an identical 10.0-second window.

### What the Code Does
The script [`Experiments/05_signal_acquisition_comparison.py`](file:///Users/selink/Desktop/ECG/Experiments/05_signal_acquisition_comparison.py) attempts signal acquisition across all candidate packages over a strict 10.0-second window:
1. **NeuroKit2:** Generates a synthetic signal via `nk.ecg_simulate(duration=10.0, sampling_rate=500)`.
2. **physioKIT:** Synthesizes a signal via `pk.ecg.synthesize(signal_length=5000, sample_rate=500)`.
3. **BioSPPy:** Inspects library functions, verifying that BioSPPy provides processing routines only and lacks signal acquisition or synthesis APIs.
4. **WFDB:** Reads real clinical data directly from the PhysioNet MIT-BIH Arrhythmia Database (`wfdb.rdrecord("100", sampfrom=0, sampto=3600, pn_dir="mitdb")`).
5. **SciPy:** Verifies that SciPy provides general DSP routines but no domain-specific ECG acquisition tools.
6. Plots all five outputs vertically on aligned time axes, explicitly labeling data sources and physical units.

### Libraries / Tools
- `wfdb`: Native reader for PhysioNet clinical waveform records and annotations.
- `neurokit2` & `physiokit`: Synthetic ECG generation engines.
- `biosppy` & `scipy`: Evaluated to document their lack of native acquisition/synthesis APIs.
- `matplotlib`: Comparative waveform visualization.

### Important Parameters
- `DURATION = 10.0` seconds: Standardized temporal window applied across all panels.
- `wfdb.rdrecord("100", sampfrom=0, sampto=3600)`: Reads the first 10 seconds (3600 samples at $360\text{ Hz}$) of Record 100 from PhysioNet.

### Measurements / Metrics
- **Source Type:** Categorized as Synthetic Dynamical Model, Synthetic Preset Generator, Real Clinical Recording, or None.
- **Output Data Structure:** Identifies the returned Python object (e.g., `np.ndarray`, `tuple`, `wfdb.Record`).
- **Physical Voltage Units:** Documents whether amplitudes represent physical millivolts ($\text{mV}$) calibrated against hardware gains or dimensionless model arbitrary units ($\text{a.u.}$).

### Outputs
- Figure: `outputs/05_signal_acquisition_comparison.png` (5 vertical panels highlighting supported vs. unsupported acquisition workflows).

### What I Learned
Only **WFDB** provides access to calibrated physical clinical recordings ($V \in \text{mV}$). **NeuroKit2** and **physioKIT** produce dimensionless mathematical outputs ($\text{a.u.}$). **BioSPPy** and **SciPy** contain no acquisition functions and operate strictly as downstream processing libraries. This distinction is critical: amplitude-dependent features (such as peak-to-peak voltage or spectral power) cannot be directly compared between synthetic models and clinical recordings without normalization.

### Limitations
Evaluates a single 10-second window from one clinical record (MIT-BIH Record 100) and does not examine multi-channel header metadata or calibration gains in detail.

### Connection to the Next Experiment
Having identified that clinical recordings contain physical noise and baseline wander that synthetic waveforms often omit, in Experiment 06 we benchmarked the digital filtering algorithms used to clean noisy signals.

---

## Experiment 06 — Quantitative Noise Filtering Benchmark across Target SNRs

### Research Question
How effectively do digital filtering pipelines from **BioSPPy**, **physioKIT**, **SciPy**, and **NeuroKit2** remove combined baseline wander and high-frequency noise across standardized input signal-to-noise ratios ($+20\text{ dB}$, $+10\text{ dB}$, $0\text{ dB}$, $-5\text{ dB}$)?

### Why This Experiment Was Needed
Raw ECG recordings are contaminated by low-frequency baseline drift (respiration, patient motion) and high-frequency noise (muscle tremor, powerline interference). Prior to feature extraction, signals must be filtered. This experiment quantitatively measures how much signal quality improves ($\Delta\text{SNR}$) across four standardized noise regimes.

### What the Code Does
The script [`Experiments/06_noise_filtering_snr_benchmark.py`](file:///Users/selink/Desktop/ECG/Experiments/06_noise_filtering_snr_benchmark.py) benchmarks filtering performance:
1. Generates a clean synthetic ECG reference ($10\text{s}$, $1000\text{ Hz}$, $70\text{ BPM}$) using NeuroKit2.
2. Contaminates the clean signal to achieve four target input SNRs: $+20\text{ dB}$ (clean), $+10\text{ dB}$ (moderate noise), $0\text{ dB}$ (heavy noise), and $-5\text{ dB}$ (extreme noise), adding scaled Gaussian white noise and a $0.3\text{ Hz}$ sinusoidal baseline wander.
3. Processes each noisy signal through four toolkit filtering routines:
   - **BioSPPy:** Built-in FIR bandpass filter (`biosppy.signals.ecg.ecg`).
   - **physioKIT:** Forward-backward 3rd-order Butterworth bandpass ($0.5\text{--}35\text{ Hz}$).
   - **SciPy:** 3rd-order Butterworth bandpass ($0.5\text{--}35\text{ Hz}$) combined with an IIR notch filter at $50\text{ Hz}$ ($Q=30$).
   - **NeuroKit2:** Default `ecg_clean` using the NeuroKit bandpass method.
4. Normalizes all filtered outputs to zero mean and unit variance before calculating the output SNR and $\Delta\text{SNR}_{\text{dB}}$ improvement.

### Libraries / Tools
- `biosppy.signals.ecg`: Clinical biosignal analysis suite with automated FIR filtering.
- `physiokit.ecg`: IIR zero-phase Butterworth filter (`clean`).
- `scipy.signal`: Cascaded Butterworth bandpass (`butter`, `filtfilt`) and IIR notch (`iirnotch`).
- `neurokit2`: Automated biosignal cleaner (`ecg_clean`).

### Important Parameters
- `TARGET_SNRS_DB = [20.0, 10.0, 0.0, -5.0]`: Spans the range from high-quality clinical recordings ($+20\text{ dB}$) to severely corrupted recordings ($-5\text{ dB}$).
- `bw = 0.2 * np.sin(2 * np.pi * 0.3 * t)`: Injects a $0.3\text{ Hz}$ baseline drift modeling respiratory artifact.
- `compute_snr_db(reference, test)`: Standardized metric function that normalizes signals to zero mean and unit variance before computing residual error variance.

### Measurements / Metrics
- **Input SNR ($\text{Actual\_SNR\_in\_dB}$):** Realized logarithmic power ratio of clean signal to added noise prior to filtering.
- **Output SNR ($\text{SNR\_out}$):** Logarithmic power ratio of clean reference to the residual difference between normalized filtered output and normalized clean reference:
  $$\text{SNR}_{\text{out}} = 10 \log_{10}\left(\frac{\text{Var}(\text{ref}_{\text{norm}})}{\text{Var}(\text{test}_{\text{norm}} - \text{ref}_{\text{norm}})}\right)$$
- **$\Delta\text{SNR}$ ($\text{Delta\_SNR\_dB}$):** Quantitative improvement:
  $$\Delta\text{SNR}_{\text{dB}} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$$

### Outputs
- Figure: `outputs/06_noise_filtering_snr_benchmark.png` (4 panels showing noisy inputs vs. filtered outputs across all four SNR levels).
- Summary CSV: `outputs/06_snr_filtering_summary.csv` (table of $\Delta\text{SNR}$ values for each toolkit and noise condition).

### What I Learned
Digital filtering provides substantial signal quality gains under heavy noise ($0\text{ dB}$ and $-5\text{ dB}$ input SNR), yielding improvements of $+2.5\text{ dB}$ to $+4.2\text{ dB}$. BioSPPy's linear-phase FIR bandpass filter achieved the highest quantitative $\Delta\text{SNR}$ across all noise levels ($+1.99\text{ dB}$ at $+20\text{ dB}$; $+4.23\text{ dB}$ at $-5\text{ dB}$). However, filtering clean signals ($+20\text{ dB}$) yields modest gains ($<+2\text{ dB}$) because bandpass cutoffs alter physiological ST-segment and T-wave amplitudes.

### Limitations
Noise was simulated as Gaussian white noise combined with a synthetic sine wave, which does not capture non-stationary clinical artifacts like electrode motion or EMG bursts.

### Connection to the Next Experiment
Once an ECG signal is filtered, the next step in any analysis pipeline is detecting R-peaks. In Experiment 07, we evaluated how accurately each toolkit identifies R-peaks on annotated clinical data.

---

## Experiment 07 — R-Peak Detection Performance Benchmark (10 MIT-BIH Records)

### Research Question
What is the quantitative detection performance—measured by Sensitivity, Positive Predictive Value, and $F_1$-score—of R-peak detection algorithms from **WFDB**, **BioSPPy**, **NeuroKit2**, **physioKIT**, and **SciPy** when evaluated against cardiologist-verified `.atr` annotations across 10 MIT-BIH records?

### Why This Experiment Was Needed
R-peak detection is the foundation of cardiac feature extraction: all RR interval calculations, heart-rate variability metrics, and beat-segmented templates depend directly on peak locations. Evaluating detection accuracy across a diverse set of real clinical records with varying noise and arrhythmias establishes which tools are dependable for automated feature extraction.

### What the Code Does
The script [`Experiments/07_qrs_detection_multi_record.py`](file:///Users/selink/Desktop/ECG/Experiments/07_qrs_detection_multi_record.py) implements an EC57-style benchmark:
1. Loads the first 100 seconds ($36,000\text{ samples}$ at $360\text{ Hz}$) of 10 records from the MIT-BIH Arrhythmia Database: `100`, `101`, `102`, `103`, `105`, `106`, `119`, `200`, `201`, and `203`.
2. Extracts cardiologist reference annotations (`.atr`), filtering for valid beat symbols (`N`, `L`, `R`, `B`, `A`, `a`, `J`, `S`, `V`, `F`, `e`, `j`, `n`, `E`, `/`, `f`, `Q`, `?`).
3. Runs automated R-peak detection using each library:
   - **WFDB:** `processing.gqrs_detect`
   - **BioSPPy:** Hamilton-Tompkins segmenter (`biosppy.signals.ecg.ecg`)
   - **NeuroKit2:** Pan-Tompkins algorithm (`nk.ecg_peaks`, method `"neurokit"`)
   - **physioKIT:** Adaptive peak locator (`pk.ecg.find_peaks`)
   - **SciPy:** A custom Pan-Tompkins cascade (Butterworth $5\text{--}15\text{ Hz}$ bandpass, derivative, squaring, moving-window integrator, dynamic thresholding)
4. Matches detected peaks against reference annotations using an EC57-standard $\pm 150\text{ ms}$ tolerance window.
5. Computes True Positives (**TP**), False Positives (**FP**), False Negatives (**FN**), Sensitivity, PPV, and $F_1$-score.

### Libraries / Tools
- `wfdb`: Dataset loader and GQRS peak detection algorithm (`processing.gqrs_detect`).
- `biosppy`: Hamilton-Tompkins QRS segmentation.
- `neurokit2`: Pan-Tompkins / NeuroKit peak detection.
- `physiokit`: Native peak detection routine.
- `scipy.signal`: Custom implementation of the 1985 Pan-Tompkins digital signal processing cascade.
- `matplotlib` & `csv`: Performance bar plots and detailed record-by-record metric exports.

### Important Parameters
- `TOLERANCE_SAMPLES = int(0.150 * 360) = 54 samples` ($\pm 150\text{ ms}$): Standard EC57 temporal matching window. A detected peak is counted as a True Positive if it falls within $150\text{ ms}$ of an unmatched reference annotation.
- `BEAT_SYMBOLS`: Filter set excluding non-beat annotations (e.g., rhythm changes or measurement artifacts).

### Measurements / Metrics
- **True Positive (TP):** A detected peak matching an annotated beat within $\pm 150\text{ ms}$.
- **False Positive (FP):** A detected peak with no corresponding annotated beat within the tolerance window (over-detection).
- **False Negative (FN):** An annotated reference beat that was not detected (missed beat).
- **Sensitivity / Recall ($\text{Sensitivity}$):**
  $$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- **Positive Predictive Value / Precision ($\text{PPV}$):**
  $$\text{PPV} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- **$F_1$-Score:** Harmonic mean of Sensitivity and PPV:
  $$F_1 = 2 \cdot \frac{\text{Sensitivity} \cdot \text{PPV}}{\text{Sensitivity} + \text{PPV}}$$

### Outputs
- Figure: `outputs/07_qrs_detection_multi_record.png` (Comparative bar chart showing Mean $\pm$ Std of Sensitivity, PPV, and $F_1$-score).
- Summary CSV: `outputs/07_qrs_detection_summary.csv` (Overall toolkit aggregate metrics).
- Detailed CSV: `outputs/07_qrs_detection_detailed_results.csv` (Breakdown for every record-toolkit combination).

### Performance Summary Table ($N=10$ Records, First 100s)

| Toolkit | Algorithm | Sensitivity (%) | PPV (%) | $F_1$-Score (%) |
| :--- | :--- | :---: | :---: | :---: |
| **WFDB** | GQRS (`processing.gqrs_detect`) | **99.78 ± 0.35** | 99.69 ± 0.39 | **99.73 ± 0.22** |
| **BioSPPy** | Hamilton-Tompkins Segmenter | 98.93 ± 1.60 | **99.94 ± 0.18** | 99.42 ± 0.89 |
| **NeuroKit2** | Pan-Tompkins (`method="neurokit"`) | 98.73 ± 3.26 | 99.76 ± 0.37 | 99.21 ± 1.70 |
| **physioKIT** | Adaptive Peak Locator | 98.96 ± 3.12 | 99.36 ± 0.61 | 99.13 ± 1.66 |
| **SciPy** | Custom Pan-Tompkins DSP Cascade | 92.64 ± 10.01 | 100.00 ± 0.00 | 95.87 ± 5.86 |

### What I Learned
WFDB's GQRS detector achieved the highest overall $F_1$-score ($99.73\%$), closely followed by BioSPPy ($99.42\%$) and NeuroKit2 ($99.21\%$). However, on challenging records containing frequent premature ventricular contractions and baseline drift (e.g., Record `106` and Record `203`), detection sensitivity dropped noticeably across several algorithms. This underscored that clinical biosignal pipelines require robust, validated peak detectors to prevent false peak detections from corrupting downstream heart rate variability metrics.

### Limitations
Evaluates only the first 100 seconds of each record rather than full 30-minute recordings, and focuses on single-channel Lead II / MLII data without multi-lead spatial cross-validation.

### Connection to the Next Experiment
Having validated signal processing and R-peak detection across standard databases, we moved to the core objective of the research: applying these tools to real clinical seizure recordings in Experiment 08.

---

## Experiment 08 — Real Epileptic Seizure & Cardiac Arrhythmia Empirical Characterization Pipeline

### Research Question
What are the empirical physiological feature distributions (rhythm, heart rate variability, wave morphology, beat templates, spectral noise bands, and signal quality) across distinct clinical seizure phases in the PhysioNet **SZDB** database and a real cardiac arrhythmia control from **MITDB**? Can these empirical distributions replace manually chosen simulation parameters?

### Why This Experiment Was Needed
In Experiment 03, we built a proof-of-concept seizure simulation, but its heart rate ramps, variability drops, and tremor parameters were hand-tuned heuristics. To build credible synthetic models, we must first extract real empirical parameters from clinical recordings. Experiment 08 creates an end-to-end characterization pipeline that processes real patient data without synthetic fallbacks.

### Datasets Processed & Seizure Timing
The pipeline processes two clinical databases from PhysioNet:
1. **PhysioNet szdb (Post-Ictal Heart Rate Oscillations in Partial Epilepsy):**
   Contains multi-hour ECG recordings from patients with partial epilepsy undergoing video-EEG monitoring. The pipeline processes all seven available records (`sz01`, `sz02`, `sz03`, `sz04`, `sz05`, `sz06`, and `sz07`), which encompass **10 EEG-confirmed seizure events**:
   - `sz01`: Seizure 1 ($876\text{--}972\text{ s}$, duration $96\text{s}$)
   - `sz02`: Seizure 1 ($3763\text{--}3823\text{ s}$, duration $60\text{s}$); Seizure 2 ($10551\text{--}10576\text{ s}$, duration $25\text{s}$)
   - `sz03`: Seizure 1 ($5074\text{--}5182\text{ s}$, duration $108\text{s}$); Seizure 2 ($9267\text{--}9377\text{ s}$, duration $110\text{s}$)
   - `sz04`: Seizure 1 ($1210\text{--}1315\text{ s}$, duration $105\text{s}$)
   - `sz05`: Seizure 1 ($1447\text{--}1530\text{ s}$, duration $83\text{s}$)
   - `sz06`: Seizure 1 ($3085\text{--}3139\text{ s}$, duration $54\text{s}$); Seizure 2 ($7485\text{--}7570\text{ s}$, duration $85\text{s}$)
   - `sz07`: Seizure 1 ($4082\text{--}4171\text{ s}$, duration $89\text{s}$)
2. **PhysioNet mitdb (MIT-BIH Arrhythmia Database):**
   Record `203` ($360\text{ Hz}$, $60\text{s}$ window) processed with verified cardiologist `.atr` annotations, serving as a real-world cardiac arrhythmia hard-negative control (Atrial Fibrillation, ventricular flutter, and PVCs).

### Seizure Phase Segmentation Protocol
For each seizure event, the pipeline extracts four non-overlapping temporal phases based on clinical onset ($t_{\text{onset}}$) and offset ($t_{\text{offset}}$):
- **Interictal Baseline:** A 60-second window extracted approximately 10 minutes prior to seizure onset ($[t_{\text{onset}} - 660\text{s}, t_{\text{onset}} - 600\text{s}]$), representing resting non-seizure cardiac activity.
- **Preictal Prediction:** A 60-second window extracted 1 to 2 minutes prior to electrographic seizure onset ($[t_{\text{onset}} - 120\text{s}, t_{\text{onset}} - 60\text{s}]$), capturing early autonomic changes.
- **Ictal Seizure Phase:** Extracted from confirmed onset to offset ($[t_{\text{onset}}, t_{\text{offset}}]$), preserving the full clinical duration (varying from $25\text{s}$ to $110\text{s}$).
- **Postictal Recovery:** The 60-second window immediately following seizure cessation ($[t_{\text{offset}}, t_{\text{offset}} + 60\text{s}]$), capturing postictal deceleration and heart rate instability.

### Metadata Hierarchy and Provenance
The pipeline tracks data provenance using a hierarchical metadata structure:
- `dataset`: Name of the source repository (`szdb` or `mitdb`).
- `record_id`: Physical file identifier (e.g., `sz01`, `sz02`, `203`).
- `seizure_id`: Unique identifier for each event (e.g., `sz02_sz1`, `sz02_sz2`).
- `phase`: Clinical phase (`Interictal_Baseline`, `Preictal_Prediction`, `Ictal_Seizure`, `Postictal_Recovery`, or `Hard_Negative_Arrhythmia`).
- `patient_id`: Set to the specific patient identity when documented (`mitdb_patient_203`), or left explicitly empty/unmapped for `szdb`. **Crucially, the pipeline does not assume `record_id` equals `patient_id`, because multiple recordings in clinical databases can originate from the same individual.**

### Feature Families & Extraction Details

The pipeline computes four feature families across every window:

#### 1. Rhythm and Heart Rate Variability Features
- `Detected_Peaks`: Total number of R-peaks detected via Pan-Tompkins.
- `Valid_RR_Count`: Number of RR intervals satisfying physiological plausibility criteria ($250\text{ ms} \le \text{RR} \le 2500\text{ ms}$, corresponding to $24\text{--}240\text{ BPM}$). Intervals outside this window (e.g., due to dropped beats or motion spikes) are excluded.
- `Mean_HR_BPM`: Mean heart rate, derived from valid RR intervals as the mean of instantaneous rates ($60000 / \text{RR}_{\text{ms}}$).
- `Median_HR_BPM`: Median of instantaneous heart rates, providing a metric robust to occasional outlier intervals.
- `Std_HR_BPM`: Sample standard deviation ($ddof=1$) of instantaneous heart rates.
- `Mean_RR_ms` & `Median_RR_ms`: Average and median normal-to-normal RR intervals in milliseconds.
- `SDNN_ms`: Standard deviation of valid RR intervals, measuring overall autonomic variability.
- `RMSSD_ms`: Root mean square of successive differences between consecutive valid RR intervals, quantifying short-term parasympathetic modulation.
- `RR_Min_ms` & `RR_Max_ms`: Extremal range of valid RR intervals within the window.
- `RR_CV`: Coefficient of variation ($\text{SDNN} / \overline{\text{RR}}$), a normalized metric of rhythm irregularity.
- `RR_IQR_ms`: Interquartile range of RR intervals ($75\text{th} - 25\text{th}$ percentile).

#### 2. Morphological Features and Median Beat Templates
- `QRS_Duration_Median_ms` & `QRS_Duration_IQR_ms`: Derived by automated Discrete Wavelet Transform (**DWT**) delineation (`nk.ecg_delineate`, method `"dwt"`), calculated as the temporal difference between detected S-wave and Q-wave peaks: $(S_{\text{peak}} - Q_{\text{peak}}) / f_s \times 1000$. Plausibility-filtered to $[20\text{ ms}, 300\text{ ms}]$.
- `R_Amplitude_Median_mV`: Median raw ECG voltage at detected R-peak sample indices.
- `QRS_Amplitude_Median_mV`: Median peak-to-peak amplitude across the QRS complex ($|V(S_{\text{peak}}) - V(Q_{\text{peak}})|$).
- `QT_Interval_Median_ms`: Measured from Q-wave peak to T-wave offset ($T_{\text{offset}} - Q_{\text{peak}}$), plausibility-filtered to $[150\text{ ms}, 800\text{ ms}]$.
- `QTc_Median_ms`: Bazett rate-corrected QT interval ($\text{QT} / \sqrt{\overline{\text{RR}}_{\text{sec}}}$).
- `P_Amplitude_Median_mV` & `T_Amplitude_Median_mV`: Median signal amplitudes at detected P-wave and T-wave peaks.
- **Beat Templates (`08_beat_templates.npz`):** Segments each beat across a fixed window from $-200\text{ ms}$ before the R-peak to $+400\text{ ms}$ after the R-peak ($120\text{ samples}$ at $200\text{ Hz}$). Computes the point-wise median beat vector across all clean beats in the window.
- `Beat_Correlation_Median` & `Beat_Correlation_IQR`: Median and IQR of Pearson correlation coefficients between individual beat segments and the median beat template.
- `Beat_RMSE`: Median root mean squared error between individual beats and the median template.
- `Morphology_Variability`: Quantified as $1.0 - \text{Beat\_Correlation\_Median}$, capturing morphological instability.

> [!IMPORTANT]
> **Implementation Verification Note:** In the actual codebase implementation, `PR_Interval_Median_ms` and `ST_Level_Median_mV` are initialized to `np.nan`. Because the automated DWT delineation routine did not reliably identify P-wave onsets and J-points across noisy clinical seizure recordings, these two fields were not successfully computed and remain `NaN`. They are not claimed as valid features in this repository.

#### 3. Spectral Features (Welch Power Spectral Density)
Estimated using Welch's method with a 4-second window ($nperseg = \min(\text{len}, 4 \cdot f_s)$):
- `Total_Spectral_Power`: Integral of the power spectral density across the entire frequency range ($0\text{ Hz}$ to Nyquist $f_s / 2$).
- `Baseline_Wander_Band_Power`: Spectral power integrated over the low-frequency band $[0.0\text{ Hz}, 0.5\text{ Hz})$, measuring respiratory drift and patient movement.
- `ECG_Dominant_Band_Power`: Spectral power integrated over the physiological cardiac band $[0.5\text{ Hz}, 20.0\text{ Hz}]$, capturing the primary energy of QRS complexes and T-waves.
- `EMG_Associated_High_Frequency_Power`: Spectral power integrated over the high-frequency band $[20.0\text{ Hz}, \min(100.0, f_s/2)\text{ Hz}]$. **Crucially, this is a SPECTRAL PROXY for high-frequency noise and tremor; it is NOT a direct electromyographic electrode measurement.**
- `Relative_High_Frequency_Power`: The ratio of high-frequency power to total spectral power ($\text{EMG\_Power} / \text{Total\_Power}$).
- `Spectral_Centroid_Hz`: The power-weighted average frequency ($\sum f \cdot \text{PSD} / \sum \text{PSD}$), reflecting shifts toward higher or lower frequency content.

#### 4. Signal Quality Indicators
- `Signal_Mean_mV`, `Signal_Std_mV`, `Signal_RMS_mV`: Basic statistical moments of the raw voltage signal.
- `Peak_to_Peak_mV`: Total dynamic voltage range ($\max(V) - \min(V)$).
- `Robust_Amplitude_Range_mV`: The difference between the 95th and 5th percentiles ($P_{95} - P_5$), providing an amplitude metric resistant to transient motion spikes.
- `NaN_Fraction`: Fraction of missing or non-finite values. Sparse gaps ($\le 1\%$) are repaired via linear interpolation; windows exceeding $1\%$ are rejected.
- `Clipping_Detected` & `Clipping_Fraction`: Checks whether signal values saturate at identical minimum or maximum extremes for $>1\%$ of samples.
- `signal_valid`, `rhythm_valid`, and `Quality_OK`: Composite Boolean flags verifying that the signal is non-flatline, non-clipped, contains sufficient peaks ($\ge 3$), and satisfies all validity criteria.

### Summary of Empirical Feature Table Outputs

Aggregating valid clinical windows across all records in `outputs/08_empirical_seizure_parameters.csv` reveals clear physiological trends across phases:

| Clinical Phase | Mean HR (BPM) | SDNN (ms) | RMSSD (ms) | High-Freq Power Proxy ($20\text{--}100\text{ Hz}$) | Signal Std (mV) | Total Beats |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Interictal Baseline** | 69.55 | 64.38 | 76.43 | 0.002427 | 0.2867 | 690 |
| **Preictal Prediction** | 71.63 | 79.91 | 96.95 | 0.001865 | 0.2744 | 703 |
| **Ictal Seizure Phase** | **103.82** | **88.95** | 93.61 | **0.003185** | **0.5033** | **1413** |
| **Postictal Recovery** | 89.76 | 100.44 | 104.29 | 0.001635 | 0.3850 | 869 |
| **Hard Negative Arrhythmia** | 112.83 | 206.97 | 269.16 | 0.000375 | 0.5177 | 100 |

### Outputs Generated
- `outputs/08_empirical_ecg_feature_table.csv`: The complete, primary feature dataset containing 73 columns of metadata, rhythm, morphology, spectral, and quality features for all 42 extracted windows.
- `outputs/08_rr_intervals.csv`: Long-form table recording every detected beat-to-beat interval with its physiological plausibility status.
- `outputs/08_beat_templates.npz`: Compressed archive of median beat templates across every record and phase.
- `outputs/08_data_quality_report.csv`: Quality validation diagnostics, recording clipping fractions, NaN counts, and processing statuses.
- `outputs/08_dataset_summary.csv`: Summary of processed seizure events and usable phase counts for records `sz01` through `sz07`.
- `outputs/08_real_seizure_phases.png`: Waveform inspection figure showing raw and cleaned ECG across phases.
- `outputs/08_real_feature_distributions.png`: Boxplot visualizations of HR, SDNN, spectral centroid, and amplitude distributions across clinical classes.
- `outputs/08_empirical_seizure_parameters.csv`: Compact aggregate summary table shown above.

### What I Learned
The real data demonstrates distinct physiological patterns:
1. **Ictal Autonomic Surge:** Ictal phases show a pronounced increase in mean heart rate ($103.82\text{ BPM}$ vs. $69.55\text{ BPM}$ interictal) accompanied by increased signal amplitude and elevated high-frequency power ($0.003185$ vs. $0.002427$), capturing the combined effect of sympathetic tachycardia and somatic muscle tremor.
2. **Preictal Dynamics:** Preictal windows show subtle, variable autonomic shifts (slight HR elevation to $71.63\text{ BPM}$ and increased SDNN to $79.91\text{ ms}$) rather than dramatic changes, highlighting the challenge of early seizure forecasting.
3. **Arrhythmia Discrimination:** The cardiac arrhythmia control (MIT-BIH 203) exhibits extreme rhythm irregularity ($\text{SDNN} = 206.97\text{ ms}$, $\text{RMSSD} = 269.16\text{ ms}$) without the sustained high-frequency spectral signature characteristic of ictal muscle tremor, providing a clear basis for distinguishing cardiac arrhythmias from epileptic seizures.

### Limitations
The PhysioNet SZDB dataset provides ECG from only seven monitored individuals. Automated morphology delineation frequently fails on noisy clinical traces (leaving fields like PR interval and ST level as `NaN`). Most importantly, high-frequency spectral power reflects total band energy from all sources (movement, muscle tremor, electrode drift) and cannot be interpreted as a clean EMG recording.

### Connection to the Next Experiment
Experiment 08 established the real physiological feature targets we need. In Experiment 09, we use these targets to drive an optimization-calibrated synthetic ECG generator.

---

## Experiment 09 — Empirically Calibrated Parametric Synthetic ECG Generation Engine

### Research Question
How can a parametric ordinary differential equation model (**ECGSYN**) be systematically calibrated—via numerical optimization across explicit parameter search spaces—to reproduce empirical heart rate, heart rate variability, and spectral noise targets extracted from real patient recordings in Experiment 08?

### Core Scientific Identity

> [!IMPORTANT]
> **What This Generator Is and What It Is Not:**
> - **IT IS:** An empirically calibrated **PARAMETRIC / DYNAMICAL MODEL-BASED** synthetic ECG generator based on the McSharry et al. (2003) ordinary differential equation framework.
> - **IT IS NOT A VARIATIONAL AUTOENCODER (VAE).**
> - **IT IS NOT A GENERATIVE ADVERSARIAL NETWORK (GAN).**
> - **IT IS NOT A DIFFUSION MODEL.**
> - It does not train neural network weights, does not utilize gradient backpropagation, and does not learn deep latent representations.

### The Parametric Calibration Loop

Rather than using manual heuristics or uncalibrated presets, Experiment 09 runs an automated **calibration-generation-validation loop** for every empirical observation extracted in Experiment 08:

```
Real Patient Recording (SZDB / MITDB)
               ↓
Experiment 08 Empirical Feature Extraction
  (Target Mean HR, Target SDNN, Target Spectral Bands)
               ↓
Parametric Optimization Engine
  ├── 1. Rhythm Calibration: Search HR_STD_SEARCH_BOUNDS to minimize |achieved_SDNN - target_SDNN|
  └── 2. Spectral Calibration: Search NOISE_SEARCH_GRID to minimize multi-band PSD error
               ↓
Selected Simulator Parameters (heart_rate, selected_hr_std, selected_noise)
               ↓
Continuous 60-Second ECGSYN Dynamical Integration (200 Hz, Single Call)
               ↓
Re-Extract Synthetic Features via Aligned Experiment 08 Pipeline
               ↓
Compute Discrepancy (Target vs. Achieved Error Metrics)
               ↓
Export Calibrated Synthetic Signals & Audit Records
```

### Detailed Calibration Mechanics

#### 1. Rhythm & Heart Rate Variability Calibration
The relationship between the simulator's input variability parameter (`heart_rate_std`, in BPM) and the output time-domain interval variability ($\text{SDNN}$, in ms) is non-linear and governed by the internal ODE integration dynamics. The calibration engine resolves this using a two-step optimization process:
1. **Initial Search Center:** Computes an initial estimate using the analytical approximation:
   $$\text{initial\_hr\_std} = \frac{\text{target\_SDNN} \cdot (\text{target\_HR})^2 / 60000.0}{0.94}$$
   This estimate is bounded within a verified numerical stability window:
   $$\text{HR\_STD\_SEARCH\_BOUNDS} = [0.5\text{ BPM}, 14.0\text{ BPM}]$$
   *(The lower bound of $0.5\text{ BPM}$ avoids periodic flatlines; the upper bound of $14.0\text{ BPM}$ prevents negative RR intervals in the ECGSYN ODE integrator, which cause non-convergence in its Runge-Kutta integration loop).*
2. **Local Candidate Evaluation:** Generates candidate signals across a local multiplier grid:
   $$\text{factors} = [0.75, 0.85, 1.00, 1.15, 1.25]$$
   For each candidate, it synthesizes a trial segment, runs Pan-Tompkins peak detection, filters valid RR intervals ($250\text{--}2500\text{ ms}$), calculates the achieved synthetic SDNN, and selects the parameter value that minimizes absolute error:
   $$\text{selected\_hr\_std} = \arg\min_{\theta} |\text{SDNN}_{\text{achieved}}(\theta) - \text{target\_SDNN}|$$

#### 2. Spectral Noise Calibration
To replicate the spectral energy distribution (baseline wander, high-frequency noise proxy) observed in the real recording, the engine performs a bounded grid search over candidate noise values:
$$\text{NOISE\_SEARCH\_GRID} = [0.005, 0.010, 0.025, 0.060, 0.120]$$
For each candidate noise setting, it:
1. Generates a synthetic signal at the previously calibrated `selected_hr_std`.
2. Computes Welch PSD spectral features identical to Experiment 08.
3. Calculates a normalized multi-feature objective across all valid target spectral bands:
   $$\text{Error}_{\text{spectral}} = \frac{1}{K} \sum_{k=1}^{K} \frac{|\text{Feature}_{k, \text{achieved}} - \text{Feature}_{k, \text{target}}|}{\text{Normalizer}_k}$$
   evaluating `Relative_High_Frequency_Power`, `Baseline_Wander_Band_Power`, and `Spectral_Centroid_Hz`.
4. Selects the noise parameter yielding the lowest multi-feature error.

### Directly Calibrated vs. Post-Hoc Evaluated Features

A critical methodological distinction in Experiment 09 is separating features the simulator directly controls from those that can only be evaluated after generation:

- **Directly Calibrated Features:**
  - **Mean Heart Rate:** Controlled directly via the simulator's `heart_rate` parameter.
  - **Heart Rate Variability (SDNN):** Controlled via optimization of `heart_rate_std`.
  - **Spectral Noise Bands:** Controlled via optimization of `noise`.
- **Post-Hoc Evaluated Features (Not Directly Controlled):**
  - **QRS Duration, QTc Interval, R-peak Amplitude, and Beat Templates:** The ECGSYN mathematical formulation uses fixed kinematic attractor equations that do not expose independent parameters for wave durations or intervals. These morphological metrics are **evaluated post-hoc** against real targets (measuring QRS error, QTc error, and template correlation) rather than claimed as direct simulator controls.

### Continuous 60-Second Generation vs. Chunk Concatenation
Unlike Experiment 03—which concatenated sixty 1-second chunks and required manual DC-offset adjustments at boundary seams—Experiment 09 generates each 60-second signal ($12,000\text{ samples}$ at $200\text{ Hz}$) in a **single continuous integration call**. This preserves continuous dynamical trajectories, prevents artificial step discontinuities, and eliminates transient restart artifacts.

### Verified Target vs. Achieved Calibration Results

Re-extracting features from the generated synthetic signals confirms close agreement with real clinical targets across all classes (from `outputs/09_fitted_synthetic_signals.csv`):

| Clinical Class | Target HR | Achieved HR | HR Rel Err | Target SDNN | Achieved SDNN | SDNN Rel Err | Selected `hr_std` | Selected Noise |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Interictal Baseline** | 69.55 BPM | 68.61 BPM | **1.35%** | 64.38 ms | 56.40 ms | 12.40% | 4.364 BPM | 0.0050 |
| **Preictal Prediction** | 71.63 BPM | 70.83 BPM | **1.12%** | 79.91 ms | 82.26 ms | **2.94%** | 6.069 BPM | 0.0050 |
| **Ictal Seizure Phase** | 103.82 BPM | 103.11 BPM | **0.68%** | 88.95 ms | 83.18 ms | **6.49%** | 14.000 BPM | 0.0100 |
| **Postictal Recovery** | 89.76 BPM | 88.58 BPM | **1.31%** | 100.44 ms | 98.42 ms | **2.01%** | 12.083 BPM | 0.0050 |
| **Hard Negative Arrhythmia** | 112.83 BPM | 110.15 BPM | **2.38%** | 206.97 ms | 185.40 ms | 10.42% | 14.000 BPM | 0.0100 |

### Outputs Generated
- `outputs/09_synthetic_ecg_signals.npz`: Compressed archive storing all calibrated, continuous 60-second synthetic ECG arrays.
- `outputs/09_empirical_generation_parameters.csv`: Complete table of per-signal calibration metadata, search bounds, evaluated candidate counts, and selected parameters.
- `outputs/09_synthetic_feature_table.csv`: Re-extracted feature table computed from the synthetic signals using the aligned Experiment 08 extraction pipeline.
- `outputs/09_empirical_vs_synthetic_parameters.csv`: Side-by-side comparison table reporting absolute and relative errors for every physiological metric.
- `outputs/09_generation_quality_report.csv`: Diagnostic audit logging continuity checks, maximum step differences, and calibration statuses.
- `outputs/09_fitted_synthetic_signals.csv`: Class-level summary table recording calibrated generation parameters.
- `outputs/09_empirical_synthetic_fitting.png`: Multi-panel visualization comparing real and synthetic waveforms, beat templates, and parameter distributions.

### What I Learned
Parametric models can accurately match macro-physiological targets like mean heart rate ($\text{relative error} \le 2.4\%$) and heart rate variability ($\text{relative error} \le 12.4\%$) without requiring complex neural network architectures. Optimization over bounded candidate grids provides stable, reproducible parameter selection.

### Limitations
1. **Kinematic Attractor Constraints:** Because ECGSYN models cardiac waves using fixed mathematical equations, it cannot independently adjust QRS duration, ST-segment elevation, or T-wave morphology to match real pathological variations.
2. **Simplified RR Distributions:** While the generator calibrates the second moment of rhythm variability (SDNN), it does not reproduce the full, complex non-Gaussian RR interval distribution observed in real patient recordings.
3. **Generalization Scope:** Calibrations are derived from a small cohort of seven SZDB patients and one MIT-BIH record; they should not be assumed to capture the full clinical diversity of human epilepsy.

### Connection to the Next Experiments
With an operational parametric generation engine, we proceeded to evaluate its outputs through quantitative fidelity validation (Experiment 10) and downstream classifier testing (Experiment 11). Both experiments represent active, exploratory investigations and are documented below.

---

# 4. Experiments 10 and 11 — STILL IN PROGRESS

> [!CAUTION]
> ## 🚧 STILL IN PROGRESS — NOT FINAL VALIDATION RESULTS
> Experiments 10 and 11 represent **preliminary research scaffolding and exploratory prototypes**. They contain identified methodological constraints, limited sample sizes, and unresolved evaluation issues (such as data leakage). **Their outputs must NOT be presented, cited, or interpreted as completed, validated scientific results or clinical proof.**

---

## Experiment 10 — Quantitative Real vs. Synthetic Feature Validation (IN PROGRESS)

### Intended Research Purpose
Experiment 10 ([`Experiments/10_quantitative_real_vs_synthetic_validation.py`](file:///Users/selink/Desktop/ECG/Experiments/10_quantitative_real_vs_synthetic_validation.py)) was conceived to evaluate the fidelity of synthetic ECG by comparing generated waveforms directly against genuine clinical recordings across four metric families:
1. **Rhythm & Interval Distributions:** Measuring the distance between real and synthetic RR interval series using the Wasserstein metric ($W_1$, Earth Mover's Distance) and two-sample Kolmogorov-Smirnov (**KS**) hypothesis tests.
2. **Morphological Waveform Fidelity:** Calculating Pearson correlation coefficients and root mean squared error (**RMSE**) between real and synthetic median beat templates.
3. **Spectral Distribution Fidelity:** Measuring mean squared error (**MSE**) between normalized power spectral densities to compare spectral shapes independent of amplitude scale differences ($\text{mV}$ vs. $\text{a.u.}$).
4. **Spectral Centroid Discrepancy:** Comparing the power-weighted center frequencies of real and synthetic signals.

### Planned Metric Families & Mathematical Formulations

```
Real Clinical ECG (sz01 / sz04 / 203)        Synthetic ECG (Exp 09 Parameterization)
                  \                                /
                   \                              /
                    ↓                            ↓
          [Quantitative Multi-Dimensional Comparison Framework]
          ├── Rhythm Fidelity:      Wasserstein Distance W₁(RR_real, RR_synth)
          │                         Two-Sample Kolmogorov-Smirnov Test
          ├── Morphological Match:  Beat Template Pearson Correlation r(T_real, T_synth)
          │                         Beat Template Root Mean Squared Error (RMSE)
          └── Spectral Fidelity:    Normalized PSD Mean Squared Error
                                    Spectral Centroid Error |SC_real - SC_synth|
```

### Why Results Are Incomplete & Unvalidated
While the script executes and exports summary files (`outputs/10_quantitative_validation_metrics.csv` and `outputs/10_real_vs_synthetic_validation.png`), the analysis has several major methodological limitations:
- **Single-Record References:** The comparisons rely on individual representative segments from single recordings (`sz01` and `203`) rather than population-level distributions drawn from multi-patient cohorts.
- **Uncalibrated Metric Baselines:** Thresholds for what constitutes an acceptable Wasserstein distance or PSD MSE on arbitrary-unit signals have not been clinically validated.
- **Morphology Mismatch:** Because the ECGSYN generator does not independently control wave shapes, template correlations vary widely across classes, and no statistical equivalence test has been established.

---

## Experiment 11 — Downstream AI Seizure Detection Evaluation Scaffold (IN PROGRESS)

### Intended Research Purpose
Experiment 11 ([`Experiments/11_downstream_seizure_classifier.py`](file:///Users/selink/Desktop/ECG/Experiments/11_downstream_seizure_classifier.py)) was designed as an exploratory evaluation scaffold to test whether augmenting training sets with synthetic data improves downstream machine learning classification. The experimental design compares three training cohorts evaluated on held-out real patient windows:
- **Cohort A (Real Only):** Model trained exclusively on genuine clinical ECG windows.
- **Cohort B (Real + Conventional Augmentation):** Model trained on real windows combined with baseline-wander and Gaussian-noise augmentations.
- **Cohort C (Real + Empirically Calibrated Synthetic):** Model trained on real windows combined with synthetic 5-second ECG windows generated via Experiment 09 parameters.

### CRITICAL METHODOLOGICAL ISSUE: Data Leakage via Random Window Splitting

> [!WARNING]
> ### Methodological Flaw: Random Train/Test Splitting on Continuous Recordings
> In the current implementation of Experiment 11, continuous recordings from just three files (`sz01`, `sz04`, and `203`) are sliced into 5-second windows and then partitioned using scikit-learn's standard `train_test_split`:
> ```python
> # Implementation in Experiments/11_downstream_seizure_classifier.py (Lines 149-155)
> tr_feats, te_feats = train_test_split(feats, test_size=0.35, random_state=42, shuffle=True)
> ```
> **This introduces severe data leakage:**
> 1. **Temporal Autocorrelation Leakage:** Slicing a continuous physiological recording into 5-second windows and randomly shuffling them places temporally adjacent windows from the exact same patient session into both the training set and the test set.
> 2. **Subject Identity Leakage:** The random forest classifier learns to recognize the unique, idiosyncratic baseline voltage, noise texture, and morphology of patient `sz01` and `sz04`, rather than learning generalizable electrophysiological markers of epileptic seizures.
> 3. **Artificially Inflated Metrics:** Because of this leakage, the script reports near $100\%$ accuracy across models. **These numbers are methodologically compromised artifacts of data leakage and must NOT be cited as evidence of true diagnostic capability.**

### What Must Be Done to Resolve This
To establish valid downstream evaluation, future work must implement:
1. **Patient-Wise Holdout (`GroupKFold` / Leave-One-Patient-Out):** Slicing must group all windows by distinct patient identity. Models must be trained on Patients $1 \dots N-1$ and tested exclusively on Patient $N$.
2. **Multi-Center Clinical Cohorts:** Expanding beyond the limited PhysioNet SZDB sample to larger ambulatory datasets (e.g., the Siena Scalp EEG/ECG Database or the Temple University Hospital Seizure Corpus).
3. **Strict Prohibition of Claims:** **Under no circumstances should this repository claim that synthetic ECG generation improves seizure classification accuracy or enables reliable preictal seizure prediction.**

---

# 5. Current Research Pipeline

The complete end-to-end research pipeline is summarized below:

```
                  ┌─────────────────────────────────────────────────┐
                  │          Real Clinical Seizure ECG              │
                  │   (PhysioNet SZDB sz01-sz07 & MITDB 203)        │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │                 Experiment 08                   │
                  │       Empirical Feature Extraction Pipeline     │
                  │  (Zero Synthetic Fallbacks | Provable Metadata) │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │     Empirical Physiological Target Vectors      │
                  │  • Mean HR & Valid RR Interval Distributions    │
                  │  • Heart Rate Variability (SDNN, RMSSD)         │
                  │  • DWT Morphology & Aligned Beat Templates      │
                  │  • Welch PSD Spectral Bands & Noise Proxies     │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │                 Experiment 09                   │
                  │    Parametric Calibration & Synthesis Engine    │
                  │   (Bounded Grid Search Minimizing Real Error)   │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │                     ECGSYN                      │
                  │   3D Dynamical Ordinary Differential Equations  │
                  │    (Continuous 60s Single-Call Trajectories)    │
                  └────────────────────────┬────────────────────────┘
                                           │
                                           ▼
                  ┌─────────────────────────────────────────────────┐
                  │           Calibrated Synthetic ECG              │
                  │  (Rhythm & Noise Calibrated | Post-Hoc Morph)   │
                  └────────────────────────┬────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│             Experiment 10             │   │             Experiment 11             │
│   Quantitative Fidelity Validation    │   │     Downstream AI Utility Scaffold    │
│  (Wasserstein, KS, PSD MSE, Template) │   │     (Real vs. Aug vs. Synth Cohorts)  │
│                                       │   │                                       │
│   🚧 STILL IN PROGRESS                │   │   🚧 STILL IN PROGRESS                │
│   (Limited References | Uncalibrated) │   │   (Data Leakage | Subject-Overlap)    │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
```

---

# 6. Relationship to Deep Generative Models

To place this work in the broader context of biomedical machine learning, it is helpful to understand how parametric simulation compares to deep generative models (such as Variational Autoencoders, Generative Adversarial Networks, and Diffusion Models):

```
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SPECTRUM OF ECG SYNTHESIS APPROACHES                                  │
├──────────────────────────────────────────────────┬────────────────────────────────────────────────────┤
│           CURRENT PROJECT APPROACH               │           LEARNED DEEP GENERATIVE MODELS           │
│     (Empirically Calibrated Parametric ODE)      │            (Conditional VAE / GAN / Diffusion)     │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
│ Real Clinical ECG                                │ Real Clinical ECG Exemplars (Thousands)            │
│         ↓                                        │         ↓                                          │
│ Feature Extraction Pipeline (Exp 08)             │ Deep Neural Network Architecture                   │
│   (Mean HR, SDNN, PSD bands)                     │   (Latent Space Mapping / Denoising U-Net)         │
│         ↓                                        │         ↓                                          │
│ Bounded Numerical Optimization                   │ Backpropagation & Loss Optimization                │
│   (Minimizing explicit physiological error)      │   (KL Divergence, Adversarial, or Score Matching)  │
│         ↓                                        │         ↓                                          │
│ ECGSYN Dynamical ODE Model                       │ Sampled Latent Vector $z \sim \mathcal{N}(0, I)$   │
│         ↓                                        │         ↓                                          │
│ Synthetic ECG Waveform                           │ Synthetic ECG Waveform                             │
├──────────────────────────────────────────────────┼────────────────────────────────────────────────────┤
│ Key Characteristics:                             │ Key Characteristics:                               │
│ • Fully interpretable state equations            │ • High-capacity, non-linear representation         │
│ • Biomechanically bounded parameters             │ • Capable of learning arbitrary wave morphology   │
│ • Guaranteed mathematical stability              │ • Susceptible to mode collapse and hallucinations  │
│ • Rigid morphology (cannot warp individual waves)│ • Requires extensive training data                 │
│ • Zero black-box neural network weights          │ • Opaque internal feature representations          │
└──────────────────────────────────────────────────┴────────────────────────────────────────────────────┘
```

### Why the Parametric Approach Matters
1. **Interpretable Baseline:** Parametric generation provides an explicit, biologically grounded baseline against which deep neural networks can be evaluated. If a complex diffusion model cannot outperform a calibrated ordinary differential equation model on downstream tasks, its computational overhead may not be justified.
2. **Explicit Parameter Control:** Clinicians can directly inspect and verify the generation parameters (e.g., commanding a heart rate of $103.82\text{ BPM}$ and an SDNN of $88.95\text{ ms}$) without navigating complex latent spaces.
3. **No Claim of Deep Learning Generation:** **No VAE, GAN, or diffusion architecture has been trained or deployed in this repository.** This project establishes the empirical feature extraction and dynamical modeling foundations that would be necessary to condition such models in future research.

---

# 7. Presentation-Friendly Summary

This section provides concise summaries for explaining this project in meetings, research seminars, or presentations.

### 30-Second Explanation
> "Our research investigates how to model the cardiovascular changes that accompany epileptic seizures. We benchmarked six Python biosignal libraries, audited their signal processing and augmentation limits, and analyzed real seizure ECG recordings from PhysioNet. Rather than relying on guesswork, we built a calibration pipeline that tunes an ordinary differential equation model directly to real physiological targets—matching clinical heart rate, rhythm variability, and spectral noise. This provides an interpretable, parametric baseline for synthetic ECG generation before moving to complex deep generative architectures."

### 1-Minute Explanation
> "The goal of this project is to develop and evaluate methodologies for generating synthetic electrocardiogram signals informed by real epileptic seizures. We began by evaluating six Python toolkits—including NeuroKit2, physioKIT, and torch_ecg—benchmarking their parameter control, filtering performance, and R-peak detection accuracy.
> 
> In our early experiments, we built a proof-of-concept seizure simulation, but we realized that its parameters were hand-tuned heuristics. To fix this, we created an empirical characterization pipeline that processes real seizure recordings from PhysioNet's SZDB database across interictal, preictal, ictal, and postictal phases, alongside cardiac arrhythmia controls.
> 
> We then built an optimization engine that calibrates a 3D dynamical model—ECGSYN—directly to these real patient metrics, achieving less than two percent error on target heart rate and matching real heart rate variability and spectral noise bands. While our quantitative validation and downstream classifier experiments are still in progress due to sample-size limits and data leakage challenges, this work establishes an interpretable, data-driven foundation for biosignal modeling."

### Main Research Question
> *"How can biosignal simulation toolkits be systematically calibrated against real clinical recordings to produce synthetic ECG signals that accurately reflect the autonomic changes of epileptic seizures?"*

### Main Contribution So Far
> *Developing an automated, optimization-based calibration pipeline that extracts empirical rhythm, variability, and spectral targets from real clinical seizure recordings (PhysioNet SZDB) and tunes a continuous dynamical ODE model (ECGSYN) to reproduce those physiological targets with quantified accuracy, replacing manual heuristic simulation.*

### Biggest Current Limitation
> *The downstream evaluation framework (Experiment 11) suffers from data leakage because 5-second windows from the same continuous recordings were randomly shuffled into training and test sets, artificially inflating classifier accuracy. In addition, the underlying ECGSYN model uses rigid kinematic equations that prevent independent control over detailed wave morphology.*

### Logical Next Step
> *Resolving the evaluation framework by implementing strict patient-independent holdout validation (`GroupKFold` / Leave-One-Patient-Out) across an expanded multi-patient clinical dataset, followed by evaluating whether conditional deep generative models (such as conditional diffusion models) can improve morphological fidelity over this parametric baseline.*

---

# 8. Repository Structure & Execution

### Directory Layout

```
ECG/
├── Experiments/
│   ├── 01_generation_parameter_sweeps.py       # Systematic parameter sweeps (NeuroKit2 vs. physioKIT)
│   ├── 02_physiokit_arrhythmia_presets.py      # physioKIT preset audit & clinical classification
│   ├── 03_dynamic_ictal_surge_simulation.py    # Rule-based proof-of-concept seizure simulation
│   ├── 04_torch_ecg_augmentations.py          # Native torch_ecg tensor augmentations & risk audit
│   ├── 05_signal_acquisition_comparison.py     # Signal acquisition audit (synthetic a.u. vs. real mV)
│   ├── 06_noise_filtering_snr_benchmark.py     # Quantitative filtering benchmark (+20 dB to -5 dB SNR)
│   ├── 07_qrs_detection_multi_record.py       # R-peak detection benchmark (10 MIT-BIH records)
│   ├── 08_real_seizure_ecg_extraction.py      # Real SZDB seizure & MITDB arrhythmia characterization
│   ├── 09_empirical_synthetic_fitting.py       # Empirically calibrated parametric ECGSYN generator
│   ├── 10_quantitative_real_vs_synthetic_validation.py # [IN PROGRESS] Fidelity evaluation
│   └── 11_downstream_seizure_classifier.py    # [IN PROGRESS] Downstream classifier & leakage audit
├── outputs/                                    # Generated figures, summary CSVs, and NPZ archives
├── requirements.txt                            # Environment dependencies
├── run_all_experiments.py                     # Sequential test runner script
└── README.md                                   # Comprehensive documentation
```

### Installation

Clone the repository and install dependencies within a virtual environment:

```bash
git clone https://github.com/selingulk/ECG.git
cd ECG
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Execution

Execute the core methodological experiments (01 through 09) sequentially:

```bash
python Experiments/01_generation_parameter_sweeps.py
python Experiments/02_physiokit_arrhythmia_presets.py
python Experiments/03_dynamic_ictal_surge_simulation.py
python Experiments/04_torch_ecg_augmentations.py
python Experiments/05_signal_acquisition_comparison.py
python Experiments/06_noise_filtering_snr_benchmark.py
python Experiments/07_qrs_detection_multi_record.py
python Experiments/08_real_seizure_ecg_extraction.py
python Experiments/09_empirical_synthetic_fitting.py
```

To run the exploratory in-progress research scripts (with full awareness of their preliminary status and documented methodological limitations):

```bash
python Experiments/10_quantitative_real_vs_synthetic_validation.py
python Experiments/11_downstream_seizure_classifier.py
```

---

## References

1. **Goldberger, A. L., et al.** (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*, 101(23), e215–e220.
2. **McSharry, P. E., Clifford, G. D., Tarassenko, L., & Smith, L. A.** (2003). A dynamical model for generating synthetic electrocardiogram signals. *IEEE Transactions on Biomedical Engineering*, 50(3), 289–294.
3. **Makowski, D., et al.** (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689–1696.
4. **Pan, J., & Tompkins, W. J.** (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, BME-32(3), 230–236.
5. **Carreiras, C., et al.** (2015). BioSPPy: Biosignal Processing in Python. *Online documentation and software release*.
6. **Wen, H., et al.** (2023). torch_ecg: An open-source Python library for deep learning on electrocardiograms. *Software documentation and release*.
