# Evaluation of Python Toolkits for ECG Simulation, Processing and Augmentation

A comprehensive benchmark evaluating six Python toolkits (**NeuroKit2, physioKIT, torch_ecg, BioSPPy, WFDB, SciPy**) for ECG generation, parameter control, arrhythmia synthesis, signal augmentation, and preprocessing.

---

## Executive Summary & Toolkit Scoring Rubric

To evaluate each toolkit transparently, suitabilities are scored against a defined 100-point rubric:
- **ECG Generation & Simulation (25%):** Native ability to generate synthetic multi-beat ECG signals.
- **Parameter Control & Sweeps (20%):** Fine-grained control over HR, HRV, noise, sampling rate, and morphology.
- **Arrhythmia & Pathological Presets (20%):** Native support for generating pathological rhythms (AFIB, STEMI, LBBB).
- **Signal Augmentation Capabilities (15%):** Dedicated pipeline transformations (scaling, cropping, noise, lead masking).
- **Multi-lead & Scalability Support (10%):** Ability to process/generate multi-channel ECG data.
- **Usability & Documentation (10%):** API cleanliness, standard output structures, and documentation quality.

### Composite Scorecard

| Toolkit | Generation (25) | Parameter Control (20) | Arrhythmia Presets (20) | Augmentation (15) | Multi-lead (10) | Usability (10) | Total Score / 100 | Primary Project Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NeuroKit2** | 24 | 19 | 10 | 12 | 7 | 9 | **81 / 100** | **Primary Parametric ECG & HRV Generator** |
| **torch_ecg** | 5 | 8 | 5 | 15 | 9 | 8 | **50 / 100** | **Deep Learning ECG Augmentation Library** |
| **physioKIT** | 20 | 11 | 18 | 6 | 9 | 8 | **72 / 100** | **Pathological Arrhythmia Preset Generator** |
| **BioSPPy** | 0 | 0 | 0 | 4 | 5 | 8 | **17 / 100** | **Signal Preprocessing & Template Extraction** |
| **WFDB** | 0 | 0 | 0 | 0 | 9 | 9 | **18 / 100** | **PhysioNet Dataset I/O & Ground Truth Reader** |
| **SciPy** | 0 | 5 | 0 | 5 | 5 | 7 | **22 / 100** | **Low-level DSP Primitive Engine** |

---

## Key Findings: ECG Generation & Augmentation

1. **Generation Capabilities:**
   - **NeuroKit2** provides the most flexible continuous parametric simulation (`nk.ecg_simulate`), allowing smooth control over mean heart rate, HRV (`heart_rate_std`), noise, and sampling rate.
   - **physioKIT** provides the richest suite of rule-based pathological presets (8 presets including AFIB, STEMI, LBBB).
   - **BioSPPy, WFDB, and SciPy** contain zero signal synthesis APIs.
2. **Arrhythmic Generation vs. Real Recordings:**
   - **physioKIT** generates rule-based mathematical waveforms for AFIB, STEMI, LBBB, and LAHB. These are suitable for ML pretraining but are not clinically validated patient recordings.
   - **NeuroKit2** generates sinus rhythm with respiratory sinus arrhythmia (HRV noise); it does not natively synthesize named pathological arrhythmias.
   - **WFDB** reads real clinical arrhythmia recordings (e.g., MIT-BIH Record 106 containing PVCs), which represent actual physiological pathology.
3. **Signal Augmentation & Clinical Risks (`torch_ecg` Native Augmenters):**
   - **`RandomRenormalize` (Amplitude Scaling):** Rescaling signal amplitude poses a **HIGH CLINICAL RISK** of altering voltage criteria for Left Ventricular Hypertrophy (Sokolow-Lyon index).
   - **`CutMix` / `StretchCompressOffline` (Time Stretching / Segment Mixing):** Distorting temporal duration or slicing segments poses a **HIGH CLINICAL RISK** of artificially modifying QRS duration and QT intervals, mimicking bundle branch blocks or Long QT Syndrome.
   - **`RandomFlip` (Signal Inversion):** Reversing polarity mimics dextrocardia or inverted electrode placement.
   - **`RandomMasking` (Lead Masking / Dropout):** Zeroing out channels tests AI model robustness against electrode failure.

---

## Project Structure

```
MAU/
├── Experiments/
│   ├── 01_generation_parameter_sweeps.py     # NeuroKit2 & physioKIT systematic parameter sweeps
│   ├── 02_physiokit_arrhythmia_presets.py    # Benchmark of all 8 physioKIT pathological presets
│   ├── 03_dynamic_ictal_surge_simulation.py  # Proof-of-concept seizure simulation with explicit physiological disclaimers
│   ├── 04_torch_ecg_augmentations.py        # Native torch_ecg.augmenters class transformations & risk audit
│   ├── 05_signal_acquisition_comparison.py   # Signal acquisition audit (strict 10s window)
│   ├── 06_noise_filtering_snr_benchmark.py   # Quantitative SNR filtering (+20 dB to -5 dB)
│   └── 07_qrs_detection_multi_record.py     # R-peak detection benchmark (10 MIT-BIH records, first 100s)
├── outputs/
│   ├── 01_generation_parameter_sweeps.png
│   ├── 02_physiokit_arrhythmia_presets.png
│   ├── 02_physiokit_presets_summary.csv
│   ├── 03_dynamic_ictal_surge_simulation.png
│   ├── 04_torch_ecg_augmentations.png
│   ├── 04_augmentation_clinical_risk_audit.csv
│   ├── 05_signal_acquisition_comparison.png
│   ├── 06_noise_filtering_snr_benchmark.png
│   ├── 06_snr_filtering_summary.csv
│   ├── 07_qrs_detection_multi_record.png
│   ├── 07_qrs_detection_summary.csv
│   └── 07_qrs_detection_detailed_results.csv
├── requirements.txt
└── README.md
```

---

## Installation & Execution

```bash
pip install -r requirements.txt
```

Run all 7 experiments sequentially from the repository root:

```bash
python Experiments/01_generation_parameter_sweeps.py
python Experiments/02_physiokit_arrhythmia_presets.py
python Experiments/03_dynamic_ictal_surge_simulation.py
python Experiments/04_torch_ecg_augmentations.py
python Experiments/05_signal_acquisition_comparison.py
python Experiments/06_noise_filtering_snr_benchmark.py
python Experiments/07_qrs_detection_multi_record.py
```

---

## Detailed Experimental Benchmark Results

### Experiment 01: Generation Parameter Sweeps (NeuroKit2 & physioKIT)

| Parameter Swept | Tested Values | Observed Visual & Morphological Effect |
| :--- | :--- | :--- |
| **Heart Rate (HR)** | 50, 70, 100, 150 BPM | Alters inter-beat R-R spacing without changing individual P-QRS-T wave durations. |
| **HRV (`heart_rate_std`)** | 0, 5, 15, 30 | Modulates beat-to-beat interval variability (simulating respiratory sinus arrhythmia). |
| **Noise (`noise`)** | 0.0, 0.01, 0.05, 0.10 | High noise obscures low-amplitude P and T waves, leaving only QRS peaks visible. |
| **Sampling Rate ($f_s$)** | 100, 250, 500, 1000 Hz | Low sampling rate (100 Hz) causes time quantization and R-peak amplitude attenuation. |
| **Simulation Method** | `"neurokit"` vs `"ecgsyn"` | `"neurokit"` uses wavelet synthesis; `"ecgsyn"` uses McSharry's dynamic 3D ODE model. |

*Visual Output:* `outputs/01_generation_parameter_sweeps.png`

---

### Experiment 02: physioKIT Pathological Arrhythmia Presets

| Preset Name | Clinical Category | Key Morphological Signature | Arrhythmia Type | AI Pretraining Suitability |
| :--- | :--- | :--- | :---: | :---: |
| **SR** | Normal Sinus Rhythm | Normal P-QRS-T complex, regular R-R intervals | No | High |
| **AFIB** | Atrial Fibrillation | Irregular R-R intervals, absent P-waves | Yes | High |
| **ant_STEMI** | Anterior ST-Elevation MI | Pronounced ST-segment elevation above baseline | Yes | High |
| **LAHB** | Left Anterior Hemiblock | Left axis deviation alterations | Yes | Moderate |
| **LPHB** | Left Posterior Hemiblock | Right axis deviation alterations | Yes | Moderate |
| **high_take_off**| Early Repolarization | Elevated J-point with concave ST elevation | Yes | Moderate |
| **LBBB** | Left Bundle Branch Block | Widened QRS (>120 ms), notched R-wave | Yes | High |
| **random_morphology**| Stochastic Synthesis | Randomized wave amplitudes and durations | Variable | Moderate |

*Data & Visualisation:* `outputs/02_physiokit_presets_summary.csv`, `outputs/02_physiokit_arrhythmia_presets.png`

---

### Experiment 03: Dynamic Epileptic Seizure Episode Simulation (Proof-of-Concept)

Demonstrates a continuous 60-second multi-stage simulation of an epileptic seizure episode:
1. **Inter-ictal Baseline (0--15s):** 70 BPM, normal HRV.
2. **Ictal Acceleration Surge (15--30s):** Dynamic HR surge from 70 to 160 BPM.
3. **Tonic-Clonic Motor Seizure (25--40s):** Injection of high-frequency EMG muscle tremor noise ($20\text{--}200\text{ Hz}$).
4. **Post-ictal Recovery (40--60s):** Deceleration back to 75 BPM.

**Methodological Disclaimers & Clinical Limitations:**
- **Model Nature:** Rule-based phenomenological synthetic proof-of-concept approximation for AI feature pre-training.
- **Autonomic Coupling:** Uses a piecewise target HR ramp without autonomic neuro-cardiac feedback dynamics (sympathetic surge / vagal withdrawal kinetics).
- **EMG Noise Model:** Filtered bandpass Gaussian noise ($20\text{--}200\text{ Hz}$), lacking biomechanical motor contraction dynamics.
- **Clinical Validation:** Has not been validated against ambulatory long-term video-EEG/ECG clinical epilepsy cohorts.

*Visual Output:* `outputs/03_dynamic_ictal_surge_simulation.png`

---

### Experiment 04: Native `torch_ecg` Augmentation Audit & Clinical Risk Assessment

Evaluated directly via native `torch_ecg.augmenters` classes acting on PyTorch Tensors:

| Augmentation Method | NATIVE `torch_ecg` Class | Visual Transformation | Clinical Diagnostic Risk | Preserves Labels? |
| :--- | :--- | :--- | :--- | :---: |
| **Baseline Wander** | `torch_ecg.augmenters.BaselineWanderAugmenter` | Low-frequency sinusoidal + Gaussian noise | Obscures P/T waves; may cause false peak detections | Yes |
| **Amplitude Scaling** | `torch_ecg.augmenters.RandomRenormalize` | Rescaling signal amplitude ($1.6\times$) | **HIGH RISK:** Alters LVH voltage criteria (Sokolow-Lyon) | **No** |
| **Polarity Inversion** | `torch_ecg.augmenters.RandomFlip` | Polarity flip ($+ \rightarrow -$) | **HIGH RISK:** Inverts P/T waves; mimics lead reversal | **No** |
| **Lead Masking** | `torch_ecg.augmenters.RandomMasking` | Zeroing signal segments | Moderate Risk: Simulates electrode disconnection | Yes |
| **Segment Mixing** | `torch_ecg.augmenters.CutMix` | Slicing and mixing signal segments | **HIGH RISK:** Distorts beat continuity and rhythm | **No** |

*Data & Visualisation:* `outputs/04_augmentation_clinical_risk_audit.csv`, `outputs/04_torch_ecg_augmentations.png`

---

### Experiment 05: Signal Acquisition & Data Representation Audit

Evaluated over a strict 10.0-second window across all toolkits:

| Toolkit | Acquisition Method | Source Type | Output Data Structure | Physical Units |
| :--- | :--- | :--- | :--- | :---: |
| **NeuroKit2** | `nk.ecg_simulate()` | Synthetic dynamical model | `np.ndarray` (1D) | Model arbitrary units ($\text{a.u.}$) |
| **physioKIT** | `pk.ecg.synthesize()` | Synthetic preset generator | `tuple(signal, segs, fids)` | Digital arbitrary units ($\text{a.u.}$) |
| **BioSPPy** | None (Processing only) | — | — | — |
| **WFDB** | `wfdb.rdrecord()` | Real clinical recording (PhysioNet) | `wfdb.Record` | Calibrated $\text{mV}$ |
| **SciPy** | None (General DSP) | — | — | — |

*Visual Output:* `outputs/05_signal_acquisition_comparison.png`

---

### Experiment 06: Quantitative Filtering Benchmark across Target SNRs

Evaluated across standardized target input SNRs ($+20.00\text{ dB}$, $+10.00\text{ dB}$, $0.00\text{ dB}$, $-5.00\text{ dB}$) with zero-mean unit-variance amplitude normalization applied before calculating $\Delta\text{SNR}_{\text{dB}}$. Output metrics strictly match `outputs/06_snr_filtering_summary.csv`:

$$\Delta\text{SNR}_{\text{dB}} = \text{SNR}_{\text{out}} - \text{SNR}_{\text{in}}$$

| Toolkit | Filtering Method | $+20.00\text{ dB}$ Target ($\text{SNR}_{\text{in}} = 5.10\text{ dB}$) | $+10.00\text{ dB}$ Target ($\text{SNR}_{\text{in}} = 4.36\text{ dB}$) | $0.00\text{ dB}$ Target ($\text{SNR}_{\text{in}} = 1.35\text{ dB}$) | $-5.00\text{ dB}$ Target ($\text{SNR}_{\text{in}} = -0.38\text{ dB}$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **BioSPPy** | FIR Bandpass (built-in) | **+1.99 dB** | **+2.48 dB** | **+4.15 dB** | **+4.23 dB** |
| **physioKIT** | IIR Butterworth 0.5–35 Hz (zero-phase) | +0.54 dB | +1.07 dB | +3.12 dB | +3.65 dB |
| **SciPy** | Butterworth Bandpass + IIR Notch (zero-phase) | +0.54 dB | +1.07 dB | +3.13 dB | +3.66 dB |
| **NeuroKit2** | `nk.ecg_clean()` (Elgendi bandpass) | +0.30 dB | +0.70 dB | +2.56 dB | +3.13 dB |
| **WFDB** | None (Dataset I/O only) | — | — | — | — |

*Data & Visualisation:* `outputs/06_snr_filtering_summary.csv`, `outputs/06_noise_filtering_snr_benchmark.png`

---

### Experiment 07: R-Peak Detection Performance ($N=10$ MIT-BIH Records, First 100s)

Evaluated under a common EC57-style protocol using a $\pm 150\text{ ms}$ matching window against cardiologist `.atr` annotations across the first 100 seconds of records `100`, `101`, `102`, `103`, `105`, `106`, `119`, `200`, `201`, `203`.

| Toolkit | R-Peak Algorithm | Sensitivity (%) | PPV (%) | $F_1$-Score (%) | P-QRS-T Delineation |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WFDB** | GQRS (`processing.gqrs_detect`) | **99.78 ± 0.35** | 99.69 ± 0.39 | **99.73 ± 0.22** | No |
| **BioSPPy** | Hamilton-Tompkins / Segmenter | 98.93 ± 1.60 | **99.94 ± 0.18** | 99.42 ± 0.89 | QRS Templates |
| **NeuroKit2** | Pan-Tompkins / NeuroKit | 98.73 ± 3.26 | 99.76 ± 0.37 | 99.21 ± 1.70 | **Full (P, Q, R, S, T)** |
| **physioKIT** | Adaptive Peak Locator | 98.96 ± 3.12 | 99.36 ± 0.61 | 99.13 ± 1.66 | Unreliable (0% P/T on test) |
| **SciPy** | Pan-Tompkins DSP Cascade | 92.64 ± 10.01 | 100.00 ± 0.00 | 95.87 ± 5.86 | No |

*Data & Visualisation:* `outputs/07_qrs_detection_summary.csv`, `outputs/07_qrs_detection_detailed_results.csv`, `outputs/07_qrs_detection_multi_record.png`

---

## References

- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
- Makowski, D., et al. (2021). NeuroKit2: A Python toolbox for neurophysiological signal processing. *Behavior Research Methods*, 53(4), 1689–1696.
- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on Biomedical Engineering*, 32(3), 230–236.
- Carreiras, C., et al. (2015). BioSPPy: Biosignal processing in Python. *Documentation and Source Code*.
- Wen, H., et al. (2023). torch_ecg: A Python toolkit for ECG processing and deep learning. *Software and Documentation*.
