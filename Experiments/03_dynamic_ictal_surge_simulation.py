"""
Experiment 03: Dynamic Time-Varying Proof-of-Concept Simulation of Epileptic Seizure Onset
==========================================================================================
Research Question: How can classical ECG simulation toolkits (NeuroKit2) be extended
to construct a synthetic proof-of-concept multi-stage epileptic seizure episode?

CRITICAL METHODOLOGICAL TRANSPARENCY & CLINICAL LIMITATIONS:
  - Nature of Model: Rule-based, phenomenological synthetic proof-of-concept approximation.
  - Future Work Requirement: In future research, heart-rate, HRV, morphology, and noise
    parameters should be estimated directly from real interictal, preictal, ictal, and postictal
    ECG patient recordings rather than rule-based synthetic specifications.
  - Physiological Limitation 1: Purely piecewise frequency ramp (70 -> 160 BPM) without real autonomic
    neuro-cardiac coupling (e.g., sympathetic surge dynamics or vagal withdrawal kinetics).
  - Physiological Limitation 2: Additive filtered bandpass Gaussian noise model (20-200 Hz) representing
    myoclonic muscle artifacts, lacking biomechanical motor contraction dynamics.
  - Clinical Validation Status: NOT validated against clinical long-term ambulatory video-EEG/ECG epilepsy datasets.
    Serves exclusively as an exploratory synthetic proof-of-concept data generator.

Seizure Stages Simulated (60-second continuous recording at 500 Hz):
  1. Inter-ictal Baseline (0–15 s): Normal sinus rhythm, HR = 70 BPM.
  2. Pre-ictal Transition & Ictal Surge (15–30 s): Dynamic HR acceleration (70 -> 160 BPM).
  3. Motor Seizure Tremor (25–40 s): Overlay of high-frequency EMG muscle noise (20–200 Hz).
  4. Post-ictal Recovery (40–60 s): Gradual HR deceleration (160 -> 75 BPM).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import scipy.signal as sp_signal

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 03: Synthetic Proof-of-Concept Seizure Surge Simulation ===")

FS = 500
DURATION = 60  # seconds
N_SAMPLES = FS * DURATION
TIME_AXIS = np.arange(N_SAMPLES) / float(FS)

# Construct time-varying target Heart Rate curve (BPM)
# 0-15s: 70 BPM
# 15-30s: Ramp 70 -> 160 BPM
# 30-45s: 160 BPM
# 45-60s: Ramp 160 -> 75 BPM
hr_curve = np.zeros(N_SAMPLES)
for i, t in enumerate(TIME_AXIS):
    if t < 15:
        hr_curve[i] = 70.0
    elif 15 <= t < 30:
        hr_curve[i] = 70.0 + (160.0 - 70.0) * ((t - 15) / 15.0)
    elif 30 <= t < 45:
        hr_curve[i] = 160.0
    else:
        hr_curve[i] = 160.0 - (160.0 - 75.0) * ((t - 45) / 15.0)

# Generate continuous signal using 1-second chunks with boundary DC offset alignment
ecg_chunks = []
np.random.seed(42)
for chunk_idx in range(DURATION):
    t_start = chunk_idx
    target_hr = float(hr_curve[chunk_idx * FS])
    # Lower HRV during ictal surge phase
    std_val = 15.0 if t_start < 15 else (3.0 if 15 <= t_start < 45 else 8.0)
    chunk_ecg = nk.ecg_simulate(duration=1, sampling_rate=FS, heart_rate=target_hr, heart_rate_std=std_val, noise=0.01, random_state=42 + chunk_idx)
    ecg_chunks.append(chunk_ecg)

# Inspect & align 1-second segment boundaries to eliminate vertical step discontinuities
aligned_chunks = []
for i, chunk in enumerate(ecg_chunks):
    if i == 0:
        aligned_chunks.append(chunk)
    else:
        offset = aligned_chunks[-1][-1] - chunk[0]
        aligned_chunks.append(chunk + offset)

clean_seizure_ecg = np.concatenate(aligned_chunks)

# Generate simulated EMG muscle tremor noise (bandpass filtered 20-200 Hz noise)
emg_raw = np.random.normal(0, 0.25, N_SAMPLES)
b_emg, a_emg = sp_signal.butter(3, [20.0 / (FS/2), 200.0 / (FS/2)], btype='bandpass')
emg_filtered = sp_signal.filtfilt(b_emg, a_emg, emg_raw)

# Apply smooth time envelope to EMG tremor (active during tonic-clonic phase 25s-40s)
emg_envelope = np.zeros(N_SAMPLES)
for i, t in enumerate(TIME_AXIS):
    if 25 <= t <= 40:
        emg_envelope[i] = np.sin((t - 25) / 15.0 * np.pi) ** 2

noisy_seizure_ecg = clean_seizure_ecg + emg_filtered * emg_envelope * 1.5

# Plot continuous recording with disclaimers & arbitrary units (a.u.)
plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(TIME_AXIS, hr_curve, color='red', linewidth=2, label="Dynamic Target HR (BPM)")
plt.axvspan(0, 15, color='blue', alpha=0.1, label="1. Inter-ictal (Baseline)")
plt.axvspan(15, 30, color='orange', alpha=0.15, label="2. Ictal Surge (Tachycardia)")
plt.axvspan(25, 40, color='purple', alpha=0.15, label="3. Motor Seizure Tremor")
plt.axvspan(45, 60, color='green', alpha=0.1, label="4. Post-ictal Recovery")
plt.title("Synthetic Proof-of-Concept Seizure Simulation: Autonomic Heart Rate Profile\n"
          "(Rule-Based Phenomenological Proof of Concept — Not Clinically Validated)", fontsize=11, fontweight='bold')
plt.ylabel("Heart Rate (BPM)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)

plt.subplot(2, 1, 2)
plt.plot(TIME_AXIS, noisy_seizure_ecg, color='darkblue', linewidth=0.7, label="Simulated Seizure ECG (with EMG Tremor)")
plt.plot(TIME_AXIS, clean_seizure_ecg, color='cyan', alpha=0.5, linewidth=0.5, label="Underlying Rhythm (Boundary Aligned)")
plt.axvspan(25, 40, color='purple', alpha=0.15)
plt.title("Multi-Stage Synthetic Seizure ECG Waveform (Boundary-Discontinuity Aligned)", fontsize=10, fontweight='bold')
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (a.u.)")
plt.legend(loc="upper right", fontsize=8)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
out_img = "outputs/03_dynamic_ictal_surge_simulation.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 03 complete. Saved visualization to {out_img}\n")
plt.close()
