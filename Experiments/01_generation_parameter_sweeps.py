"""
Experiment 01: NeuroKit2 & physioKIT Systematic Generation Parameter Sweeps
=============================================================================
Research Question: How do individual generation parameters (Heart Rate, HRV,
Noise, Sampling Rate, Duration, Simulation Method) systematically alter the
resulting synthetic ECG waveforms?

Parameters Swept:
  1. Heart Rate (HR): 50, 70, 100, 150 BPM
  2. HRV (heart_rate_std): 0, 5, 15, 30
  3. Noise (noise): 0.0, 0.01, 0.05, 0.10
  4. Sampling Rate (fs): 100, 250, 500, 1000 Hz
  5. Duration: 5, 10, 60 seconds
  6. Simulation Method: "neurokit" (wavelet) vs "ecgsyn" (McSharry ODE model)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import physiokit as pk

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 01: Generation Parameter Sweeps ===")

fig, axes = plt.subplots(6, 1, figsize=(12, 16))
fig.suptitle("Experiment 01: Systematic Generation Parameter Sweeps (NeuroKit2 & physioKIT)\n"
             "Visualising visual & morphological effects of parameter changes",
             fontsize=12, fontweight='bold', y=0.995)

# 1. Heart Rate Sweep (50, 70, 100, 150 BPM)
time_10s = np.linspace(0, 10, 5000)
hrs = [50, 70, 100, 150]
colors_hr = ["blue", "green", "orange", "red"]
for hr, col in zip(hrs, colors_hr):
    sig = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=hr, noise=0, random_state=42)
    axes[0].plot(time_10s, sig, label=f"{hr} BPM", color=col, alpha=0.7, linewidth=1.0)
axes[0].set_title("1. Heart Rate Sweep (50, 70, 100, 150 BPM) — Effect: Alters inter-beat R-R spacing without changing single beat duration", fontsize=9)
axes[0].set_ylabel("Amplitude")
axes[0].legend(loc="upper right", fontsize=8, ncol=4)
axes[0].grid(True, linestyle="--", alpha=0.4)

# 2. HRV Sweep (heart_rate_std: 0, 5, 15, 30)
std_vals = [0, 5, 15, 30]
for std, col in zip(std_vals, colors_hr):
    sig = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70, heart_rate_std=std, noise=0, random_state=42)
    axes[1].plot(time_10s, sig, label=f"std={std}", color=col, alpha=0.7, linewidth=1.0)
axes[1].set_title("2. HRV Sweep (heart_rate_std = 0, 5, 15, 30) — Effect: Introduces respiratory sinus arrhythmia (beat-to-beat variability)", fontsize=9)
axes[1].set_ylabel("Amplitude")
axes[1].legend(loc="upper right", fontsize=8, ncol=4)
axes[1].grid(True, linestyle="--", alpha=0.4)

# 3. Noise Sweep (noise: 0, 0.01, 0.05, 0.10)
noise_vals = [0.0, 0.01, 0.05, 0.10]
for n_val, col in zip(noise_vals, colors_hr):
    sig = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70, noise=n_val, random_state=42)
    axes[2].plot(time_10s, sig, label=f"noise={n_val}", color=col, alpha=0.7, linewidth=1.0)
axes[2].set_title("3. Noise Sweep (noise = 0, 0.01, 0.05, 0.10) — Effect: High noise obscuring low-amplitude P and T waves", fontsize=9)
axes[2].set_ylabel("Amplitude")
axes[2].legend(loc="upper right", fontsize=8, ncol=4)
axes[2].grid(True, linestyle="--", alpha=0.4)

# 4. Sampling Rate Sweep (100, 250, 500, 1000 Hz)
fs_vals = [100, 250, 500, 1000]
for fs, col in zip(fs_vals, colors_hr):
    sig = nk.ecg_simulate(duration=3, sampling_rate=fs, heart_rate=70, noise=0, random_state=42)
    t = np.arange(len(sig)) / fs
    axes[3].plot(t, sig, label=f"{fs} Hz", color=col, alpha=0.8, linewidth=1.0, marker='o' if fs==100 else None, markersize=3)
axes[3].set_title("4. Sampling Rate Sweep (100, 250, 500, 1000 Hz, 3s window) — Effect: 100 Hz causes time quantization & QRS peak attenuation", fontsize=9)
axes[3].set_ylabel("Amplitude")
axes[3].legend(loc="upper right", fontsize=8, ncol=4)
axes[3].grid(True, linestyle="--", alpha=0.4)

# 5. Duration Sweep (5, 10, 60 seconds)
sig_60s = nk.ecg_simulate(duration=60, sampling_rate=500, heart_rate=70, noise=0, random_state=42)
t_60s = np.arange(len(sig_60s)) / 500
axes[4].plot(t_60s, sig_60s, color="navy", linewidth=0.6)
axes[4].axvspan(0, 5, color="yellow", alpha=0.3, label="5s Window")
axes[4].axvspan(0, 10, color="orange", alpha=0.2, label="10s Window")
axes[4].set_title("5. Duration Sweep (60s continuous generation with 5s and 10s window highlighted)", fontsize=9)
axes[4].set_ylabel("Amplitude")
axes[4].legend(loc="upper right", fontsize=8)
axes[4].grid(True, linestyle="--", alpha=0.4)

# 6. Simulation Method Comparison ("neurokit" vs "ecgsyn")
sig_nk_method = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70, method="neurokit", random_state=42)
sig_syn_method = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=70, method="ecgsyn")
axes[5].plot(time_10s, sig_nk_method, color="steelblue", label="method='neurokit' (Wavelet-based)", linewidth=1.0)
axes[5].plot(time_10s, sig_syn_method, color="darkred", label="method='ecgsyn' (McSharry ODE model)", linewidth=1.0, alpha=0.8)
axes[5].set_title("6. Simulation Method Comparison — NeuroKit wavelet vs ECGSYN dynamic ODE model", fontsize=9)
axes[5].set_ylabel("Amplitude")
axes[5].set_xlabel("Time (seconds)")
axes[5].legend(loc="upper right", fontsize=8)
axes[5].grid(True, linestyle="--", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.99])
out_img = "outputs/01_generation_parameter_sweeps.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 01 complete. Saved visualization to {out_img}\n")
plt.close()
