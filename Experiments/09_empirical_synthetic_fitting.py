"""
Experiment 09: Empirical Real-Fitted Synthetic ECG Generation Pipeline
========================================================================
Objective: Drive NeuroKit2 and physioKIT ECG generation using empirical parameters
(Mean HR, SDNN, EMG noise) extracted from real epileptic seizure stages in Experiment 08.

Fixes & Disclaimers:
  - Boundary Discontinuity Alignment: Aligns vertical DC offsets across independently
    generated 1-second synthetic chunks to prevent step artifacts.
  - Arbitrary Units: All synthetic waveform amplitudes are reported strictly in arbitrary units (a.u.).
  - Proof of Concept: Synthetic signals serve as proof-of-concept models; pretraining suitability remains unevaluated.

Outputs:
  - Figure: outputs/09_empirical_synthetic_fitting.png
  - Summary CSV: outputs/09_fitted_synthetic_signals.csv
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import physiokit as pk

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 09: Empirical Real-Fitted Synthetic ECG Generation ===")

params_csv = "outputs/08_empirical_seizure_parameters.csv"
if not os.path.exists(params_csv):
    raise FileNotFoundError("Please run Experiment 08 first to extract empirical parameters.")

df_params = pd.read_csv(params_csv)

FS = 200 # Matched to PhysioNet szdb (200 Hz)
DURATION_PHASE = 60 # seconds per stage
N_SAMPLES_PHASE = FS * DURATION_PHASE

fitted_signals = {}
records_09 = []

for idx, row in df_params.iterrows():
    stage_name = row["Seizure_Stage"]
    mean_hr = float(row["Mean_HR_BPM"])
    sdnn_ms = float(row["SDNN_ms"])
    emg_power = float(row["EMG_Band_Power"])
    
    # Map SDNN (ms) to NeuroKit2 heart_rate_std parameter
    # Bound heart_rate_std to avoid ODE integration bottleneck
    hr_std_param = min(15.0, max(0.0, sdnn_ms / 10.0))
    noise_param = min(0.15, max(0.005, emg_power * 50.0))
    
    # Generate continuous signal using 1-second chunks with boundary DC offset alignment
    ecg_chunks = []
    for chunk_idx in range(DURATION_PHASE):
        chunk_sig = nk.ecg_simulate(
            duration=1,
            sampling_rate=FS,
            heart_rate=mean_hr,
            heart_rate_std=hr_std_param,
            noise=noise_param,
            random_state=42 + chunk_idx + idx * 100
        )
        ecg_chunks.append(chunk_sig)
        
    # Align boundary DC offsets to eliminate step discontinuities
    aligned_chunks = []
    for c_idx, chunk in enumerate(ecg_chunks):
        if c_idx == 0:
            aligned_chunks.append(chunk)
        else:
            offset = aligned_chunks[-1][-1] - chunk[0]
            aligned_chunks.append(chunk + offset)
            
    fitted_sig = np.concatenate(aligned_chunks)
    fitted_signals[stage_name] = fitted_sig
    
    records_09.append({
        "Seizure_Stage": stage_name,
        "Target_Mean_HR_BPM": mean_hr,
        "Empirical_SDNN_ms": sdnn_ms,
        "Fitted_hr_std_param": round(hr_std_param, 2),
        "Fitted_noise_param": round(noise_param, 4),
        "Synthetic_Signal_Samples": len(fitted_sig),
        "Boundary_Discontinuity_Aligned": True,
        "Amplitude_Units": "Arbitrary Units (a.u.)",
        "Pretraining_Suitability": "Unevaluated"
    })

# Plot empirically fitted synthetic ECG stages
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
fig.suptitle("Experiment 09: Empirical Real-Fitted Synthetic ECG Generation (PhysioNet szdb Parameters)\n"
             "Synthesized from Real Seizure Stage Parameters (Boundary-Discontinuity Aligned, Units: a.u.)",
             fontsize=11, fontweight='bold', y=0.995)

colors = ["steelblue", "darkorange", "purple", "seagreen"]
t_axis = np.arange(N_SAMPLES_PHASE) / float(FS)

for idx, (stage_name, sig) in enumerate(fitted_signals.items()):
    ax = axes[idx]
    ax.plot(t_axis, sig, color=colors[idx], linewidth=0.8)
    row_info = next(r for r in records_09 if r["Seizure_Stage"] == stage_name)
    ax.set_title(f"Stage {idx+1}: {stage_name.replace('_', ' ')}  |  HR: {row_info['Target_Mean_HR_BPM']} BPM  |  hr_std: {row_info['Fitted_hr_std_param']}  |  Noise: {row_info['Fitted_noise_param']}", fontsize=9, fontweight='bold')
    ax.set_ylabel("Amplitude (a.u.)")
    ax.grid(True, linestyle="--", alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout(rect=[0, 0, 1, 0.98])
out_img = "outputs/09_empirical_synthetic_fitting.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 09 complete. Saved visualization to {out_img}")
plt.close()

# Save summary CSV
csv_path = "outputs/09_fitted_synthetic_signals.csv"
df_09 = pd.DataFrame(records_09)
df_09.to_csv(csv_path, index=False)
print(f"Fitted synthetic parameters saved to {csv_path}\n")

