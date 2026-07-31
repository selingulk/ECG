"""
Experiment 06: Quantitative Filtering Benchmark across Distinct Target SNRs
=============================================================================
Research Question: How effectively do different toolkit preprocessing filters
improve signal quality across well-separated, standardized input SNRs (+20 dB, +10 dB, 0 dB, -5 dB)?

Fixes Applied:
  - Strongly separated target input SNRs: +20 dB (clean), +10 dB (moderate), 0 dB (heavy), -5 dB (extreme).
  - Consistent amplitude normalization: All outputs normalized to zero-mean unit-variance
    prior to computing SNR improvement (Delta SNR in dB) to ensure fair comparison.

Formula:
  SNR_in  = 10 * log10( var(clean) / var(noise_added) )
  SNR_out = 10 * log10( var(clean) / var(processed_normalized - clean_normalized) )
  Delta_SNR = SNR_out - SNR_in  (dB improvement, higher is better)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import physiokit as pk
import biosppy.signals.ecg as bsp_ecg
import scipy.signal as sp_signal

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 06: Noise Filtering SNR Benchmark ===")

FS = 1000  # Hz
DURATION = 10
N_SAMPLES = FS * DURATION
TIME_AXIS = np.linspace(0, DURATION, N_SAMPLES)

clean_signal = nk.ecg_simulate(duration=DURATION, sampling_rate=FS, heart_rate=70, noise=0, random_state=42)
clean_norm = (clean_signal - np.mean(clean_signal)) / np.std(clean_signal)

TARGET_SNRS_DB = [20.0, 10.0, 0.0, -5.0]

def add_noise_at_snr(signal, target_snr_db, seed=42):
    np.random.seed(seed)
    signal_power = np.var(signal)
    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    # Add baseline wander (0.3 Hz)
    bw = 0.2 * np.sin(2 * np.pi * 0.3 * TIME_AXIS)
    return signal + noise + bw, noise + bw

def compute_snr_db(reference, test_signal):
    ref_norm = (reference - np.mean(reference)) / np.std(reference)
    test_norm = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    error = test_norm - ref_norm
    var_err = np.var(error)
    if var_err == 0:
        return float('inf')
    return float(10.0 * np.log10(np.var(ref_norm) / var_err))

results = []

fig, axes = plt.subplots(len(TARGET_SNRS_DB), 1, figsize=(12, 14), sharex=True)
fig.suptitle("Experiment 06: Filtering Performance across Standardized Input SNRs (+20 dB to -5 dB)\n"
             "Quantitative metric: Delta SNR (dB) = SNR_out - SNR_in (higher is better)",
             fontsize=12, fontweight='bold', y=0.995)

for idx, snr_target in enumerate(TARGET_SNRS_DB):
    noisy_sig, noise_added = add_noise_at_snr(clean_signal, snr_target)
    snr_in_actual = compute_snr_db(clean_signal, noisy_sig)

    # 1. NeuroKit2
    nk_clean = nk.ecg_clean(noisy_sig, sampling_rate=FS, method="neurokit")
    snr_out_nk = compute_snr_db(clean_signal, nk_clean)
    delta_snr_nk = snr_out_nk - snr_in_actual

    # 2. physioKIT
    pk_clean = pk.ecg.clean(noisy_sig, lowcut=0.5, highcut=35, sample_rate=FS, order=3, forward_backward=True)
    snr_out_pk = compute_snr_db(clean_signal, pk_clean)
    delta_snr_pk = snr_out_pk - snr_in_actual

    # 3. BioSPPy
    bsp_out = bsp_ecg.ecg(signal=noisy_sig, sampling_rate=FS, show=False)
    bsp_clean = bsp_out["filtered"]
    snr_out_bsp = compute_snr_db(clean_signal, bsp_clean)
    delta_snr_bsp = snr_out_bsp - snr_in_actual

    # 4. SciPy
    b_bp, a_bp = sp_signal.butter(3, [0.5 / (FS/2), 35.0 / (FS/2)], btype='bandpass')
    sp_clean_bp = sp_signal.filtfilt(b_bp, a_bp, noisy_sig)
    b_notch, a_notch = sp_signal.iirnotch(w0=50, Q=30, fs=FS)
    sp_clean = sp_signal.filtfilt(b_notch, a_notch, sp_clean_bp)
    snr_out_sp = compute_snr_db(clean_signal, sp_clean)
    delta_snr_sp = snr_out_sp - snr_in_actual

    results.append({
        "Target_SNR_dB": snr_target,
        "Actual_SNR_in_dB": round(snr_in_actual, 2),
        "Delta_SNR_BioSPPy_dB": round(delta_snr_bsp, 2),
        "Delta_SNR_physioKIT_dB": round(delta_snr_pk, 2),
        "Delta_SNR_NeuroKit2_dB": round(delta_snr_nk, 2),
        "Delta_SNR_SciPy_dB": round(delta_snr_sp, 2)
    })

    ax = axes[idx]
    ax.plot(TIME_AXIS, noisy_sig, color='lightgray', alpha=0.7, label=f"Noisy input (SNR_in = {snr_in_actual:.1f} dB)")
    ax.plot(TIME_AXIS, bsp_clean, color='mediumpurple', alpha=0.8, label=f"BioSPPy (+{delta_snr_bsp:.1f} dB)")
    ax.plot(TIME_AXIS, pk_clean, color='darkorange', alpha=0.8, label=f"physioKIT (+{delta_snr_pk:.1f} dB)")
    ax.plot(TIME_AXIS, nk_clean, color='steelblue', alpha=0.8, label=f"NeuroKit2 (+{delta_snr_nk:.1f} dB)")
    ax.plot(TIME_AXIS, sp_clean, color='seagreen', alpha=0.8, label=f"SciPy (+{delta_snr_sp:.1f} dB)")

    ax.set_title(f"Target Input SNR: {snr_target:+.0f} dB (Actual SNR_in: {snr_in_actual:.1f} dB)", fontsize=9.5)
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', fontsize=8, ncol=5)
    ax.grid(True, linestyle='--', alpha=0.4)

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout(rect=[0, 0, 1, 0.99])
out_img = "outputs/06_noise_filtering_snr_benchmark.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 06 complete. Saved visualization to {out_img}")
plt.close()

# Save summary CSV
csv_path = "outputs/06_snr_filtering_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"Results saved to {csv_path}\n")
