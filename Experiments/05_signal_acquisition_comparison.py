"""
Experiment 05: ECG Signal Acquisition & Multi-Toolkit Representation Audit
=============================================================================
Research Question: Which toolkits natively support signal acquisition (synthesis vs. real data I/O),
and what are their output data structures and physical unit properties?

Fixes Applied:
  - Strict 10.0-second duration across ALL panels: WFDB calculates wfdb_samples = int(10 * wfdb_fs).
  - Time axis defined via np.arange(len(sig)) / sampling_rate across all plots.
  - Physical unit labels explicitly marked (model-generated a.u. vs calibrated mV).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import physiokit as pk
import wfdb

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 05: Signal Acquisition Comparison ===")

DURATION = 10.0  # seconds across all panels
FS_SYNTH = 500
N_SYNTH  = int(DURATION * FS_SYNTH)
t_synth  = np.arange(N_SYNTH) / FS_SYNTH

# 1. NeuroKit2 — Synthetic
ecg_nk = nk.ecg_simulate(duration=DURATION, sampling_rate=FS_SYNTH, heart_rate=70, noise=0.01, random_state=42)

# 2. physioKIT — Synthetic
pk_sig, _, _ = pk.ecg.synthesize(signal_length=N_SYNTH, sample_rate=FS_SYNTH, leads=1, heart_rate=70, preset=pk.ecg.EcgPreset.SR, noise_multiplier=0.05)
ecg_pk = pk_sig[0]

# 3. BioSPPy — No synthesis

# 4. WFDB — Real PhysioNet Record 100 (exactly 10 seconds = 3600 samples at 360 Hz)
wfdb_record = wfdb.rdrecord("100", sampfrom=0, sampto=3600, pn_dir="mitdb")
ecg_wfdb = wfdb_record.p_signal[:, 0]
wfdb_fs = wfdb_record.fs
t_wfdb = np.arange(len(ecg_wfdb)) / wfdb_fs

# 5. SciPy — No synthesis

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
fig.suptitle("Experiment 05: ECG Signal Acquisition & Data Representation Audit\n"
             "Strict 10.0-second window across all toolkits  |  Distinguishing Synthetic (a.u.) vs Real Data (mV)",
             fontsize=12, fontweight='bold', y=0.995)

# Panel 1: NeuroKit2
axes[0].plot(t_synth, ecg_nk, color='steelblue', linewidth=1.0)
axes[0].set_title("NeuroKit2  |  nk.ecg_simulate()  |  Synthetic dynamical model  |  Units: Model arbitrary units (a.u.)", fontsize=9)
axes[0].set_ylabel("Amplitude (a.u.)")
axes[0].grid(True, linestyle='--', alpha=0.4)

# Panel 2: physioKIT
axes[1].plot(t_synth, ecg_pk, color='darkorange', linewidth=1.0)
axes[1].set_title("physioKIT  |  pk.ecg.synthesize()  |  Synthetic preset model  |  Units: Digital arbitrary units (a.u.)", fontsize=9)
axes[1].set_ylabel("Amplitude (a.u.)")
axes[1].grid(True, linestyle='--', alpha=0.4)
axes[1].text(0.01, 0.08, "NOTE: physioKIT output is in arbitrary units (a.u.); amplitude scale is not directly comparable to physical mV.",
            transform=axes[1].transAxes, fontsize=7.5, color='#8B4513',
            bbox=dict(boxstyle='round', facecolor='#fff3cd', edgecolor='#8B4513', alpha=0.9))

# Panel 3: BioSPPy — Not Supported
axes[2].set_facecolor('#f8f8f8')
axes[2].text(0.5, 0.5, "NOT SUPPORTED\n\nBioSPPy has no ECG synthesis functions.\nIt is a processing-only library requiring external signal inputs.",
            ha='center', va='center', transform=axes[2].transAxes, fontsize=10, color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdecea', edgecolor='#c0392b'))
axes[2].set_title("BioSPPy  |  No synthesis API", fontsize=9)
axes[2].set_yticks([])

# Panel 4: WFDB — Real Data
axes[3].plot(t_wfdb, ecg_wfdb, color='darkgreen', linewidth=0.8)
axes[3].set_title("WFDB  |  wfdb.rdrecord('100', pn_dir='mitdb')  |  REAL clinical recording (360 Hz)  |  Units: Calibrated mV", fontsize=9)
axes[3].set_ylabel("Amplitude (mV)")
axes[3].grid(True, linestyle='--', alpha=0.4)

# Panel 5: SciPy — Not Supported
axes[4].set_facecolor('#f8f8f8')
axes[4].text(0.5, 0.5, "NOT SUPPORTED\n\nSciPy provides general DSP primitives (filters, FFTs, peak finders).\nIt contains no domain-specific ECG simulation functions.",
            ha='center', va='center', transform=axes[4].transAxes, fontsize=10, color='#c0392b',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fdecea', edgecolor='#c0392b'))
axes[4].set_title("SciPy  |  No synthesis API", fontsize=9)
axes[4].set_yticks([])
axes[4].set_xlabel("Time (seconds)")

axes[0].set_xlim(0, 10.0)

plt.tight_layout(rect=[0, 0, 1, 0.99])
out_img = "outputs/05_signal_acquisition_comparison.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 05 complete. Saved visualization to {out_img}\n")
plt.close()
