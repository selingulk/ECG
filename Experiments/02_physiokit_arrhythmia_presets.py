"""
Experiment 02: Comprehensive Evaluation of physioKIT Arrhythmia Presets
========================================================================
Research Question: Which pathological arrhythmia waveforms can physioKIT generate,
what are their morphological signatures, and what are their limitations?

Presets Evaluated (Iterated over pk.ecg.EcgPreset):
  1. SR: Normal Sinus Rhythm
  2. AFIB: Atrial Fibrillation (Irregular R-R intervals, absent P-waves)
  3. ant_STEMI: Anterior ST-Elevation Myocardial Infarction (ST segment elevation)
  4. LAHB: Left Anterior Hemiblock (Left axis deviation)
  5. LPHB: Left Posterior Hemiblock (Right axis deviation)
  6. high_take_off: High Take-Off / Early Repolarization (J-point elevation)
  7. LBBB: Left Bundle Branch Block (Widened QRS, notched R-wave)
  8. random_morphology: Randomized parameter synthesis
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import physiokit as pk

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 02: physioKIT Arrhythmia Presets ===")

presets = list(pk.ecg.EcgPreset)
FS = 500
DURATION = 5  # 5 seconds
N_SAMPLES = FS * DURATION
TIME_AXIS = np.linspace(0, DURATION, N_SAMPLES)

preset_info = [
    {"name": "SR", "desc": "Normal Sinus Rhythm", "morphology": "Normal P-QRS-T complex, regular R-R intervals"},
    {"name": "AFIB", "desc": "Atrial Fibrillation", "morphology": "Irregular R-R intervals, chaotic baseline (no P-wave)"},
    {"name": "ant_STEMI", "desc": "Anterior ST-Elevation MI", "morphology": "Significant ST-segment elevation above baseline"},
    {"name": "LAHB", "desc": "Left Anterior Hemiblock", "morphology": "Q1S3 pattern, left axis deviation alterations"},
    {"name": "LPHB", "desc": "Left Posterior Hemiblock", "morphology": "S1Q3 pattern, right axis deviation alterations"},
    {"name": "high_take_off", "desc": "High Take-Off / Early Repolarization", "morphology": "Elevated J-point with concave ST elevation"},
    {"name": "LBBB", "desc": "Left Bundle Branch Block", "morphology": "Widened QRS (>120 ms), notched R-wave, inverted T-wave"},
    {"name": "random_morphology", "desc": "Randomized Parameters", "morphology": "Stochastic variation of wave amplitudes & intervals"}
]

fig, axes = plt.subplots(len(presets), 1, figsize=(12, 16), sharex=True)
fig.suptitle("Experiment 02: physioKIT Pathological Arrhythmia Presets Benchmark\n"
             "All 8 EcgPreset options evaluated side-by-side (5s, 500 Hz, Lead 0)",
             fontsize=12, fontweight='bold', y=0.995)

csv_records = []

for idx, p_enum in enumerate(presets):
    p_name = p_enum.value
    # Find matching meta
    meta = next((item for item in preset_info if item["name"] == p_name),
                {"desc": p_name, "morphology": "Synthesized preset waveform"})

    signal_array, _, _ = pk.ecg.synthesize(
        signal_length=N_SAMPLES,
        sample_rate=FS,
        leads=1,
        heart_rate=70,
        preset=p_enum,
        noise_multiplier=0.02
    )
    ecg_lead0 = signal_array[0]

    ax = axes[idx]
    color = "darkred" if p_name in ["AFIB", "ant_STEMI", "LBBB"] else "steelblue"
    ax.plot(TIME_AXIS, ecg_lead0, color=color, linewidth=1.0)
    ax.set_title(f"Preset {idx+1}: {p_name} — {meta['desc']} ({meta['morphology']})", fontsize=8.5)
    ax.set_ylabel("Amp (a.u.)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    csv_records.append({
        "Preset_Name": p_name,
        "Clinical_Description": meta["desc"],
        "Morphological_Signature": meta["morphology"],
        "Is_Arrhythmia": "Yes" if p_name != "SR" else "No",
        "Leads_Supported": "Up to 12 leads",
        "Clinical_Validity_Note": "Rule-based mathematical preset; suitable for AI pretraining, not clinical diagnosis."
    })

axes[-1].set_xlabel("Time (seconds)")
plt.tight_layout(rect=[0, 0, 1, 0.99])
out_img = "outputs/02_physiokit_arrhythmia_presets.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 02 complete. Saved visualization to {out_img}")
plt.close()

# Save summary CSV
csv_path = "outputs/02_physiokit_presets_summary.csv"
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=csv_records[0].keys())
    writer.writeheader()
    writer.writerows(csv_records)
print(f"Results saved to {csv_path}\n")
