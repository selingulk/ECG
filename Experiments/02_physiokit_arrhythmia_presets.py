"""
Experiment 02: Evaluation of physioKIT Pathological Presets (Arrhythmias & ECG Abnormalities)
=============================================================================================
Research Question: Which pathological ECG waveforms can physioKIT generate,
how are they categorized clinically (Arrhythmias vs. Structural/Conduction Abnormalities),
and what are their diagnostic and synthesis limitations?

Presets Evaluated (Iterated over pk.ecg.EcgPreset):
  1. SR: Normal Sinus Rhythm (Baseline)
  2. AFIB: Cardiac Arrhythmia (Atrial Fibrillation)
  3. ant_STEMI: Ischemic Injury Abnormality (Anterior ST-Elevation Myocardial Infarction)
  4. LAHB: Conduction System Abnormality (Left Anterior Hemiblock)
  5. LPHB: Conduction System Abnormality (Left Posterior Hemiblock)
  6. high_take_off: Repolarization Variant Abnormality (Early Repolarization)
  7. LBBB: Conduction System Abnormality (Left Bundle Branch Block)
  8. random_morphology: Stochastic Parameter Synthesis
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import physiokit as pk

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 02: physioKIT Pathological Presets ===")

presets = list(pk.ecg.EcgPreset)
FS = 500
DURATION = 5  # 5 seconds
N_SAMPLES = FS * DURATION
TIME_AXIS = np.arange(N_SAMPLES) / float(FS)

preset_info = [
    {"name": "SR", "category": "Normal Baseline", "is_arrhythmia": "No", "desc": "Normal Sinus Rhythm", "morphology": "Normal P-QRS-T complex, regular R-R intervals"},
    {"name": "AFIB", "category": "Cardiac Arrhythmia", "is_arrhythmia": "Yes (Arrhythmia)", "desc": "Atrial Fibrillation", "morphology": "Irregular R-R intervals, chaotic baseline (absent P-waves)"},
    {"name": "ant_STEMI", "category": "Ischemic Pathology", "is_arrhythmia": "No (ECG Abnormality)", "desc": "Anterior ST-Elevation MI", "morphology": "Significant ST-segment elevation above baseline"},
    {"name": "LAHB", "category": "Conduction Abnormality", "is_arrhythmia": "No (ECG Abnormality)", "desc": "Left Anterior Hemiblock", "morphology": "Q1S3 pattern, left axis deviation alterations"},
    {"name": "LPHB", "category": "Conduction Abnormality", "is_arrhythmia": "No (ECG Abnormality)", "desc": "Left Posterior Hemiblock", "morphology": "S1Q3 pattern, right axis deviation alterations"},
    {"name": "high_take_off", "category": "Repolarization Variant", "is_arrhythmia": "No (ECG Abnormality)", "desc": "Early Repolarization", "morphology": "Elevated J-point with concave ST elevation"},
    {"name": "LBBB", "category": "Conduction Abnormality", "is_arrhythmia": "No (ECG Abnormality)", "desc": "Left Bundle Branch Block", "morphology": "Widened QRS (>120 ms), notched R-wave, inverted T-wave"},
    {"name": "random_morphology", "category": "Stochastic Synthesis", "is_arrhythmia": "Variable", "desc": "Randomized Parameters", "morphology": "Stochastic variation of wave amplitudes & intervals"}
]

fig, axes = plt.subplots(len(presets), 1, figsize=(12, 16), sharex=True)
fig.suptitle("Experiment 02: physioKIT Pathological Presets Benchmark\n"
             "Distinguishing Cardiac Arrhythmias from Conduction & Ischemic ECG Abnormalities (5s, 500 Hz, Lead II index 1)",
             fontsize=12, fontweight='bold', y=0.995)

csv_records = []

for idx, p_enum in enumerate(presets):
    p_name = p_enum.value
    meta = next((item for item in preset_info if item["name"] == p_name),
                {"category": "Synthetic Preset", "is_arrhythmia": "Unknown", "desc": p_name, "morphology": "Synthesized preset waveform"})

    signal_array, _, _ = pk.ecg.synthesize(
        signal_length=N_SAMPLES,
        sample_rate=FS,
        leads=12,
        heart_rate=70,
        preset=p_enum,
        noise_multiplier=0.02
    )
    # Lead II (index 1)
    ecg_lead = signal_array[1]

    ax = axes[idx]
    color = "darkred" if p_name in ["AFIB", "ant_STEMI", "LBBB"] else "steelblue"
    ax.plot(TIME_AXIS, ecg_lead, color=color, linewidth=1.0)
    ax.set_title(f"Preset {idx+1}: {p_name} — {meta['desc']} [{meta['category']}] ({meta['morphology']})", fontsize=8.5)
    ax.set_ylabel("Amplitude (a.u.)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    csv_records.append({
        "Preset_Name": p_name,
        "Clinical_Category": meta["category"],
        "Clinical_Description": meta["desc"],
        "Morphological_Signature": meta["morphology"],
        "Is_Arrhythmia": meta["is_arrhythmia"],
        "Downstream_Pretraining_Suitability": "Unevaluated (Requires evaluation on real clinical data and downstream diagnostic models)",
        "Clinical_Validity_Note": "Rule-based synthetic preset for demonstration; not validated for clinical diagnostic pretraining."
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
