"""
Experiment 07: R-Peak Detection Performance Benchmark across 10 MIT-BIH Records
================================================================================
Research Question: What is the exact quantitative performance (Sensitivity, PPV, F1-Score)
of each toolkit's R-peak detector when evaluated against cardiologist (.atr) annotations?

Evaluation Protocol:
  - Dataset: 10 MIT-BIH Arrhythmia Database records (100, 101, 102, 103, 105, 106, 119, 200, 201, 203).
  - Scope: First 100 seconds of each recording (36,000 samples at 360 Hz).
  - Protocol: Common EC57-style evaluation using a ±150 ms tolerance window for matching detected to reference peaks.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import physiokit as pk
import biosppy.signals.ecg as bsp_ecg
import wfdb
from wfdb import processing
import scipy.signal as sp_signal

os.makedirs("outputs", exist_ok=True)

print("=== Running Experiment 07: QRS Detection Benchmark (First 100s, 10 MIT-BIH Records) ===")

RECORDS = ["100", "101", "102", "103", "105", "106", "119", "200", "201", "203"]
FS_MITDB = 360
SAMPLE_WINDOW = 36000  # First 100 seconds
TOLERANCE_SAMPLES = int(0.150 * FS_MITDB)  # ±150 ms tolerance window
BEAT_SYMBOLS = set("NLRBAaJSVFejnE/fQ?")
LIBRARIES = ["WFDB", "BioSPPy", "NeuroKit2", "physioKIT", "SciPy"]

def scipy_pan_tompkins(signal, fs):
    nyq = fs / 2.0
    b, a = sp_signal.butter(2, [5.0 / nyq, 15.0 / nyq], btype='bandpass')
    filtered = sp_signal.filtfilt(b, a, signal)
    diff = np.diff(filtered, prepend=filtered[0])
    squared = diff ** 2
    win = int(0.15 * fs)
    kernel = np.ones(win) / win
    integrated = np.convolve(squared, kernel, mode='same')
    threshold = np.percentile(integrated, 85)
    min_dist = int(0.3 * fs)
    peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=min_dist)
    return peaks

def evaluate_detector(detected_peaks, reference_peaks, tol=TOLERANCE_SAMPLES):
    if len(detected_peaks) == 0:
        return 0, 0, len(reference_peaks), 0.0, 0.0, 0.0

    tp = 0
    fp = 0
    ref_matched = np.zeros(len(reference_peaks), dtype=bool)

    for det in detected_peaks:
        distances = np.abs(reference_peaks - det)
        min_idx = np.argmin(distances)
        if distances[min_idx] <= tol and not ref_matched[min_idx]:
            tp += 1
            ref_matched[min_idx] = True
        else:
            fp += 1

    fn = len(reference_peaks) - tp
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0.0

    return tp, fp, fn, sensitivity, ppv, f1

all_record_results = []
metrics_per_lib = {lib: {"se": [], "ppv": [], "f1": []} for lib in LIBRARIES}

for rec_id in RECORDS:
    record = wfdb.rdrecord(rec_id, sampfrom=0, sampto=SAMPLE_WINDOW, pn_dir="mitdb")
    ann = wfdb.rdann(rec_id, "atr", sampfrom=0, sampto=SAMPLE_WINDOW, pn_dir="mitdb")
    signal = record.p_signal[:, 0]

    beat_mask = np.array([s in BEAT_SYMBOLS for s in ann.symbol])
    ref_peaks = ann.sample[beat_mask]

    lib_peaks = {}
    try:
        lib_peaks["WFDB"] = processing.gqrs_detect(sig=signal, fs=FS_MITDB)
    except Exception:
        lib_peaks["WFDB"] = np.array([])

    try:
        out = bsp_ecg.ecg(signal=signal, sampling_rate=FS_MITDB, show=False)
        lib_peaks["BioSPPy"] = out["rpeaks"]
    except Exception:
        lib_peaks["BioSPPy"] = np.array([])

    try:
        _, info = nk.ecg_peaks(signal, sampling_rate=FS_MITDB, method="neurokit")
        lib_peaks["NeuroKit2"] = info["ECG_R_Peaks"]
    except Exception:
        lib_peaks["NeuroKit2"] = np.array([])

    try:
        lib_peaks["physioKIT"] = pk.ecg.find_peaks(signal, sample_rate=FS_MITDB)
    except Exception:
        lib_peaks["physioKIT"] = np.array([])

    try:
        lib_peaks["SciPy"] = scipy_pan_tompkins(signal, FS_MITDB)
    except Exception:
        lib_peaks["SciPy"] = np.array([])

    for lib in LIBRARIES:
        tp, fp, fn, se, ppv, f1 = evaluate_detector(lib_peaks[lib], ref_peaks)
        metrics_per_lib[lib]["se"].append(se)
        metrics_per_lib[lib]["ppv"].append(ppv)
        metrics_per_lib[lib]["f1"].append(f1)

        all_record_results.append({
            "Record": rec_id,
            "Library": lib,
            "Ref_Beats": len(ref_peaks),
            "Det_Beats": len(lib_peaks[lib]),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Sensitivity": round(se, 4),
            "PPV": round(ppv, 4),
            "F1_Score": round(f1, 4)
        })

summary_rows = []
print("=== R-Peak Detection Performance Summary (Mean ± Std, N=10 Records) ===")
for lib in LIBRARIES:
    m_se, s_se = np.mean(metrics_per_lib[lib]["se"])*100, np.std(metrics_per_lib[lib]["se"])*100
    m_ppv, s_ppv = np.mean(metrics_per_lib[lib]["ppv"])*100, np.std(metrics_per_lib[lib]["ppv"])*100
    m_f1, s_f1 = np.mean(metrics_per_lib[lib]["f1"])*100, np.std(metrics_per_lib[lib]["f1"])*100

    summary_rows.append({
        "Library": lib,
        "Sensitivity (%)": f"{m_se:.2f} ± {s_se:.2f}",
        "PPV (%)": f"{m_ppv:.2f} ± {s_ppv:.2f}",
        "F1_Score (%)": f"{m_f1:.2f} ± {s_f1:.2f}"
    })
    print(f"{lib:10s} | Se: {m_se:6.2f}% ± {s_se:4.2f}% | PPV: {m_ppv:6.2f}% ± {s_ppv:4.2f}% | F1: {m_f1:6.2f}% ± {s_f1:4.2f}%")

# Save detailed CSV
csv_detail = "outputs/07_qrs_detection_detailed_results.csv"
with open(csv_detail, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_record_results[0].keys())
    writer.writeheader()
    writer.writerows(all_record_results)

# Save summary CSV
csv_summary = "outputs/07_qrs_detection_summary.csv"
with open(csv_summary, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# Plot summary metrics bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(LIBRARIES))
width = 0.25

se_means = [np.mean(metrics_per_lib[l]["se"])*100 for l in LIBRARIES]
se_stds  = [np.std(metrics_per_lib[l]["se"])*100 for l in LIBRARIES]
ppv_means = [np.mean(metrics_per_lib[l]["ppv"])*100 for l in LIBRARIES]
ppv_stds  = [np.std(metrics_per_lib[l]["ppv"])*100 for l in LIBRARIES]
f1_means = [np.mean(metrics_per_lib[l]["f1"])*100 for l in LIBRARIES]
f1_stds  = [np.std(metrics_per_lib[l]["f1"])*100 for l in LIBRARIES]

ax.bar(x - width, se_means, width, yerr=se_stds, label='Sensitivity (%)', capsize=4, color='steelblue')
ax.bar(x, ppv_means, width, yerr=ppv_stds, label='PPV (%)', capsize=4, color='darkorange')
ax.bar(x + width, f1_means, width, yerr=f1_stds, label='F1-Score (%)', capsize=4, color='seagreen')

ax.set_title("Experiment 07: R-Peak Detection Accuracy across 10 MIT-BIH Records (First 100s)\nCommon EC57-Style Matching Protocol (±150 ms Window)", fontsize=11, fontweight='bold')
ax.set_ylabel("Performance (%)")
ax.set_xticks(x)
ax.set_xticklabels(LIBRARIES, fontsize=10)
ax.set_ylim([70, 105])
ax.legend(loc='lower right')
ax.grid(True, linestyle='--', alpha=0.4, axis='y')

plt.tight_layout()
out_img = "outputs/07_qrs_detection_multi_record.png"
plt.savefig(out_img, dpi=300, bbox_inches="tight")
print(f"Experiment 07 complete. Saved visualization to {out_img}\n")
plt.close()
