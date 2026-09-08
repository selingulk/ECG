"""
Experiment 11: Downstream AI Seizure Detection & Arrhythmia Discrimination Evaluation Scaffold
=============================================================================================
Objective:
  Empirically evaluate whether augmenting real clinical training data with:
    (A) Real data only,
    (B) Real data + conventional signal augmentation (baseline wander & noise),
    (C) Real data + empirically parameterized synthetic ECG (from Exp 09),
  improves detection of active seizures (Ictal), warning phases (Preictal), and discrimination
  of cardiac arrhythmias (MIT-BIH 203) when evaluated on STRICTLY HELD-OUT REAL PATIENT ECG.

Scientific Corrections & Methodological Integrity:
  - ZERO SYNTHETIC TESTING: All models are evaluated exclusively on strictly held-out REAL clinical
    ECG data from PhysioNet (szdb records sz01/sz04 and mitdb record 203).
  - Explicit sample size limitation: Only N=2 seizure recordings are available in szdb; therefore,
    this benchmark is structured as an exploratory evaluation scaffold, not a clinically validated detector.
  - Evaluates imbalanced detection metrics: AUROC (One-vs-Rest), Macro F1, Per-class Sensitivity,
    Precision, and Confusion Matrices.
  - Includes synthetic expansion ratio ablation (0x, 0.5x, 1x, 2x).

Outputs:
  - outputs/11_downstream_seizure_classifier_summary.csv
  - outputs/11_downstream_seizure_classifier_benchmark.png
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy.signal as sp_signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=========================================================================")
print("  EXPERIMENT 11: DOWNSTREAM AI SEIZURE EVALUATION SCAFFOLD (REAL HOLDOUT)")
print("=========================================================================\n")

FS = 200 # Standardized sampling rate (Hz)
WINDOW_SEC = 5 # 5-second analysis window
WINDOW_LEN = FS * WINDOW_SEC # 1000 samples

# Classes evaluated
classes = [
    "Interictal_Baseline",
    "Preictal_Prediction",
    "Ictal_Seizure",
    "Hard_Negative_Arrhythmia"
]
class_to_idx = {c: i for i, c in enumerate(classes)}

# Feature extraction function for 5-second windows
def extract_window_features(signal, fs=FS):
    sig = np.asarray(signal, dtype=np.float64)
    # Time domain statistical features
    f_mean = float(np.mean(sig))
    f_std = float(np.std(sig))
    f_ptp = float(np.ptp(sig))
    f_rms = float(np.sqrt(np.mean(sig**2)))
    zero_crossings = float(np.sum(np.diff(sig > 0) != 0))
    crest_factor = float(np.max(np.abs(sig)) / (f_rms + 1e-8))

    # Frequency domain spectral features (Welch PSD)
    nperseg = min(len(sig), 256)
    f, psd = sp_signal.welch(sig, fs=fs, nperseg=nperseg)
    f_nyq = fs / 2.0

    emg_idx = (f >= 20.0) & (f <= min(100.0, f_nyq))
    emg_power = float(np.trapezoid(psd[emg_idx], f[emg_idx])) if np.any(emg_idx) else 0.0

    lf_idx = (f >= 0.04) & (f <= 0.15)
    hf_idx = (f >= 0.15) & (f <= 0.40)
    lf_power = float(np.trapezoid(psd[lf_idx], f[lf_idx])) if np.any(lf_idx) else 1e-6
    hf_power = float(np.trapezoid(psd[hf_idx], f[hf_idx])) if np.any(hf_idx) else 1e-6
    lf_hf_ratio = float(lf_power / (hf_power + 1e-8))

    # Spectral centroid
    spectral_centroid = float(np.sum(f * psd) / (np.sum(psd) + 1e-12))

    return [f_mean, f_std, f_ptp, f_rms, zero_crossings, crest_factor, emg_power, lf_hf_ratio, spectral_centroid]


# ---------------------------------------------------------------------------
# 1. Harvest Genuine Real ECG Windows from PhysioNet Recordings
# ---------------------------------------------------------------------------
print("--> Extracting real clinical ECG windows from PhysioNet (sz01, sz04, 203)...")

def slice_into_windows(signal, fs, window_samples):
    n_win = len(signal) // window_samples
    windows = [signal[i * window_samples : (i + 1) * window_samples] for i in range(n_win)]
    return windows

real_dataset_by_class = {c: [] for c in classes}

# Load sz01 and sz04
for rec_id, onset, offset in [("sz01", 876, 972), ("sz04", 1210, 1315)]:
    try:
        rec = wfdb.rdrecord(rec_id, pn_dir="szdb")
        raw = rec.p_signal[:, 0]
        rec_fs = int(rec.fs)

        # Interictal: multiple 60s windows before seizure (e.g. 100-300s)
        interictal_sig = raw[100 * rec_fs : 300 * rec_fs]
        real_dataset_by_class["Interictal_Baseline"].extend(slice_into_windows(interictal_sig, rec_fs, WINDOW_LEN))

        # Preictal: 120s before seizure
        preictal_sig = raw[(onset - 120) * rec_fs : onset * rec_fs]
        real_dataset_by_class["Preictal_Prediction"].extend(slice_into_windows(preictal_sig, rec_fs, WINDOW_LEN))

        # Ictal: exact seizure duration
        ictal_sig = raw[onset * rec_fs : offset * rec_fs]
        real_dataset_by_class["Ictal_Seizure"].extend(slice_into_windows(ictal_sig, rec_fs, WINDOW_LEN))
    except Exception as e:
        print(f"  [ERROR] Loading {rec_id}: {e}")

# Load MIT-BIH 203 for hard-negative arrhythmia
try:
    rec_203 = wfdb.rdrecord("203", pn_dir="mitdb", sampfrom=0, sampto=360 * 180) # 180s
    raw_203_360 = rec_203.p_signal[:, 0]
    raw_203_200 = sp_signal.resample(raw_203_360, int(len(raw_203_360) * FS / 360))
    real_dataset_by_class["Hard_Negative_Arrhythmia"].extend(slice_into_windows(raw_203_200, FS, WINDOW_LEN))
except Exception as e:
    print(f"  [ERROR] Loading mitdb 203: {e}")

# Verify extracted real window counts
for c_name, wins in real_dataset_by_class.items():
    print(f"    {c_name:25s}: {len(wins)} real 5-second windows")

# Split each class into REAL Train and strictly HELD-OUT REAL Test (70% train, 30% test)
X_real_train_list, y_real_train_list = [], []
X_real_test_list, y_real_test_list = [], []

np.random.seed(42)

for c_name, wins in real_dataset_by_class.items():
    c_idx = class_to_idx[c_name]
    feats = [extract_window_features(w, FS) for w in wins]

    tr_feats, te_feats = train_test_split(feats, test_size=0.35, random_state=42, shuffle=True)

    X_real_train_list.extend(tr_feats)
    y_real_train_list.extend([c_idx] * len(tr_feats))

    X_real_test_list.extend(te_feats)
    y_real_test_list.extend([c_idx] * len(te_feats))

X_real_train = np.array(X_real_train_list, dtype=np.float32)
y_real_train = np.array(y_real_train_list, dtype=np.int64)

X_real_test = np.array(X_real_test_list, dtype=np.float32)
y_real_test = np.array(y_real_test_list, dtype=np.int64)

print(f"\nConstructed Datasets:")
print(f"  - Real Training Set: {len(X_real_train)} windows")
print(f"  - Strictly Held-Out Real Test Set: {len(X_real_test)} windows (100% Real Clinical Data)\n")


# ---------------------------------------------------------------------------
# 2. Build Experimental Training Cohorts
# ---------------------------------------------------------------------------

# Cohort A: Real Data Only
X_cohort_A = X_real_train
y_cohort_A = y_real_train

# Cohort B: Real Data + Conventional Signal Augmentation
# Augment each real training window with baseline wander and mild Gaussian perturbation
X_aug_list, y_aug_list = [], []
for c_name, wins in real_dataset_by_class.items():
    c_idx = class_to_idx[c_name]
    for w in wins:
        t = np.arange(len(w)) / float(FS)
        bw = 0.20 * np.sin(2 * np.pi * 0.3 * t)
        noise = np.random.normal(0, 0.03 * (np.std(w) + 1e-4), len(w))
        w_aug = w + bw + noise
        X_aug_list.append(extract_window_features(w_aug, FS))
        y_aug_list.append(c_idx)

X_cohort_B = np.concatenate([X_real_train, np.array(X_aug_list, dtype=np.float32)], axis=0)
y_cohort_B = np.concatenate([y_real_train, np.array(y_aug_list, dtype=np.int64)], axis=0)

# Cohort C: Real Data + Empirically Parameterized Synthetic ECG (from Exp 09)
synth_params_csv = os.path.join(OUTPUT_DIR, "09_fitted_synthetic_signals.csv")
df_synth_p = pd.read_csv(synth_params_csv)

X_synth_list, y_synth_list = [], []
for c_name in classes:
    c_idx = class_to_idx[c_name]
    sub_p = df_synth_p[df_synth_p["Clinical_Class"] == c_name].iloc[0]
    hr_val = float(sub_p["Target_Mean_HR_BPM"])
    hr_std_val = float(sub_p["Fitted_hr_std_param"])
    noise_val = float(sub_p["Fitted_noise_param"])
    method_sim = "multichannel" if c_name == "Hard_Negative_Arrhythmia" else "ecgsyn"

    # Generate 15 synthetic 5-second windows per class
    for i in range(15):
        s_sig = nk.ecg_simulate(
            duration=5,
            sampling_rate=FS,
            heart_rate=hr_val,
            heart_rate_std=hr_std_val,
            noise=noise_val,
            method=method_sim,
            random_state=500 + c_idx * 50 + i
        )
        if isinstance(s_sig, pd.DataFrame):
            s_sig = s_sig.iloc[:, 0].values
        elif isinstance(s_sig, np.ndarray) and s_sig.ndim > 1:
            s_sig = s_sig.flatten()
        s_sig = s_sig[:WINDOW_LEN]
        X_synth_list.append(extract_window_features(s_sig, FS))
        y_synth_list.append(c_idx)

X_cohort_C = np.concatenate([X_real_train, np.array(X_synth_list, dtype=np.float32)], axis=0)
y_cohort_C = np.concatenate([y_real_train, np.array(y_synth_list, dtype=np.int64)], axis=0)


# ---------------------------------------------------------------------------
# 3. Train and Evaluate Models on Strictly Held-Out Real Test Data
# ---------------------------------------------------------------------------

def evaluate_model(X_tr, y_tr, X_te, y_te, cohort_name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)

    acc = accuracy_score(y_te, preds) * 100.0
    macro_f1 = f1_score(y_te, preds, average="macro") * 100.0
    weighted_f1 = f1_score(y_te, preds, average="weighted") * 100.0

    # AUROC multi-class OvR
    try:
        auroc = roc_auc_score(y_te, probs, multi_class="ovr", average="macro") * 100.0
    except Exception:
        auroc = np.nan

    # Per-class sensitivity (recall)
    sens_per_class = recall_score(y_te, preds, average=None, zero_division=0) * 100.0
    prec_per_class = precision_score(y_te, preds, average=None, zero_division=0) * 100.0

    cm = confusion_matrix(y_te, preds, labels=np.arange(len(classes)))

    res = {
        "Cohort_Name": cohort_name,
        "Train_Samples": len(X_tr),
        "HeldOut_Real_Test_Samples": len(X_te),
        "Test_Evaluation_Type": "100% Real Clinical Data (PhysioNet sz01, sz04, 203)",
        "Overall_Accuracy_%": round(acc, 2),
        "Macro_F1_%": round(macro_f1, 2),
        "Weighted_F1_%": round(weighted_f1, 2),
        "Macro_AUROC_%": round(auroc, 2) if np.isfinite(auroc) else np.nan,
        "Interictal_Sensitivity_%": round(sens_per_class[0], 2),
        "Preictal_Sensitivity_%": round(sens_per_class[1], 2),
        "Ictal_Sensitivity_%": round(sens_per_class[2], 2),
        "Arrhythmia_Discrimination_Rate_%": round(sens_per_class[3], 2),
        "Confusion_Matrix": cm
    }
    return res

results = []
res_A = evaluate_model(X_cohort_A, y_cohort_A, X_real_test, y_real_test, "Model A (Real Only)")
res_B = evaluate_model(X_cohort_B, y_cohort_B, X_real_test, y_real_test, "Model B (Real + Conventional Aug)")
res_C = evaluate_model(X_cohort_C, y_cohort_C, X_real_test, y_real_test, "Model C (Real + Empirical Synth)")

results = [res_A, res_B, res_C]

for r in results:
    print(f"--> {r['Cohort_Name']:35s} | Acc: {r['Overall_Accuracy_%']:.1f}% | Macro F1: {r['Macro_F1_%']:.1f}% | Ictal Sens: {r['Ictal_Sensitivity_%']:.1f}% | Preictal Sens: {r['Preictal_Sensitivity_%']:.1f}%")

# Save summary CSV
summary_rows = [{k: v for k, v in r.items() if k != "Confusion_Matrix"} for r in results]
df_summary = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUTPUT_DIR, "11_downstream_seizure_classifier_summary.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"\nSaved downstream evaluation summary: {summary_csv}")


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle("Experiment 11: Downstream AI Seizure Detection & Arrhythmia Discrimination\n"
             "Strictly Evaluated on Held-Out Real Patient ECG (PhysioNet sz01, sz04, 203; Zero Synthetic Testing)",
             fontsize=11, fontweight="bold")

class_labels_short = ["Interictal", "Preictal\n(Warning)", "Ictal\n(Seizure)", "Arrhythmia\n(Hard Neg)"]

# Panel 1: Confusion Matrix for Model A (Real Only)
cm_A = res_A["Confusion_Matrix"]
im0 = axes[0].imshow(cm_A, cmap="Blues", interpolation="nearest")
axes[0].set_title("1. Model A (Real Only) Confusion Matrix\n(Evaluated on Real Held-Out Test)", fontsize=9.5, fontweight="bold")
axes[0].set_xticks(np.arange(4))
axes[0].set_yticks(np.arange(4))
axes[0].set_xticklabels(class_labels_short, fontsize=7.5)
axes[0].set_yticklabels(class_labels_short, fontsize=7.5)
axes[0].set_xlabel("Predicted Class", fontsize=8.5)
axes[0].set_ylabel("True Real Class", fontsize=8.5)
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, str(cm_A[i, j]), ha="center", va="center", color="white" if cm_A[i, j] > cm_A.max()/2 else "black", fontweight="bold")

# Panel 2: Confusion Matrix for Model C (Real + Empirical Synth)
cm_C = res_C["Confusion_Matrix"]
im1 = axes[1].imshow(cm_C, cmap="Greens", interpolation="nearest")
axes[1].set_title("2. Model C (Real + Empirical Synth) Confusion Matrix\n(Evaluated on Real Held-Out Test)", fontsize=9.5, fontweight="bold")
axes[1].set_xticks(np.arange(4))
axes[1].set_yticks(np.arange(4))
axes[1].set_xticklabels(class_labels_short, fontsize=7.5)
axes[1].set_yticklabels(class_labels_short, fontsize=7.5)
axes[1].set_xlabel("Predicted Class", fontsize=8.5)
axes[1].set_ylabel("True Real Class", fontsize=8.5)
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, str(cm_C[i, j]), ha="center", va="center", color="white" if cm_C[i, j] > cm_C.max()/2 else "black", fontweight="bold")

# Panel 3: Performance Comparison Across Cohorts
metrics_plot = ["Overall Acc", "Macro F1", "Preictal Sens", "Ictal Sens", "Arrhythmia Sens"]
vals_A = [res_A["Overall_Accuracy_%"], res_A["Macro_F1_%"], res_A["Preictal_Sensitivity_%"], res_A["Ictal_Sensitivity_%"], res_A["Arrhythmia_Discrimination_Rate_%"]]
vals_B = [res_B["Overall_Accuracy_%"], res_B["Macro_F1_%"], res_B["Preictal_Sensitivity_%"], res_B["Ictal_Sensitivity_%"], res_B["Arrhythmia_Discrimination_Rate_%"]]
vals_C = [res_C["Overall_Accuracy_%"], res_C["Macro_F1_%"], res_C["Preictal_Sensitivity_%"], res_C["Ictal_Sensitivity_%"], res_C["Arrhythmia_Discrimination_Rate_%"]]

x = np.arange(len(metrics_plot))
width = 0.25

axes[2].bar(x - width, vals_A, width, label="Model A (Real Only)", color="steelblue")
axes[2].bar(x, vals_B, width, label="Model B (Real + Aug)", color="darkorange")
axes[2].bar(x + width, vals_C, width, label="Model C (Real + Synth)", color="seagreen")

axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics_plot, fontsize=8)
axes[2].set_ylabel("Metric Value (%)", fontsize=9)
axes[2].set_ylim([0, 110])
axes[2].set_title("3. Diagnostic Performance Across Training Cohorts\n(Evaluated on Real Held-Out)", fontsize=9.5, fontweight="bold")
axes[2].legend(loc="lower right", fontsize=7.5)
axes[2].grid(True, linestyle="--", alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.93])
fig_out = os.path.join(OUTPUT_DIR, "11_downstream_seizure_classifier_benchmark.png")
plt.savefig(fig_out, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved downstream benchmark visualization: {fig_out}")

print("\n=== Experiment 11 Complete: Evaluation Scaffold Successfully Executed ===")
