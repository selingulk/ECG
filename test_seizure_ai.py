"""
Quick Interactive Test Script: AI Seizure & Arrhythmia Classifier
===================================================================
Run this script to test the trained AI model on synthetic or custom ECG signals.
Outputs predicted class and confidence probabilities across 4 categories:
  - Class 0: Interictal Baseline (Normal)
  - Class 1: Preictal Warning (Seizure Prediction)
  - Class 2: Ictal Seizure (Active Seizure)
  - Class 3: Hard Negative Arrhythmia (Non-Epileptic Heart Condition)
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
import scipy.signal as sp_signal
from sklearn.ensemble import RandomForestClassifier

FS = 200
WINDOW_LEN = 1000

# 1. Load parameters and train model
params_csv = "outputs/08_empirical_seizure_parameters.csv"
df_params = pd.read_csv(params_csv)
c_col = "Clinical_Class" if "Clinical_Class" in df_params.columns else "Seizure_Stage"
param_dict = {row[c_col]: row for _, row in df_params.iterrows()}


def extract_multi_domain_features(signal, fs=FS):
    f_mean = float(np.mean(signal))
    f_std  = float(np.std(signal))
    f_ptp  = float(np.ptp(signal))
    f_rms  = float(np.sqrt(np.mean(signal**2)))
    zero_crossings = float(np.sum(np.diff(signal > 0) != 0))
    crest_factor = float(np.max(np.abs(signal)) / (f_rms + 1e-8))
    
    f, psd = sp_signal.welch(signal, fs=fs, nperseg=min(256, len(signal)))
    emg_idx = (f >= 20.0) & (f <= 100.0)
    emg_power = float(np.trapz(psd[emg_idx], f[emg_idx])) if np.any(emg_idx) else 0.0
    lf_idx = (f >= 0.04) & (f <= 0.15)
    hf_idx = (f >= 0.15) & (f <= 0.40)
    lf_power = float(np.trapz(psd[lf_idx], f[lf_idx])) if np.any(lf_idx) else 1e-6
    hf_power = float(np.trapz(psd[hf_idx], f[hf_idx])) if np.any(hf_idx) else 1e-6
    lf_hf_ratio = float(lf_power / (hf_power + 1e-8))
    
    return [f_mean, f_std, f_ptp, f_rms, zero_crossings, crest_factor, emg_power, lf_hf_ratio]

def generate_window(class_name, seed):
    info = param_dict.get(class_name, {"Mean_HR_BPM": 75.0, "SDNN_ms": 30.0, "EMG_Band_Power": 0.01})
    method_sim = "multichannel" if class_name == "Hard_Negative_Arrhythmia" else "ecgsyn"
    sig = nk.ecg_simulate(duration=5, sampling_rate=FS, heart_rate=float(info["Mean_HR_BPM"]), noise=float(info["EMG_Band_Power"])*5.0, method=method_sim, random_state=seed)
    if isinstance(sig, pd.DataFrame):
        sig = sig.iloc[:, 0].values
    elif isinstance(sig, np.ndarray) and sig.ndim > 1:
        sig = sig.flatten()
    return sig[:WINDOW_LEN]


# Build training set
classes = ["Interictal_Baseline", "Preictal_Prediction", "Ictal_Seizure", "Hard_Negative_Arrhythmia"]
X_tr, y_tr = [], []
for c_idx, c_name in enumerate(classes):
    for i in range(50):
        sig = generate_window(c_name, seed=100 + c_idx*1000 + i)
        X_tr.append(extract_multi_domain_features(sig))
        y_tr.append(c_idx)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_tr, y_tr)

print("=========================================================================")
print("          AI EPILEPTIC SEIZURE & ARRHYTHMIA TEST INFERENCE               ")
print("=========================================================================")

test_samples = [
    ("Test Sample 1 (Normal Resting)", "Interictal_Baseline", 9991),
    ("Test Sample 2 (Early Warning)", "Preictal_Prediction", 9992),
    ("Test Sample 3 (Active Seizure)", "Ictal_Seizure", 9993),
    ("Test Sample 4 (Cardiac Arrhythmia)", "Hard_Negative_Arrhythmia", 9994)
]

for title, true_class, seed in test_samples:
    test_sig = generate_window(true_class, seed=seed)
    feats = extract_multi_domain_features(test_sig)
    pred_idx = clf.predict([feats])[0]
    probs = clf.predict_proba([feats])[0]
    
    pred_name = classes[pred_idx]
    confidence = probs[pred_idx] * 100.0
    
    status = "SUCCESS (Correctly Classified)" if pred_name == true_class else "MISCLASSIFIED"
    print(f"\n{title}:")
    print(f"  - Target Ground Truth: {true_class}")
    print(f"  - AI Prediction:       {pred_name} ({confidence:.1f}% confidence)")
    print(f"  - Status:              {status}")

print("\n=========================================================================")
