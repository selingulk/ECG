"""
Experiment 08: Real Epileptic Seizure & Cardiac Arrhythmia Empirical Characterization Pipeline
=============================================================================================
Objective:
  Extract empirical physiological parameters (rhythm/HRV, morphology, beat templates,
  Welch spectral proxies, and signal quality) across distinct clinical seizure phases and real
  cardiac arrhythmia recordings from PhysioNet without synthetic fallback.

Datasets:
  - PhysioNet szdb: Post-Ictal Heart Rate Oscillations in Partial Epilepsy
    (Full available dataset: records sz01, sz02, sz03, sz04, sz05, sz06, sz07; 10 seizure events)
  - PhysioNet mitdb: MIT-BIH Arrhythmia Database
    (Record 203 with cardiologist .atr annotations as real cardiac hard-negative control)

Key Scientific Integrity Principles:
  - ZERO synthetic fallbacks (synthetic simulation is strictly prohibited).
  - Missing or uncalculable features produce NaN and explicit failure reasons.
  - Dynamically identifies ECG channels and sampling frequencies from WFDB metadata.
  - Preserves exact EEG-confirmed seizure onset, offset, and duration for ictal phases.
  - Distinguishes record_id from patient_id (patient identity set to unknown/NaN where unmapped).

Outputs:
  - outputs/08_empirical_ecg_feature_table.csv (Primary feature table with complete provenance)
  - outputs/08_rr_intervals.csv (Long-form beat-to-beat RR interval data)
  - outputs/08_beat_templates.npz (Aligned median beat templates across phases)
  - outputs/08_data_quality_report.csv (Quality validation and audit diagnostics)
  - outputs/08_dataset_summary.csv (Comprehensive summary across sz01-sz07 records)
  - outputs/08_real_seizure_phases.png (Representative real ECG waveform inspection figure)
  - outputs/08_real_feature_distributions.png (Empirical feature distribution figure across full dataset)
  - outputs/08_empirical_seizure_parameters.csv (Backward-compatible aggregate table)

Scientific Disclaimer:
  This script performs exploratory empirical characterization on real clinical recordings.
  Extracted morphology and spectral proxies are automated computational estimates and must not
  be interpreted as direct EMG measurements or clinically certified diagnostic metrics.
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
import urllib.request

# Guarantee outputs directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=========================================================================")
print("  EXPERIMENT 08: FULL SZDB REAL SEIZURE & ARRHYTHMIA CHARACTERIZATION")
print("=========================================================================\n")


# ---------------------------------------------------------------------------
# Step 1: Dataset Audit & Seizure Metadata Registry
# ---------------------------------------------------------------------------

SZDB_VERIFIED_SEIZURES = [
    {"record_id": "sz01", "seizure_id": "sz01_sz1", "onset_s": 876,  "offset_s": 972},   # 00:14:36 - 00:16:12 (96s)
    {"record_id": "sz02", "seizure_id": "sz02_sz1", "onset_s": 3763, "offset_s": 3823},  # 01:02:43 - 01:03:43 (60s)
    {"record_id": "sz02", "seizure_id": "sz02_sz2", "onset_s": 10551,"offset_s": 10576}, # 02:55:51 - 02:56:16 (25s)
    {"record_id": "sz03", "seizure_id": "sz03_sz1", "onset_s": 5074, "offset_s": 5182},  # 01:24:34 - 01:26:22 (108s)
    {"record_id": "sz03", "seizure_id": "sz03_sz2", "onset_s": 9267, "offset_s": 9377},  # 02:34:27 - 02:36:17 (110s)
    {"record_id": "sz04", "seizure_id": "sz04_sz1", "onset_s": 1210, "offset_s": 1315},  # 00:20:10 - 00:21:55 (105s)
    {"record_id": "sz05", "seizure_id": "sz05_sz1", "onset_s": 1447, "offset_s": 1530},  # 00:24:07 - 00:25:30 (83s)
    {"record_id": "sz06", "seizure_id": "sz06_sz1", "onset_s": 3085, "offset_s": 3139},  # 00:51:25 - 00:52:19 (54s)
    {"record_id": "sz06", "seizure_id": "sz06_sz2", "onset_s": 7485, "offset_s": 7570},  # 02:04:45 - 02:06:10 (85s)
    {"record_id": "sz07", "seizure_id": "sz07_sz1", "onset_s": 4082, "offset_s": 4171},  # 01:08:02 - 01:09:31 (89s)
]


def parse_hhmmss_to_seconds(t_str):
    """Converts HH:MM:SS string to seconds float."""
    parts = [int(p) for p in t_str.strip().split(":")]
    if len(parts) == 3:
        return float(parts[0] * 3600 + parts[1] * 60 + parts[2])
    elif len(parts) == 2:
        return float(parts[0] * 60 + parts[1])
    return float(parts[0])


def fetch_or_load_szdb_seizures():
    """
    Attempts to fetch authoritative seizure timings from PhysioNet times.seize.
    Falls back to verified clinical records if network access is unavailable.
    """
    url = "https://physionet.org/files/szdb/1.0.0/times.seize"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            text = resp.read().decode("utf-8")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            records = []
            rec_counts = {}
            for line in lines:
                parts = line.split()
                if len(parts) >= 3:
                    rec_id = parts[0]
                    rec_counts[rec_id] = rec_counts.get(rec_id, 0) + 1
                    sz_id = f"{rec_id}_sz{rec_counts[rec_id]}"
                    on_s = parse_hhmmss_to_seconds(parts[1])
                    off_s = parse_hhmmss_to_seconds(parts[2])
                    records.append({
                        "record_id": rec_id,
                        "seizure_id": sz_id,
                        "onset_s": on_s,
                        "offset_s": off_s
                    })
            if len(records) >= 10:
                print(f"[SZDB Audit] Successfully fetched {len(records)} seizure events from PhysioNet times.seize.")
                return records
    except Exception as e:
        print(f"[SZDB Audit] times.seize remote fetch note ({e}); using verified local clinical timestamp registry.")
    return SZDB_VERIFIED_SEIZURES


# ---------------------------------------------------------------------------
# Step 3 & 4: Channel Identification, Input Validation & Signal Quality
# ---------------------------------------------------------------------------

def identify_ecg_channel(record):
    """
    Identifies the most appropriate ECG channel from a WFDB Record or Header.
    Returns (channel_index, channel_name, units, identification_notes).
    Returns (None, None, None, error_notes) if no valid ECG channel is identified.
    """
    if record.n_sig == 0:
        return None, None, None, "Zero signal channels present in record."

    sig_names = [str(name).upper() for name in (record.sig_name or [])]
    preferred_keywords = ["ECG", "EKG", "MLII", "V1", "V2", "V3", "V4", "V5", "V6", "II", "I"]

    for kw in preferred_keywords:
        for idx, name in enumerate(sig_names):
            if kw == name or kw in name:
                units = record.units[idx] if (record.units and idx < len(record.units)) else "mV"
                return idx, record.sig_name[idx], units, f"Identified via keyword '{kw}'."

    # Secondary check: search units for mV / uV
    if record.units:
        for idx, u in enumerate(record.units):
            if u and str(u).lower() in ["mv", "uv"]:
                ch_name = record.sig_name[idx] if idx < len(record.sig_name) else f"channel_{idx}"
                return idx, ch_name, u, f"Identified via electrical unit '{u}'."

    return None, None, None, f"No ECG channel identified (channels={record.sig_name}, units={record.units})."


def validate_and_preprocess_signal(raw_signal, fs, max_nan_fraction=0.01):
    """
    Validates signal for emptiness, constants, flatlining, clipping, and NaNs.
    Performs minimal linear interpolation for sparse NaNs if fraction <= max_nan_fraction.
    Returns (processed_signal, nan_fraction, is_valid, quality_notes_list).
    """
    notes = []
    if raw_signal is None or len(raw_signal) == 0:
        return None, 1.0, False, ["Empty or None signal array"]

    n_samples = len(raw_signal)
    nan_mask = ~np.isfinite(raw_signal)
    nan_count = int(np.sum(nan_mask))
    nan_fraction = float(nan_count) / float(n_samples)

    if nan_fraction > max_nan_fraction:
        return None, nan_fraction, False, [f"Excessive NaN/Inf fraction ({nan_fraction:.4f} > {max_nan_fraction})"]

    signal_work = raw_signal.astype(np.float64).copy()
    if nan_count > 0:
        indices = np.arange(n_samples)
        valid_indices = indices[~nan_mask]
        signal_work[nan_mask] = np.interp(indices[nan_mask], valid_indices, signal_work[~nan_mask])
        notes.append(f"Interpolated {nan_count} sparse NaN samples ({nan_fraction*100:.2f}%)")

    std_val = float(np.std(signal_work))
    if std_val < 1e-6:
        return signal_work, nan_fraction, False, ["Flatline signal (std < 1e-6)"]

    return signal_work, nan_fraction, True, notes


def compute_signal_quality_metrics(raw_signal, cleaned_sig, nan_fraction, is_valid_input, peak_count, fs):
    """
    Calculates signal amplitude distribution, saturation/clipping, and multidimensional validity flags.
    """
    sig_mean = float(np.mean(raw_signal))
    sig_std = float(np.std(raw_signal))
    sig_rms = float(np.sqrt(np.mean(raw_signal**2)))
    sig_ptp = float(np.ptp(raw_signal))
    p95 = float(np.percentile(raw_signal, 95))
    p5 = float(np.percentile(raw_signal, 5))
    robust_range = float(p95 - p5)

    # Clipping detection: repeated identical values at extremal limits
    min_val = np.min(raw_signal)
    max_val = np.max(raw_signal)
    clip_min = np.sum(np.isclose(raw_signal, min_val, atol=1e-5))
    clip_max = np.sum(np.isclose(raw_signal, max_val, atol=1e-5))
    clipping_fraction = float(clip_min + clip_max) / float(len(raw_signal))
    is_clipped = clipping_fraction > 0.01

    signal_valid = bool(is_valid_input and (not is_clipped) and nan_fraction <= 0.01 and sig_std > 1e-6)
    rhythm_valid = bool(signal_valid and peak_count >= 3)

    notes = []
    if not is_valid_input:
        notes.append("Input signal invalid")
    if is_clipped:
        notes.append(f"Signal clipping/saturation ({clipping_fraction*100:.1f}%)")
    if peak_count < 3:
        notes.append(f"Insufficient detected peaks ({peak_count} < 3)")
    if nan_fraction > 0:
        notes.append(f"Sparse NaNs ({nan_fraction*100:.2f}%)")

    quality_notes = "; ".join(notes) if notes else "Signal quality satisfactory"

    return {
        "Signal_Mean_mV": round(sig_mean, 4),
        "Signal_Std_mV": round(sig_std, 4),
        "Signal_RMS_mV": round(sig_rms, 4),
        "Peak_to_Peak_mV": round(sig_ptp, 4),
        "Robust_Amplitude_Range_mV": round(robust_range, 4),
        "NaN_Fraction": round(nan_fraction, 6),
        "Clipping_Detected": bool(is_clipped),
        "Clipping_Fraction": round(clipping_fraction, 6),
        "signal_valid": bool(signal_valid),
        "rhythm_valid": bool(rhythm_valid),
        "Quality_OK": bool(signal_valid and rhythm_valid),
        "Quality_Notes": quality_notes
    }


# ---------------------------------------------------------------------------
# Step 5: Rhythm and HRV Feature Extraction
# ---------------------------------------------------------------------------

def extract_rhythm_and_hrv(cleaned_signal, fs, provenance_dict):
    """
    Detects R-peaks using Pan-Tompkins with physiological RR interval filtering.
    Returns (rhythm_features_dict, rr_records_list, peaks_array).
    Strictly returns NaN on failure; never returns synthetic defaults.
    """
    rr_records = []
    try:
        _, peak_info = nk.ecg_peaks(cleaned_signal, sampling_rate=int(fs), method="pantompkins1985")
        peaks = peak_info.get("ECG_R_Peaks", np.array([], dtype=int))
    except Exception:
        try:
            _, peak_info = nk.ecg_peaks(cleaned_signal, sampling_rate=int(fs), method="neurokit")
            peaks = peak_info.get("ECG_R_Peaks", np.array([], dtype=int))
        except Exception:
            peaks = np.array([], dtype=int)

    detected_peaks = len(peaks)
    null_rhythm = {
        "Detected_Peaks": detected_peaks,
        "Valid_RR_Count": 0,
        "Mean_HR_BPM": np.nan,
        "Median_HR_BPM": np.nan,
        "Std_HR_BPM": np.nan,
        "Mean_RR_ms": np.nan,
        "Median_RR_ms": np.nan,
        "SDNN_ms": np.nan,
        "RMSSD_ms": np.nan,
        "RR_Min_ms": np.nan,
        "RR_Max_ms": np.nan,
        "RR_CV": np.nan,
        "RR_IQR_ms": np.nan,
        "peak_detection_ok": False,
        "peak_notes": f"Insufficient peaks detected ({detected_peaks} < 3)."
    }

    if detected_peaks < 3:
        return null_rhythm, rr_records, peaks

    raw_rr_sec = np.diff(peaks) / float(fs)
    raw_rr_ms = raw_rr_sec * 1000.0

    # Physiological plausibility filtering: 250 ms to 2500 ms (24 to 240 bpm)
    valid_rr_mask = (raw_rr_ms >= 250.0) & (raw_rr_ms <= 2500.0)

    for idx, (rr_val, is_valid) in enumerate(zip(raw_rr_ms, valid_rr_mask)):
        reason = "Valid" if is_valid else ("Tachycardia <250ms" if rr_val < 250.0 else "Bradycardia/Pause >2500ms")
        rr_records.append({
            "record_id": provenance_dict.get("record_id", np.nan),
            "patient_id": provenance_dict.get("patient_id", np.nan),
            "seizure_id": provenance_dict.get("seizure_id", np.nan),
            "phase": provenance_dict.get("phase", "unknown"),
            "rr_index": idx,
            "rr_ms": round(float(rr_val), 2),
            "rr_interval_ms": round(float(rr_val), 2),  # Backward-compatible alias
            "dataset": provenance_dict.get("dataset", "szdb"),
            "valid_rr": bool(is_valid),
            "exclusion_reason": reason
        })

    valid_rr_ms = raw_rr_ms[valid_rr_mask]
    valid_count = len(valid_rr_ms)

    if valid_count < 2:
        null_rhythm["Valid_RR_Count"] = valid_count
        null_rhythm["peak_detection_ok"] = True
        null_rhythm["peak_notes"] = f"Detected {detected_peaks} peaks, but only {valid_count} valid RR intervals."
        return null_rhythm, rr_records, peaks

    inst_hr_bpm = 60000.0 / valid_rr_ms
    mean_rr = float(np.mean(valid_rr_ms))
    median_rr = float(np.median(valid_rr_ms))
    sdnn = float(np.std(valid_rr_ms, ddof=1)) if valid_count > 1 else 0.0
    rr_cv = float(sdnn / mean_rr) if mean_rr > 0 else np.nan

    diff_rr = np.diff(valid_rr_ms)
    rmssd = float(np.sqrt(np.mean(diff_rr**2))) if len(diff_rr) > 0 else np.nan
    rr_iqr = float(np.percentile(valid_rr_ms, 75) - np.percentile(valid_rr_ms, 25))

    rhythm_feats = {
        "Detected_Peaks": detected_peaks,
        "Valid_RR_Count": valid_count,
        "Mean_HR_BPM": round(float(np.mean(inst_hr_bpm)), 2),
        "Median_HR_BPM": round(float(np.median(inst_hr_bpm)), 2),
        "Std_HR_BPM": round(float(np.std(inst_hr_bpm, ddof=1)), 2) if valid_count > 1 else 0.0,
        "Mean_RR_ms": round(mean_rr, 2),
        "Median_RR_ms": round(median_rr, 2),
        "SDNN_ms": round(sdnn, 2),
        "RMSSD_ms": round(rmssd, 2) if np.isfinite(rmssd) else np.nan,
        "RR_Min_ms": round(float(np.min(valid_rr_ms)), 2),
        "RR_Max_ms": round(float(np.max(valid_rr_ms)), 2),
        "RR_CV": round(rr_cv, 4) if np.isfinite(rr_cv) else np.nan,
        "RR_IQR_ms": round(rr_iqr, 2),
        "peak_detection_ok": True,
        "peak_notes": f"Detected {detected_peaks} peaks ({valid_count} valid RR intervals)."
    }
    return rhythm_feats, rr_records, peaks


# ---------------------------------------------------------------------------
# Step 6: Morphology Features, Delineation & Aligned Beat Templates
# ---------------------------------------------------------------------------

def extract_morphology_and_templates(cleaned_signal, peaks, fs):
    """
    Extracts P-QRS-T delineation metrics, aligns beat waveforms, and creates median templates.
    Returns (morphology_dict, median_template_array, usable_beat_count).
    Does NOT describe automated morphology values as clinically certified measurements.
    """
    morph_feats = {
        "QRS_Duration_Median_ms": np.nan,
        "QRS_Duration_IQR_ms": np.nan,
        "QRS_Amplitude_Median_mV": np.nan,
        "R_Amplitude_Median_mV": np.nan,
        "PR_Interval_Median_ms": np.nan,
        "QT_Interval_Median_ms": np.nan,
        "QTc_Median_ms": np.nan,
        "ST_Level_Median_mV": np.nan,
        "P_Amplitude_Median_mV": np.nan,
        "T_Amplitude_Median_mV": np.nan,
        "Beat_Correlation_Median": np.nan,
        "Beat_Correlation_IQR": np.nan,
        "Beat_RMSE": np.nan,
        "Morphology_Variability": np.nan,
        "morphology_valid": False,
        "morphology_notes": "Not attempted."
    }

    if len(peaks) < 3:
        morph_feats["morphology_notes"] = "Insufficient peaks for morphology extraction."
        return morph_feats, np.array([]), 0

    delineation_ok = False
    try:
        _, waves = nk.ecg_delineate(cleaned_signal, peaks, sampling_rate=int(fs), method="dwt")
        q_peaks = np.array(waves.get("ECG_Q_Peaks", []))
        s_peaks = np.array(waves.get("ECG_S_Peaks", []))
        p_peaks = np.array(waves.get("ECG_P_Peaks", []))
        t_peaks = np.array(waves.get("ECG_T_Peaks", []))
        t_offsets = np.array(waves.get("ECG_T_Offsets", []))

        # QRS duration: S_peak - Q_peak
        valid_qs = [i for i in range(min(len(q_peaks), len(s_peaks)))
                    if np.isfinite(q_peaks[i]) and np.isfinite(s_peaks[i])]
        if len(valid_qs) >= 2:
            q_idx = np.array(q_peaks[valid_qs], dtype=int)
            s_idx = np.array(s_peaks[valid_qs], dtype=int)
            qrs_dur_ms = (s_idx - q_idx) / float(fs) * 1000.0
            qrs_dur_ms = qrs_dur_ms[(qrs_dur_ms > 20.0) & (qrs_dur_ms < 300.0)]
            if len(qrs_dur_ms) >= 2:
                morph_feats["QRS_Duration_Median_ms"] = round(float(np.median(qrs_dur_ms)), 2)
                morph_feats["QRS_Duration_IQR_ms"] = round(float(np.percentile(qrs_dur_ms, 75) - np.percentile(qrs_dur_ms, 25)), 2)
                delineation_ok = True

        # R-peak and QRS amplitude
        r_amps = cleaned_signal[peaks]
        morph_feats["R_Amplitude_Median_mV"] = round(float(np.median(r_amps)), 4)
        if len(valid_qs) >= 2:
            qrs_amps = np.abs(cleaned_signal[s_idx] - cleaned_signal[q_idx])
            morph_feats["QRS_Amplitude_Median_mV"] = round(float(np.median(qrs_amps)), 4)

        # P-wave and T-wave amplitudes
        valid_p = [int(p) for p in p_peaks if np.isfinite(p) and 0 <= int(p) < len(cleaned_signal)]
        if len(valid_p) >= 2:
            morph_feats["P_Amplitude_Median_mV"] = round(float(np.median(cleaned_signal[valid_p])), 4)

        valid_t = [int(t) for t in t_peaks if np.isfinite(t) and 0 <= int(t) < len(cleaned_signal)]
        if len(valid_t) >= 2:
            morph_feats["T_Amplitude_Median_mV"] = round(float(np.median(cleaned_signal[valid_t])), 4)

        # QT interval and Bazett QTc
        valid_qt = [i for i in range(min(len(q_peaks), len(t_offsets)))
                    if np.isfinite(q_peaks[i]) and np.isfinite(t_offsets[i])]
        if len(valid_qt) >= 2:
            q_i = np.array(q_peaks[valid_qt], dtype=int)
            t_off_i = np.array(t_offsets[valid_qt], dtype=int)
            qt_ms = (t_off_i - q_i) / float(fs) * 1000.0
            qt_ms = qt_ms[(qt_ms > 150.0) & (qt_ms < 800.0)]
            if len(qt_ms) >= 2:
                median_qt = float(np.median(qt_ms))
                morph_feats["QT_Interval_Median_ms"] = round(median_qt, 2)
                rr_sec = np.median(np.diff(peaks)) / float(fs) if len(peaks) > 1 else 0.8
                if rr_sec > 0.2:
                    morph_feats["QTc_Median_ms"] = round(median_qt / np.sqrt(rr_sec), 2)

        morph_feats["morphology_valid"] = bool(delineation_ok)
        morph_feats["morphology_notes"] = "Automated delineation successful." if delineation_ok else "Partial delineation."
    except Exception as e:
        morph_feats["morphology_valid"] = False
        morph_feats["morphology_notes"] = f"Delineation failed ({type(e).__name__})."

    # Beat Template Extraction (-200 ms to +400 ms around R-peak)
    pre_samples = int(0.200 * fs)
    post_samples = int(0.400 * fs)
    beat_epochs = []
    for p in peaks:
        if p - pre_samples >= 0 and p + post_samples <= len(cleaned_signal):
            beat_epochs.append(cleaned_signal[p - pre_samples : p + post_samples])

    usable_beat_count = len(beat_epochs)
    if usable_beat_count < 3:
        return morph_feats, np.array([]), usable_beat_count

    beat_array = np.array(beat_epochs)
    median_template = np.median(beat_array, axis=0)

    # Compute beat-to-template correlation and RMSE
    correlations = []
    rmses = []
    template_norm = np.linalg.norm(median_template - np.mean(median_template))

    for beat in beat_array:
        beat_centered = beat - np.mean(beat)
        beat_norm = np.linalg.norm(beat_centered)
        if beat_norm > 1e-6 and template_norm > 1e-6:
            r = float(np.dot(beat_centered, median_template - np.mean(median_template)) / (beat_norm * template_norm))
            correlations.append(np.clip(r, -1.0, 1.0))
        rmse = float(np.sqrt(np.mean((beat - median_template)**2)))
        rmses.append(rmse)

    if len(correlations) >= 2:
        med_corr = float(np.median(correlations))
        iqr_corr = float(np.percentile(correlations, 75) - np.percentile(correlations, 25))
        morph_feats["Beat_Correlation_Median"] = round(med_corr, 4)
        morph_feats["Beat_Correlation_IQR"] = round(iqr_corr, 4)
        morph_feats["Beat_RMSE"] = round(float(np.median(rmses)), 4)
        morph_feats["Morphology_Variability"] = round(max(0.0, 1.0 - med_corr), 4)

    return morph_feats, median_template, usable_beat_count


# ---------------------------------------------------------------------------
# Step 7: Spectral Features via Welch PSD
# ---------------------------------------------------------------------------

def compute_spectral_features(signal, fs):
    """
    Computes frequency domain power distribution using Welch PSD.
    Respects Nyquist limit dynamically.
    Cautiously designates high-frequency band as an exploratory EMG/noise proxy.
    """
    nperseg = min(len(signal), int(4 * fs))  # 4-second Welch window
    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=nperseg)
    f_nyquist = fs / 2.0

    bw_idx = (freqs >= 0.0) & (freqs < 0.5)
    ecg_idx = (freqs >= 0.5) & (freqs <= min(20.0, f_nyquist))
    hf_top = min(100.0, f_nyquist)
    emg_idx = (freqs >= 20.0) & (freqs <= hf_top) if hf_top > 20.0 else np.zeros_like(freqs, dtype=bool)

    total_power = float(np.trapezoid(psd, freqs)) if len(psd) > 1 else 0.0
    bw_power = float(np.trapezoid(psd[bw_idx], freqs[bw_idx])) if np.any(bw_idx) else 0.0
    ecg_power = float(np.trapezoid(psd[ecg_idx], freqs[ecg_idx])) if np.any(ecg_idx) else 0.0
    emg_power = float(np.trapezoid(psd[emg_idx], freqs[emg_idx])) if np.any(emg_idx) else 0.0

    rel_hf = float(emg_power / total_power) if total_power > 1e-12 else 0.0
    centroid = float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 1e-12 else 0.0
    spectral_valid = bool(total_power > 1e-12 and ecg_power > 0 and np.isfinite(centroid))

    return {
        "Total_Spectral_Power": float(f"{total_power:.6e}"),
        "Baseline_Wander_Band_Power": float(f"{bw_power:.6e}"),
        "ECG_Dominant_Band_Power": float(f"{ecg_power:.6e}"),
        "EMG_Associated_High_Frequency_Power": float(f"{emg_power:.6e}"),
        "Relative_High_Frequency_Power": round(rel_hf, 6),
        "Spectral_Centroid_Hz": round(centroid, 2),
        "spectral_valid": bool(spectral_valid)
    }


# ---------------------------------------------------------------------------
# Step 2: Generic PhysioNet SZDB Record & Seizure Phase Extraction
# ---------------------------------------------------------------------------

def process_single_seizure_event(record_id, seizure_id, patient_id, onset_s, offset_s, header):
    """
    Extracts interictal, preictal, ictal, and postictal phases for a single seizure event.
    Interictal, preictal, postictal: ~60s windows.
    Ictal: preserves actual clinical seizure onset, offset, and duration.
    Streams requested sample ranges directly via WFDB.
    """
    fs = float(header.fs)
    sig_len = header.sig_len
    total_duration_s = float(sig_len) / fs

    ch_idx, ch_name, ch_units, id_notes = identify_ecg_channel(header)
    if ch_idx is None:
        return [], [], {}, {}, f"ECG channel identification failed: {id_notes}"

    # Phase boundary definitions
    # 1. Interictal: ~10 minutes prior to seizure onset
    interictal_start = max(0.0, onset_s - 660.0)
    interictal_end = max(0.0, onset_s - 600.0)

    # 2. Preictal: ~2 minutes prior to seizure onset
    preictal_start = max(0.0, onset_s - 120.0)
    preictal_end = max(0.0, onset_s - 60.0)

    # 3. Ictal: Preserves actual seizure onset, offset, and duration
    ictal_start = float(onset_s)
    ictal_end = min(total_duration_s, float(offset_s))

    # 4. Postictal: Immediate ~60s following seizure offset
    postictal_start = float(offset_s)
    postictal_end = min(total_duration_s, postictal_start + 60.0)

    phase_definitions = [
        ("Interictal_Baseline", interictal_start, interictal_end),
        ("Preictal_Prediction", preictal_start, preictal_end),
        ("Ictal_Seizure", ictal_start, ictal_end),
        ("Postictal_Recovery", postictal_start, postictal_end)
    ]

    feature_rows = []
    rr_records = []
    templates_dict = {}
    phase_signals = {}

    for phase_name, start_s, end_s in phase_definitions:
        duration_s = end_s - start_s
        start_samp = int(round(start_s * fs))
        end_samp = int(round(end_s * fs))

        provenance = {
            "dataset": "szdb",
            "data_origin": "real_clinical_physionet",
            "patient_id": patient_id,
            "record_id": record_id,
            "seizure_id": seizure_id,
            "phase": phase_name,
            "segment_start_sec": round(start_s, 2),
            "segment_end_sec": round(end_s, 2),
            "segment_duration_sec": round(duration_s, 2),
            "window_start_s": round(start_s, 2),
            "window_end_s": round(end_s, 2),
            "window_duration_s": round(duration_s, 2),
            "fs": fs,
            "original_sampling_rate_hz": fs,
            "analysis_sampling_rate_hz": fs,
            "ecg_channel": ch_name,
            "channel_name": ch_name,
            "ecg_units": ch_units,
            "units": ch_units,
            "n_samples": end_samp - start_samp,
            "processing_status": "PENDING",
            "quality_notes": ""
        }

        if start_samp >= end_samp or start_samp >= sig_len:
            provenance["processing_status"] = "FAILED"
            provenance["quality_notes"] = f"Invalid window bounds [{start_s:.1f}s, {end_s:.1f}s]"
            provenance["signal_valid"] = False
            provenance["rhythm_valid"] = False
            provenance["morphology_valid"] = False
            provenance["spectral_valid"] = False
            provenance["Quality_OK"] = False
            provenance["Quality_Notes"] = provenance["quality_notes"]
            feature_rows.append(provenance)
            continue

        try:
            chunk = wfdb.rdrecord(record_id, pn_dir="szdb", sampfrom=start_samp, sampto=end_samp)
            raw_ecg = chunk.p_signal[:, ch_idx]
        except Exception as e:
            provenance["processing_status"] = "FAILED"
            provenance["quality_notes"] = f"WFDB segment fetch error: {e}"
            provenance["signal_valid"] = False
            provenance["rhythm_valid"] = False
            provenance["morphology_valid"] = False
            provenance["spectral_valid"] = False
            provenance["Quality_OK"] = False
            provenance["Quality_Notes"] = provenance["quality_notes"]
            feature_rows.append(provenance)
            continue

        # Signal validation and cleaning
        proc_sig, nan_frac, is_valid_input, val_notes = validate_and_preprocess_signal(raw_ecg, fs)
        if not is_valid_input:
            provenance["processing_status"] = "FAILED"
            provenance["quality_notes"] = "; ".join(val_notes)
            provenance["signal_valid"] = False
            provenance["rhythm_valid"] = False
            provenance["morphology_valid"] = False
            provenance["spectral_valid"] = False
            provenance["Quality_OK"] = False
            provenance["Quality_Notes"] = provenance["quality_notes"]
            feature_rows.append(provenance)
            continue

        cleaned_sig = nk.ecg_clean(proc_sig, sampling_rate=int(fs))
        phase_signals[phase_name] = (raw_ecg, cleaned_sig, fs, start_s, end_s)

        # 1. Rhythm & HRV
        rhythm_feats, phase_rr, peaks = extract_rhythm_and_hrv(cleaned_sig, fs, provenance)
        rr_records.extend(phase_rr)

        # 2. Morphology & Templates
        morph_feats, template, n_beats = extract_morphology_and_templates(cleaned_sig, peaks, fs)
        if len(template) > 0:
            template_key = f"{record_id}_{seizure_id}_{phase_name}"
            templates_dict[template_key] = template

        # 3. Spectral Features
        spectral_feats = compute_spectral_features(cleaned_sig, fs)

        # 4. Signal Quality Indicators
        qual_feats = compute_signal_quality_metrics(raw_ecg, cleaned_sig, nan_frac, is_valid_input, len(peaks), fs)

        provenance["processing_status"] = "SUCCESS"
        provenance["quality_notes"] = qual_feats["Quality_Notes"]

        full_row = {
            **provenance,
            **rhythm_feats,
            **morph_feats,
            **spectral_feats,
            **qual_feats
        }
        feature_rows.append(full_row)
        print(f"    [{provenance['processing_status']}] {record_id} {seizure_id} {phase_name:20s}: "
              f"HR={rhythm_feats['Mean_HR_BPM']} BPM, SDNN={rhythm_feats['SDNN_ms']} ms, Beats={len(peaks)}")

    return feature_rows, rr_records, templates_dict, phase_signals, "OK"


# ---------------------------------------------------------------------------
# MIT-BIH Arrhythmia Database: Hard Negative Arrhythmia Extraction
# ---------------------------------------------------------------------------

def process_mitbih_hard_negative(record_id="203", duration_s=60.0):
    """
    Extracts real clinical hard-negative cardiac arrhythmia data from MIT-BIH Record 203.
    Incorporates cardiologist .atr beat annotations as verified evidence.
    """
    print(f"--> Processing Real MIT-BIH Hard Negative Record: {record_id}...")
    try:
        record = wfdb.rdrecord(record_id, pn_dir="mitdb", sampto=int(360 * duration_s))
        ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb", sampto=int(360 * duration_s))
    except Exception as e:
        print(f"  [ERROR] Could not load MIT-BIH {record_id}: {e}")
        return None, [], {}, None

    fs = float(record.fs)
    ch_idx, ch_name, ch_units, _ = identify_ecg_channel(record)
    raw_ecg = record.p_signal[:, ch_idx]

    beat_symbols = list(ann.symbol)
    symbol_counts = {s: beat_symbols.count(s) for s in set(beat_symbols)}
    aux_notes = [note for note in ann.aux_note if note] if ann.aux_note else []
    evidence = f"Symbols: {symbol_counts}"
    if aux_notes:
        evidence += f" | Rhythms: {aux_notes}"

    provenance = {
        "dataset": "mitdb",
        "data_origin": "real_clinical_physionet",
        "patient_id": "mitdb_patient_203",
        "record_id": record_id,
        "seizure_id": "none",
        "phase": "Hard_Negative_Arrhythmia",
        "segment_start_sec": 0.0,
        "segment_end_sec": round(duration_s, 2),
        "segment_duration_sec": round(duration_s, 2),
        "window_start_s": 0.0,
        "window_end_s": round(duration_s, 2),
        "window_duration_s": round(duration_s, 2),
        "fs": fs,
        "original_sampling_rate_hz": fs,
        "analysis_sampling_rate_hz": fs,
        "ecg_channel": ch_name,
        "channel_name": ch_name,
        "ecg_units": ch_units,
        "units": ch_units,
        "n_samples": len(raw_ecg),
        "processing_status": "SUCCESS",
        "quality_notes": "Cardiologist annotated hard-negative rhythm",
        "Annotation_Evidence": evidence
    }

    proc_sig, nan_frac, is_valid_input, _ = validate_and_preprocess_signal(raw_ecg, fs)
    if not is_valid_input:
        return None, [], {}, None

    cleaned_sig = nk.ecg_clean(proc_sig, sampling_rate=int(fs))
    rhythm_feats, rr_recs, peaks = extract_rhythm_and_hrv(cleaned_sig, fs, provenance)
    morph_feats, template, n_beats = extract_morphology_and_templates(cleaned_sig, peaks, fs)
    spectral_feats = compute_spectral_features(cleaned_sig, fs)
    qual_feats = compute_signal_quality_metrics(raw_ecg, cleaned_sig, nan_frac, is_valid_input, len(peaks), fs)

    templates_dict = {}
    if len(template) > 0:
        templates_dict[f"{record_id}_none_Hard_Negative_Arrhythmia"] = template
        templates_dict[f"{record_id}_Hard_Negative_Arrhythmia"] = template  # Alias

    full_row = {
        **provenance,
        **rhythm_feats,
        **morph_feats,
        **spectral_feats,
        **qual_feats
    }
    phase_signals = {"Hard_Negative_Arrhythmia": (raw_ecg, cleaned_sig, fs, 0.0, duration_s)}
    print(f"    [SUCCESS] Hard_Negative_Arrhythmia: HR={rhythm_feats['Mean_HR_BPM']} BPM, "
          f"SDNN={rhythm_feats['SDNN_ms']} ms, Beats={len(peaks)}\n")
    return full_row, rr_recs, templates_dict, phase_signals


# ---------------------------------------------------------------------------
# Step 8: Main Pipeline Execution & Output Contract Generation
# ---------------------------------------------------------------------------

def run_experiment_08():
    all_feature_rows = []
    all_rr_records = []
    all_beat_templates = {}
    representative_signals = {}
    dataset_summary_rows = []

    # 1. Audit SZDB Seizure Metadata
    seizure_events = fetch_or_load_szdb_seizures()
    unique_records = sorted(list(set(s["record_id"] for s in seizure_events)))
    print(f"Audited {len(seizure_events)} seizure events across {len(unique_records)} records: {unique_records}\n")

    # 2. Process each SZDB record and its seizure events
    for rec_id in unique_records:
        rec_events = [s for s in seizure_events if s["record_id"] == rec_id]
        print(f"--> Processing PhysioNet SZDB Record: {rec_id} ({len(rec_events)} seizure event(s))...")

        try:
            header = wfdb.rdheader(rec_id, pn_dir="szdb")
            ch_idx, ch_name, ch_units, ch_notes = identify_ecg_channel(header)
            rec_fs = float(header.fs)
            dur_s = float(header.sig_len) / rec_fs
            print(f"    Channel: {ch_name} (ch #{ch_idx}), fs: {rec_fs} Hz, Total duration: {dur_s:.1f}s ({dur_s/3600:.2f}h), Units: {ch_units}")
        except Exception as e:
            print(f"  [ERROR] Failed to load header for {rec_id}: {e}")
            dataset_summary_rows.append({
                "record_id": rec_id,
                "patient_id": np.nan,
                "number_of_seizures": len(rec_events),
                "fs": np.nan,
                "ecg_channel": np.nan,
                "ecg_units": np.nan,
                "usable_phases": 0,
                "failed_phases": len(rec_events) * 4,
                "processing_notes": f"Header read failed: {e}"
            })
            continue

        usable_phases_count = 0
        failed_phases_count = 0

        for s_event in rec_events:
            sz_id = s_event["seizure_id"]
            on_s = s_event["onset_s"]
            off_s = s_event["offset_s"]

            f_rows, r_recs, t_dict, p_sigs, status_msg = process_single_seizure_event(
                record_id=rec_id,
                seizure_id=sz_id,
                patient_id=np.nan,  # Explicitly unknown per PhysioNet specification
                onset_s=on_s,
                offset_s=off_s,
                header=header
            )

            all_feature_rows.extend(f_rows)
            all_rr_records.extend(r_recs)
            all_beat_templates.update(t_dict)

            # Keep representative signals for waveform plot
            if p_sigs and (rec_id == "sz01" or len(representative_signals) < 4):
                for p_name, sig_tuple in p_sigs.items():
                    if p_name not in representative_signals:
                        representative_signals[p_name] = (rec_id, sz_id, *sig_tuple)

            for row in f_rows:
                if row.get("processing_status") == "SUCCESS" and row.get("Quality_OK", False):
                    usable_phases_count += 1
                else:
                    failed_phases_count += 1

        dataset_summary_rows.append({
            "record_id": rec_id,
            "patient_id": np.nan,
            "number_of_seizures": len(rec_events),
            "fs": rec_fs,
            "ecg_channel": ch_name,
            "ecg_units": ch_units,
            "usable_phases": usable_phases_count,
            "failed_phases": failed_phases_count,
            "processing_notes": f"Processed {len(rec_events)} seizure event(s) successfully."
        })

    # 3. Process MIT-BIH Hard Negative Arrhythmia Control
    hn_row, hn_rr, hn_templates, hn_sigs = process_mitbih_hard_negative(record_id="203", duration_s=60.0)
    if hn_row:
        all_feature_rows.append(hn_row)
        all_rr_records.extend(hn_rr)
        all_beat_templates.update(hn_templates)
        if hn_sigs:
            representative_signals["Hard_Negative_Arrhythmia"] = ("203", "none", *hn_sigs["Hard_Negative_Arrhythmia"])

    if not all_feature_rows:
        raise RuntimeError("FATAL: No real ECG data could be processed. Terminating without synthetic fallback.")

    # -----------------------------------------------------------------------
    # Save Artifacts
    # -----------------------------------------------------------------------

    # 1. Primary Empirical Feature Table
    df_features = pd.DataFrame(all_feature_rows)
    feature_table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df_features.to_csv(feature_table_path, index=False)
    print(f"\n[Artifact 1] Primary feature table: {feature_table_path} ({len(df_features)} rows)")

    # 2. Long-Form RR Intervals
    df_rr = pd.DataFrame(all_rr_records)
    rr_path = os.path.join(OUTPUT_DIR, "08_rr_intervals.csv")
    df_rr.to_csv(rr_path, index=False)
    print(f"[Artifact 2] Long-form RR intervals: {rr_path} ({len(df_rr)} intervals)")

    # 3. Aligned Beat Templates
    templates_path = os.path.join(OUTPUT_DIR, "08_beat_templates.npz")
    np.savez_compressed(templates_path, **all_beat_templates)
    print(f"[Artifact 3] Aligned beat templates: {templates_path} ({len(all_beat_templates)} templates)")

    # 4. Data Quality Report
    quality_cols = [
        "dataset", "record_id", "patient_id", "seizure_id", "phase", "ecg_channel",
        "fs", "segment_duration_sec", "Detected_Peaks", "Valid_RR_Count",
        "NaN_Fraction", "Clipping_Detected", "signal_valid", "rhythm_valid",
        "morphology_valid", "spectral_valid", "processing_status", "quality_notes"
    ]
    avail_q = [c for c in quality_cols if c in df_features.columns]
    df_quality = df_features[avail_q]
    quality_path = os.path.join(OUTPUT_DIR, "08_data_quality_report.csv")
    df_quality.to_csv(quality_path, index=False)
    print(f"[Artifact 4] Data quality report: {quality_path}")

    # 5. Dataset Summary (SZDB records sz01-sz07)
    df_summary = pd.DataFrame(dataset_summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "08_dataset_summary.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"[Artifact 5] Dataset summary: {summary_path} ({len(df_summary)} records)")

    # 6. Backward-Compatible Aggregate Table
    compat_rows = []
    classes_order = ["Interictal_Baseline", "Preictal_Prediction", "Ictal_Seizure", "Postictal_Recovery", "Hard_Negative_Arrhythmia"]
    for c_name in classes_order:
        sub = df_features[df_features["phase"] == c_name]
        if len(sub) > 0:
            mean_hr = float(sub["Mean_HR_BPM"].dropna().mean()) if len(sub["Mean_HR_BPM"].dropna()) > 0 else np.nan
            sdnn = float(sub["SDNN_ms"].dropna().mean()) if len(sub["SDNN_ms"].dropna()) > 0 else np.nan
            rmssd = float(sub["RMSSD_ms"].dropna().mean()) if len(sub["RMSSD_ms"].dropna()) > 0 else np.nan
            hf_power = float(sub["EMG_Associated_High_Frequency_Power"].dropna().mean()) if len(sub["EMG_Associated_High_Frequency_Power"].dropna()) > 0 else np.nan
            sig_std = float(sub["Signal_Std_mV"].dropna().mean()) if len(sub["Signal_Std_mV"].dropna()) > 0 else np.nan
            tot_peaks = int(sub["Detected_Peaks"].dropna().sum()) if len(sub["Detected_Peaks"].dropna()) > 0 else 0
            compat_rows.append({
                "Clinical_Class": c_name,
                "Mean_HR_BPM": round(mean_hr, 2) if np.isfinite(mean_hr) else np.nan,
                "SDNN_ms": round(sdnn, 2) if np.isfinite(sdnn) else np.nan,
                "RMSSD_ms": round(rmssd, 2) if np.isfinite(rmssd) else np.nan,
                "EMG_Band_Power": round(hf_power, 6) if np.isfinite(hf_power) else np.nan,
                "Signal_Std": round(sig_std, 4) if np.isfinite(sig_std) else np.nan,
                "Detected_Peaks": tot_peaks
            })
    df_compat = pd.DataFrame(compat_rows)
    compat_path = os.path.join(OUTPUT_DIR, "08_empirical_seizure_parameters.csv")
    df_compat.to_csv(compat_path, index=False)
    print(f"[Artifact 6] Aggregate parameters: {compat_path} ({len(df_compat)} rows)")

    # -----------------------------------------------------------------------
    # Figure 1: Representative Real Waveforms Across Phases
    # -----------------------------------------------------------------------
    plot_phases = ["Interictal_Baseline", "Preictal_Prediction", "Ictal_Seizure", "Postictal_Recovery", "Hard_Negative_Arrhythmia"]
    active_plots = [p for p in plot_phases if p in representative_signals]

    fig, axes = plt.subplots(len(active_plots), 1, figsize=(13, 11), sharex=False)
    if len(active_plots) == 1:
        axes = [axes]
    fig.suptitle("Experiment 08: Real Seizure ECG & Hard-Negative Waveforms (PhysioNet SZDB & MIT-BIH)\n"
                 "Empirical Waveform Inspection Across Full Dataset (Zero Synthetic Fallback)",
                 fontsize=11, fontweight="bold", y=0.995)

    colors = ["steelblue", "darkorange", "purple", "seagreen", "crimson"]
    for idx, p_name in enumerate(active_plots):
        rec_id, sz_id, raw_sig, clean_sig, p_fs, t_start, t_end = representative_signals[p_name]
        ax = axes[idx]
        t_axis = np.arange(len(raw_sig)) / float(p_fs)
        ax.plot(t_axis, raw_sig, color=colors[idx % len(colors)], linewidth=0.7, alpha=0.9)
        src_label = f"MIT-BIH {rec_id}" if "Hard_Negative" in p_name else f"SZDB {rec_id} ({sz_id})"
        ax.set_title(f"{src_label} | {p_name.replace('_', ' ')} (Time: {t_start:.0f}s - {t_end:.0f}s, fs: {p_fs:.0f} Hz, dur: {t_end-t_start:.1f}s)",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Amplitude (mV)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time within Phase Window (seconds)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig1_path = os.path.join(OUTPUT_DIR, "08_real_seizure_phases.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifact 7] Waveform figure: {fig1_path}")

    # -----------------------------------------------------------------------
    # Figure 2: Empirical Feature Distributions across Full Dataset
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Experiment 08: Empirical Feature Distributions Across Real Seizure Phases\n"
                 "Full PhysioNet SZDB Dataset (sz01-sz07, 10 Seizure Events) & MIT-BIH 203 Control",
                 fontsize=11, fontweight="bold")

    phases_list = [p for p in classes_order if p in df_features["phase"].values]
    x_pos = np.arange(len(phases_list))
    labels = [p.replace("_", "\n") for p in phases_list]

    # Panel 1: Heart Rate
    hr_vals = [df_features[df_features["phase"] == p]["Mean_HR_BPM"].dropna().values for p in phases_list]
    axes[0, 0].boxplot(hr_vals, positions=x_pos, tick_labels=labels, patch_artist=True,
                       boxprops=dict(facecolor="lightblue"))
    axes[0, 0].set_title("Mean Heart Rate (BPM)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("BPM")
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)

    # Panel 2: SDNN
    sdnn_vals = [df_features[df_features["phase"] == p]["SDNN_ms"].dropna().values for p in phases_list]
    axes[0, 1].boxplot(sdnn_vals, positions=x_pos, tick_labels=labels, patch_artist=True,
                       boxprops=dict(facecolor="lightgreen"))
    axes[0, 1].set_title("Heart Rate Variability (SDNN, ms)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)

    # Panel 3: High-Frequency Spectral Power Proxy
    hf_vals = [df_features[df_features["phase"] == p]["EMG_Associated_High_Frequency_Power"].dropna().values for p in phases_list]
    axes[1, 0].boxplot(hf_vals, positions=x_pos, tick_labels=labels, patch_artist=True,
                       boxprops=dict(facecolor="lightsalmon"))
    axes[1, 0].set_title("High-Frequency Spectral Power Density (Muscle Tremor Proxy)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("mV^2 / Hz")
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # Panel 4: Morphology Variability
    morph_vals = [df_features[df_features["phase"] == p]["Morphology_Variability"].dropna().values for p in phases_list]
    valid_morph = [v for v in morph_vals if len(v) > 0]
    if valid_morph:
        axes[1, 1].boxplot(morph_vals, positions=x_pos, tick_labels=labels, patch_artist=True,
                           boxprops=dict(facecolor="plum"))
    axes[1, 1].set_title("Morphology Variability (1 - Median Beat Correlation)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Variability Index [0 - 1]")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(OUTPUT_DIR, "08_real_feature_distributions.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifact 8] Feature distribution figure: {fig2_path}")

    print("\n=== Experiment 08 Complete: Full SZDB Characterization Successfully Executed ===")


if __name__ == "__main__":
    run_experiment_08()
