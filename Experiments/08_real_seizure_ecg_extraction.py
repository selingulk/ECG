"""
Experiment 08: Real Epileptic Seizure & Cardiac Arrhythmia Empirical Characterization Pipeline
=============================================================================================
Objective:
  Extract empirical physiological parameters (rhythm/HRV, morphology, beat templates,
  spectral proxies, and signal quality) across distinct clinical seizure phases and real
  cardiac arrhythmia recordings from PhysioNet without synthetic fallback.

Datasets:
  - PhysioNet szdb: Post-Ictal Heart Rate Oscillations in Partial Epilepsy (records: sz01, sz04)
  - PhysioNet mitdb: MIT-BIH Arrhythmia Database (record 203 with cardiologist .atr annotations)

Outputs:
  - outputs/08_empirical_ecg_feature_table.csv (Primary feature table with complete provenance)
  - outputs/08_rr_intervals.csv (Long-form beat-to-beat RR interval data)
  - outputs/08_beat_templates.npz (Aligned median beat templates across phases)
  - outputs/08_data_quality_report.csv (Quality validation and audit diagnostics)
  - outputs/08_real_seizure_phases.png (Waveform inspection figure)
  - outputs/08_real_feature_distributions.png (Empirical feature distribution figure)
  - outputs/08_empirical_seizure_parameters.csv (Backward-compatible aggregate table)

Scientific Disclaimer:
  This script performs exploratory empirical characterization on a limited sample of real
  clinical recordings. No clinical validation or diagnostic performance is claimed.
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

# Guarantee outputs directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=========================================================================")
print("  EXPERIMENT 08: REAL SEIZURE & ARRHYTHMIA ECG CHARACTERIZATION PIPELINE")
print("=========================================================================\n")


# ---------------------------------------------------------------------------
# Stage 1 & 2: Channel Identification, Input Validation & Signal Cleaning
# ---------------------------------------------------------------------------

def identify_ecg_channel(record):
    """
    Identify the most appropriate ECG channel from a WFDB Record.
    Returns (channel_index, channel_name, units).
    """
    sig_names = [name.upper() for name in record.sig_name]
    preferred_keywords = ["ECG", "EKG", "MLII", "V1", "V2", "V3", "V4", "V5", "V6", "II", "I"]

    for kw in preferred_keywords:
        for idx, name in enumerate(sig_names):
            if kw in name:
                units = record.units[idx] if idx < len(record.units) else "mV"
                return idx, record.sig_name[idx], units

    # Fallback to checking units
    for idx, u in enumerate(record.units):
        if u.lower() in ["mv", "uv"]:
            return idx, record.sig_name[idx], u

    # Default to channel 0 with explicit documentation
    ch_name = record.sig_name[0] if len(record.sig_name) > 0 else "channel_0"
    ch_units = record.units[0] if len(record.units) > 0 else "mV"
    return 0, ch_name, ch_units


def validate_and_preprocess_signal(raw_signal, fs, max_nan_fraction=0.01):
    """
    Validates input signal for emptiness, constants, and NaNs.
    Applies minimal linear interpolation for sparse NaNs if fraction <= max_nan_fraction.
    Returns (processed_signal, nan_fraction, is_valid, diagnostic_notes).
    """
    notes = []
    if raw_signal is None or len(raw_signal) == 0:
        return None, 1.0, False, "Empty or None signal."

    n_samples = len(raw_signal)
    nan_mask = ~np.isfinite(raw_signal)
    nan_count = int(np.sum(nan_mask))
    nan_fraction = float(nan_count) / float(n_samples)

    if nan_fraction > max_nan_fraction:
        return None, nan_fraction, False, f"Excessive NaN/Inf fraction ({nan_fraction:.4f} > {max_nan_fraction})."

    signal_work = raw_signal.astype(np.float64).copy()
    if nan_count > 0:
        indices = np.arange(n_samples)
        valid_indices = indices[~nan_mask]
        signal_work[nan_mask] = np.interp(indices[nan_mask], valid_indices, signal_work[~nan_mask])
        notes.append(f"Interpolated {nan_count} sparse NaN samples ({nan_fraction*100:.2f}%).")

    std_val = float(np.std(signal_work))
    if std_val < 1e-6:
        return signal_work, nan_fraction, False, "Flatline or constant signal (std < 1e-6)."

    return signal_work, nan_fraction, True, "; ".join(notes) if notes else "Input validated successfully."


# ---------------------------------------------------------------------------
# Stage 3: Rhythm and HRV Feature Extraction
# ---------------------------------------------------------------------------

def extract_rhythm_and_hrv(cleaned_signal, fs, provenance_dict):
    """
    Detects R-peaks and computes physiological RR/HRV parameters.
    Saves long-form RR interval entries.
    Never returns fake constants (75 bpm, 30 ms, 25 ms) on failure; returns NaN.
    """
    rr_records = []
    try:
        _, peak_info = nk.ecg_peaks(cleaned_signal, sampling_rate=fs, method="pantompkins1985")
        peaks = peak_info.get("ECG_R_Peaks", np.array([], dtype=int))
    except Exception as e:
        try:
            _, peak_info = nk.ecg_peaks(cleaned_signal, sampling_rate=fs, method="neurokit")
            peaks = peak_info.get("ECG_R_Peaks", np.array([], dtype=int))
        except Exception as e2:
            peaks = np.array([], dtype=int)

    detected_peaks = len(peaks)

    if detected_peaks < 3:
        # Insufficient peaks for meaningful rhythm analysis
        rhythm_feats = {
            "Detected_Peaks": detected_peaks,
            "Valid_RR_Count": 0,
            "Mean_HR_BPM": np.nan,
            "Median_HR_BPM": np.nan,
            "Std_HR_BPM": np.nan,
            "Mean_RR_ms": np.nan,
            "Median_RR_ms": np.nan,
            "SDNN_ms": np.nan,
            "RMSSD_ms": np.nan,
            "RR_IQR_ms": np.nan,
            "RR_Min_ms": np.nan,
            "RR_Max_ms": np.nan,
            "peak_detection_ok": False,
            "peak_notes": f"Insufficient peaks detected ({detected_peaks} < 3)."
        }
        return rhythm_feats, rr_records, peaks

    # Calculate RR intervals
    raw_rr_sec = np.diff(peaks) / float(fs)
    raw_rr_ms = raw_rr_sec * 1000.0

    # Physiological plausibility filtering (250 ms to 2500 ms corresponds to 24 to 240 bpm)
    valid_rr_mask = (raw_rr_ms >= 250.0) & (raw_rr_ms <= 2500.0)

    for idx, (rr_ms, is_valid) in enumerate(zip(raw_rr_ms, valid_rr_mask)):
        reason = "Valid" if is_valid else ("Tachycardia <250ms" if rr_ms < 250.0 else "Bradycardia/Pause >2500ms")
        rr_records.append({
            "dataset": provenance_dict.get("dataset", "unknown"),
            "record_id": provenance_dict.get("record_id", "unknown"),
            "seizure_id": provenance_dict.get("seizure_id", "none"),
            "phase": provenance_dict.get("phase", "unknown"),
            "rr_index": idx,
            "rr_interval_ms": round(float(rr_ms), 2),
            "valid_rr": bool(is_valid),
            "exclusion_reason": reason
        })

    valid_rr_ms = raw_rr_ms[valid_rr_mask]
    valid_count = len(valid_rr_ms)

    if valid_count < 2:
        rhythm_feats = {
            "Detected_Peaks": detected_peaks,
            "Valid_RR_Count": valid_count,
            "Mean_HR_BPM": np.nan,
            "Median_HR_BPM": np.nan,
            "Std_HR_BPM": np.nan,
            "Mean_RR_ms": np.nan,
            "Median_RR_ms": np.nan,
            "SDNN_ms": np.nan,
            "RMSSD_ms": np.nan,
            "RR_IQR_ms": np.nan,
            "RR_Min_ms": np.nan,
            "RR_Max_ms": np.nan,
            "peak_detection_ok": True,
            "peak_notes": f"Detected {detected_peaks} peaks, but only {valid_count} valid RR intervals after filtering."
        }
        return rhythm_feats, rr_records, peaks

    # Robust computation of HRV metrics
    inst_hr_bpm = 60000.0 / valid_rr_ms
    mean_rr = float(np.mean(valid_rr_ms))
    median_rr = float(np.median(valid_rr_ms))
    sdnn = float(np.std(valid_rr_ms, ddof=1)) if valid_count > 1 else 0.0

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
        "RR_IQR_ms": round(rr_iqr, 2),
        "RR_Min_ms": round(float(np.min(valid_rr_ms)), 2),
        "RR_Max_ms": round(float(np.max(valid_rr_ms)), 2),
        "peak_detection_ok": True,
        "peak_notes": f"Successfully detected {detected_peaks} peaks ({valid_count} valid RR intervals)."
    }
    return rhythm_feats, rr_records, peaks


# ---------------------------------------------------------------------------
# Stage 4 & 5: ECG Morphology, Delineation & Beat Templates
# ---------------------------------------------------------------------------

def extract_morphology_and_templates(cleaned_signal, peaks, fs):
    """
    Extracts P-QRS-T delineation metrics and builds median beat template.
    Returns (morphology_features_dict, median_template, usable_beat_count).
    """
    # Initialize all morphology fields to NaN
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
        morph_feats["morphology_notes"] = "Insufficient peaks for morphology analysis."
        return morph_feats, np.array([]), 0

    # Delineation via NeuroKit2
    delineation_success = False
    try:
        _, waves = nk.ecg_delineate(cleaned_signal, peaks, sampling_rate=fs, method="dwt")
        q_peaks = np.array(waves.get("ECG_Q_Peaks", []))
        s_peaks = np.array(waves.get("ECG_S_Peaks", []))
        p_peaks = np.array(waves.get("ECG_P_Peaks", []))
        t_peaks = np.array(waves.get("ECG_T_Peaks", []))
        t_offsets = np.array(waves.get("ECG_T_Offsets", []))

        # Check if Q and S peaks are available for QRS duration
        valid_qs = [i for i in range(min(len(q_peaks), len(s_peaks))) if np.isfinite(q_peaks[i]) and np.isfinite(s_peaks[i])]
        if len(valid_qs) >= 2:
            q_idx = np.array(q_peaks[valid_qs], dtype=int)
            s_idx = np.array(s_peaks[valid_qs], dtype=int)
            qrs_dur_ms = (s_idx - q_idx) / float(fs) * 1000.0
            qrs_dur_ms = qrs_dur_ms[(qrs_dur_ms > 20.0) & (qrs_dur_ms < 300.0)]
            if len(qrs_dur_ms) >= 2:
                morph_feats["QRS_Duration_Median_ms"] = round(float(np.median(qrs_dur_ms)), 2)
                morph_feats["QRS_Duration_IQR_ms"] = round(float(np.percentile(qrs_dur_ms, 75) - np.percentile(qrs_dur_ms, 25)), 2)
                delineation_success = True

        # R-peak amplitude
        r_amps = cleaned_signal[peaks]
        morph_feats["R_Amplitude_Median_mV"] = round(float(np.median(r_amps)), 4)

        # P-wave and T-wave amplitudes
        valid_p = [int(p) for p in p_peaks if np.isfinite(p) and 0 <= int(p) < len(cleaned_signal)]
        if len(valid_p) >= 2:
            morph_feats["P_Amplitude_Median_mV"] = round(float(np.median(cleaned_signal[valid_p])), 4)

        valid_t = [int(t) for t in t_peaks if np.isfinite(t) and 0 <= int(t) < len(cleaned_signal)]
        if len(valid_t) >= 2:
            morph_feats["T_Amplitude_Median_mV"] = round(float(np.median(cleaned_signal[valid_t])), 4)

        # QT interval and QTc
        valid_qt = [i for i in range(min(len(q_peaks), len(t_offsets))) if np.isfinite(q_peaks[i]) and np.isfinite(t_offsets[i])]
        if len(valid_qt) >= 2:
            q_i = np.array(q_peaks[valid_qt], dtype=int)
            t_off_i = np.array(t_offsets[valid_qt], dtype=int)
            qt_ms = (t_off_i - q_i) / float(fs) * 1000.0
            qt_ms = qt_ms[(qt_ms > 150.0) & (qt_ms < 800.0)]
            if len(qt_ms) >= 2:
                median_qt = float(np.median(qt_ms))
                morph_feats["QT_Interval_Median_ms"] = round(median_qt, 2)
                # Bazett formula QTc = QT / sqrt(RR in seconds)
                rr_sec = np.median(np.diff(peaks)) / float(fs) if len(peaks) > 1 else 0.8
                if rr_sec > 0.2:
                    morph_feats["QTc_Median_ms"] = round(median_qt / np.sqrt(rr_sec), 2)

        morph_feats["morphology_valid"] = delineation_success
        morph_feats["morphology_notes"] = "Delineation successful." if delineation_success else "Partial delineation (noisy signal)."
    except Exception as e:
        morph_feats["morphology_valid"] = False
        morph_feats["morphology_notes"] = f"Delineation failed or unachievable ({type(e).__name__})."

    # Beat Template Extraction
    # Window: -200 ms to +400 ms (600 ms total)
    pre_samples = int(0.200 * fs)
    post_samples = int(0.400 * fs)
    epoch_len = pre_samples + post_samples

    beat_epochs = []
    for p in peaks:
        if p - pre_samples >= 0 and p + post_samples <= len(cleaned_signal):
            epoch = cleaned_signal[p - pre_samples : p + post_samples]
            beat_epochs.append(epoch)

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
        # Morphology variability defined as (1.0 - median_correlation)
        morph_feats["Morphology_Variability"] = round(max(0.0, 1.0 - med_corr), 4)

    return morph_feats, median_template, usable_beat_count


# ---------------------------------------------------------------------------
# Stage 6: Spectral & Frequency Band Characterization (Welch PSD)
# ---------------------------------------------------------------------------

def compute_spectral_features(signal, fs):
    """
    Computes frequency domain power distribution using Welch PSD.
    Dynamically respects Nyquist frequency.
    Uses cautious naming: EMG_Associated_High_Frequency_Power (proxy for muscle noise).
    """
    nperseg = min(len(signal), int(4 * fs)) # 4-second Welch window
    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=nperseg)
    f_nyquist = fs / 2.0

    # Frequency bands
    bw_idx = (freqs >= 0.0) & (freqs < 0.5)               # Baseline wander band
    ecg_idx = (freqs >= 0.5) & (freqs <= min(20.0, f_nyquist)) # Dominant ECG QRS/T band

    # High frequency proxy band (20 to 100 Hz or Nyquist)
    hf_top = min(100.0, f_nyquist)
    emg_idx = (freqs >= 20.0) & (freqs <= hf_top) if hf_top > 20.0 else np.zeros_like(freqs, dtype=bool)

    total_power = float(np.trapezoid(psd, freqs)) if len(psd) > 1 else 0.0
    bw_power = float(np.trapezoid(psd[bw_idx], freqs[bw_idx])) if np.any(bw_idx) else 0.0
    ecg_power = float(np.trapezoid(psd[ecg_idx], freqs[ecg_idx])) if np.any(ecg_idx) else 0.0
    emg_power = float(np.trapezoid(psd[emg_idx], freqs[emg_idx])) if np.any(emg_idx) else 0.0

    rel_hf_power = float(emg_power / total_power) if total_power > 1e-12 else 0.0
    spectral_centroid = float(np.sum(freqs * psd) / np.sum(psd)) if np.sum(psd) > 1e-12 else 0.0

    return {
        "Total_Spectral_Power": float(f"{total_power:.6e}"),
        "Baseline_Wander_Band_Power": float(f"{bw_power:.6e}"),
        "ECG_Dominant_Band_Power": float(f"{ecg_power:.6e}"),
        "EMG_Associated_High_Frequency_Power": float(f"{emg_power:.6e}"),
        "Relative_High_Frequency_Power": round(rel_hf_power, 6),
        "Spectral_Centroid_Hz": round(spectral_centroid, 2)
    }


# ---------------------------------------------------------------------------
# Stage 7: Signal Amplitude and Quality Features
# ---------------------------------------------------------------------------

def compute_quality_and_amplitude(raw_signal, cleaned_signal, nan_fraction, is_valid_input, peak_count, fs):
    """
    Computes amplitude properties and diagnostic data quality metrics.
    """
    sig_mean = float(np.mean(raw_signal))
    sig_std = float(np.std(raw_signal))
    sig_rms = float(np.sqrt(np.mean(raw_signal**2)))
    sig_ptp = float(np.ptp(raw_signal))
    p95 = float(np.percentile(raw_signal, 95))
    p5 = float(np.percentile(raw_signal, 5))
    robust_range = p95 - p5

    # Detect saturation/clipping (repeated identical values at extreme percentiles)
    min_val = np.min(raw_signal)
    max_val = np.max(raw_signal)
    clip_min = np.sum(np.isclose(raw_signal, min_val, atol=1e-5))
    clip_max = np.sum(np.isclose(raw_signal, max_val, atol=1e-5))
    clipping_fraction = float(clip_min + clip_max) / float(len(raw_signal))
    is_clipped = clipping_fraction > 0.01

    quality_ok = bool(is_valid_input and peak_count >= 3 and nan_fraction < 0.01 and not is_clipped)
    quality_notes = []
    if not is_valid_input:
        quality_notes.append("Failed input validation")
    if peak_count < 3:
        quality_notes.append(f"Low peak count ({peak_count})")
    if is_clipped:
        quality_notes.append(f"Possible signal clipping/saturation ({clipping_fraction*100:.1f}%)")
    if nan_fraction > 0:
        quality_notes.append(f"Interpolated NaNs ({nan_fraction*100:.2f}%)")

    return {
        "Signal_Mean_mV": round(sig_mean, 4),
        "Signal_Std_mV": round(sig_std, 4),
        "Signal_RMS_mV": round(sig_rms, 4),
        "Peak_to_Peak_mV": round(sig_ptp, 4),
        "Robust_Amplitude_Range_mV": round(robust_range, 4),
        "NaN_Fraction": round(nan_fraction, 6),
        "Clipping_Detected": bool(is_clipped),
        "Clipping_Fraction": round(clipping_fraction, 6),
        "Quality_OK": bool(quality_ok),
        "Quality_Notes": "; ".join(quality_notes) if quality_notes else "Quality satisfactory"
    }


# ---------------------------------------------------------------------------
# Core Pipeline: Generic Record Phase Extraction
# ---------------------------------------------------------------------------

def process_record_phases(dataset_name, record_id, seizure_id, patient_id, onset_s, offset_s, pn_dir):
    """
    Generic record processor for PhysioNet recordings.
    Extracts interictal, preictal, ictal, and postictal phases with generic ~60s windows.
    """
    print(f"--> Processing {dataset_name} Record: {record_id} (Seizure: {seizure_id})...")
    try:
        record = wfdb.rdrecord(record_id, pn_dir=pn_dir)
    except Exception as e:
        print(f"  [ERROR] Failed to load {record_id} from {pn_dir}: {e}")
        return [], [], {}, None

    fs = float(record.fs)
    sig_len = record.sig_len
    total_duration_s = sig_len / fs
    ch_idx, ch_name, ch_units = identify_ecg_channel(record)
    raw_ecg_full = record.p_signal[:, ch_idx]

    print(f"    Channel: {ch_name} (ch #{ch_idx}), fs: {fs} Hz, Total duration: {total_duration_s:.1f}s, Units: {ch_units}")

    # Generic phase window definitions
    # Target ~60s duration bounded by signal length
    # Interictal: ~10 minutes prior to seizure onset
    interictal_start = max(0.0, onset_s - 660.0)
    interictal_end = min(total_duration_s, interictal_start + 60.0)

    # Preictal: ~2 minutes prior to seizure onset
    preictal_start = max(0.0, onset_s - 120.0)
    preictal_end = min(onset_s, preictal_start + 60.0)

    # Ictal: From seizure onset up to 60s or seizure offset
    ictal_start = float(onset_s)
    ictal_end = min(float(offset_s), min(total_duration_s, ictal_start + 60.0))

    # Postictal: Immediate 60s following seizure offset
    postictal_start = float(offset_s)
    postictal_end = min(total_duration_s, postictal_start + 60.0)

    phase_definitions = [
        ("Interictal_Baseline", interictal_start, interictal_end),
        ("Preictal_Prediction", preictal_start, preictal_end),
        ("Ictal_Seizure", ictal_start, ictal_end),
        ("Postictal_Recovery", postictal_start, postictal_end)
    ]

    feature_rows = []
    all_rr_records = []
    templates_dict = {}
    phase_signals = {}

    for phase_name, start_s, end_s in phase_definitions:
        start_samp = int(start_s * fs)
        end_samp = int(end_s * fs)
        phase_raw = raw_ecg_full[start_samp:end_samp]
        actual_dur_s = len(phase_raw) / fs

        provenance = {
            "dataset": dataset_name,
            "data_origin": "real_clinical_physionet",
            "patient_id": patient_id,
            "record_id": record_id,
            "seizure_id": seizure_id,
            "phase": phase_name,
            "channel_name": ch_name,
            "units": ch_units,
            "original_sampling_rate_hz": fs,
            "analysis_sampling_rate_hz": fs,
            "window_start_s": round(start_s, 2),
            "window_end_s": round(end_s, 2),
            "window_duration_s": round(actual_dur_s, 2),
            "n_samples": len(phase_raw)
        }

        # Validate input
        processed_sig, nan_frac, is_valid, val_notes = validate_and_preprocess_signal(phase_raw, fs)

        if not is_valid:
            print(f"    [WARN] Phase {phase_name} invalid: {val_notes}")
            row = {**provenance}
            row["Quality_OK"] = False
            row["Quality_Notes"] = val_notes
            feature_rows.append(row)
            continue

        # Clean ECG using NeuroKit2
        cleaned_sig = nk.ecg_clean(processed_sig, sampling_rate=int(fs))
        phase_signals[phase_name] = (phase_raw, cleaned_sig, fs, start_s, end_s)

        # 1. Rhythm & HRV
        rhythm_feats, rr_recs, peaks = extract_rhythm_and_hrv(cleaned_sig, fs, provenance)
        all_rr_records.extend(rr_recs)

        # 2. Morphology & Beat Templates
        morph_feats, template, n_beats = extract_morphology_and_templates(cleaned_sig, peaks, fs)
        if len(template) > 0:
            template_key = f"{record_id}_{phase_name}"
            templates_dict[template_key] = template

        # 3. Spectral Features
        spectral_feats = compute_spectral_features(cleaned_sig, fs)

        # 4. Amplitude and Quality Features
        qual_feats = compute_quality_and_amplitude(phase_raw, cleaned_sig, nan_frac, is_valid, len(peaks), fs)

        # Merge all into comprehensive feature dictionary
        full_row = {
            **provenance,
            **rhythm_feats,
            **morph_feats,
            **spectral_feats,
            **qual_feats
        }
        feature_rows.append(full_row)
        print(f"    [OK] {phase_name:20s}: HR={rhythm_feats['Mean_HR_BPM']} BPM, SDNN={rhythm_feats['SDNN_ms']} ms, Beats={len(peaks)}")

    return feature_rows, all_rr_records, templates_dict, phase_signals


# ---------------------------------------------------------------------------
# Stage 8: Real MIT-BIH Hard Negative Arrhythmia Extraction
# ---------------------------------------------------------------------------

def process_mitbih_hard_negative(record_id="203", duration_s=60.0):
    """
    Extracts real hard-negative cardiac arrhythmia data from MIT-BIH Arrhythmia Database.
    Inspects cardiologist .atr annotations in the selected window.
    Never uses synthetic signals.
    """
    print(f"--> Processing Real MIT-BIH Hard Negative Record: {record_id}...")
    try:
        record = wfdb.rdrecord(record_id, pn_dir="mitdb", sampto=int(360 * duration_s))
        ann = wfdb.rdann(record_id, "atr", pn_dir="mitdb", sampto=int(360 * duration_s))
    except Exception as e:
        print(f"  [ERROR] Could not load MIT-BIH {record_id}: {e}")
        return None, [], {}, None

    fs = float(record.fs)
    ch_idx, ch_name, ch_units = identify_ecg_channel(record)
    raw_ecg = record.p_signal[:, ch_idx]

    # Audit annotations in this exact window
    beat_symbols = list(ann.symbol)
    symbol_counts = {s: beat_symbols.count(s) for s in set(beat_symbols)}
    aux_notes = [note for note in ann.aux_note if note] if ann.aux_note else []

    # Evidence notes
    evidence = f"Symbols: {symbol_counts}"
    if aux_notes:
        evidence += f" | Rhythms: {aux_notes}"

    print(f"    Annotated beats in window: {evidence}")

    provenance = {
        "dataset": "mitdb",
        "data_origin": "real_clinical_physionet",
        "patient_id": "mitdb_patient_203",
        "record_id": record_id,
        "seizure_id": "none",
        "phase": "Hard_Negative_Arrhythmia",
        "channel_name": ch_name,
        "units": ch_units,
        "original_sampling_rate_hz": fs,
        "analysis_sampling_rate_hz": fs,
        "window_start_s": 0.0,
        "window_end_s": round(duration_s, 2),
        "window_duration_s": round(duration_s, 2),
        "n_samples": len(raw_ecg)
    }

    processed_sig, nan_frac, is_valid, val_notes = validate_and_preprocess_signal(raw_ecg, fs)
    if not is_valid:
        print(f"    [WARN] MIT-BIH signal invalid: {val_notes}")
        return None, [], {}, None

    cleaned_sig = nk.ecg_clean(processed_sig, sampling_rate=int(fs))

    # 1. Rhythm & HRV
    rhythm_feats, rr_recs, peaks = extract_rhythm_and_hrv(cleaned_sig, fs, provenance)

    # 2. Morphology & Beat Templates
    morph_feats, template, n_beats = extract_morphology_and_templates(cleaned_sig, peaks, fs)
    templates_dict = {}
    if len(template) > 0:
        templates_dict[f"{record_id}_Hard_Negative_Arrhythmia"] = template

    # 3. Spectral Features
    spectral_feats = compute_spectral_features(cleaned_sig, fs)

    # 4. Amplitude and Quality Features
    qual_feats = compute_quality_and_amplitude(raw_ecg, cleaned_sig, nan_frac, is_valid, len(peaks), fs)
    qual_feats["Annotation_Evidence"] = evidence

    full_row = {
        **provenance,
        **rhythm_feats,
        **morph_feats,
        **spectral_feats,
        **qual_feats
    }

    phase_signals = {"Hard_Negative_Arrhythmia": (raw_ecg, cleaned_sig, fs, 0.0, duration_s)}
    print(f"    [OK] Hard_Negative_Arrhythmia: HR={rhythm_feats['Mean_HR_BPM']} BPM, SDNN={rhythm_feats['SDNN_ms']} ms, Peaks={len(peaks)}\n")
    return full_row, rr_recs, templates_dict, phase_signals


# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------

def run_experiment_08():
    all_feature_rows = []
    all_rr_records = []
    all_beat_templates = {}
    representative_signals = {}

    # Metadata for szdb seizure records
    # Exact EEG-confirmed seizure timestamps from PhysioNet szdb
    seizure_records = [
        {
            "dataset": "szdb",
            "record_id": "sz01",
            "seizure_id": "sz01_sz1",
            "patient_id": "szdb_unknown",
            "onset_s": 876,
            "offset_s": 972,
            "pn_dir": "szdb"
        },
        {
            "dataset": "szdb",
            "record_id": "sz04",
            "seizure_id": "sz04_sz1",
            "patient_id": "szdb_unknown",
            "onset_s": 1210,
            "offset_s": 1315,
            "pn_dir": "szdb"
        }
    ]

    # Process all seizure records
    for s_meta in seizure_records:
        f_rows, rr_recs, t_dict, p_sigs = process_record_phases(
            dataset_name=s_meta["dataset"],
            record_id=s_meta["record_id"],
            seizure_id=s_meta["seizure_id"],
            patient_id=s_meta["patient_id"],
            onset_s=s_meta["onset_s"],
            offset_s=s_meta["offset_s"],
            pn_dir=s_meta["pn_dir"]
        )
        all_feature_rows.extend(f_rows)
        all_rr_records.extend(rr_recs)
        all_beat_templates.update(t_dict)
        if p_sigs and s_meta["record_id"] == "sz01":
            representative_signals.update(p_sigs)

    # Process real MIT-BIH hard negative
    hn_row, hn_rr, hn_templates, hn_sigs = process_mitbih_hard_negative(record_id="203", duration_s=60.0)
    if hn_row:
        all_feature_rows.append(hn_row)
        all_rr_records.extend(hn_rr)
        all_beat_templates.update(hn_templates)
        if hn_sigs:
            representative_signals.update(hn_sigs)

    if not all_feature_rows:
        raise RuntimeError("FATAL: No real ECG data could be processed. Terminating without synthetic fallback.")

    # -----------------------------------------------------------------------
    # Stage 9: Generate Output Contract Artifacts
    # -----------------------------------------------------------------------

    # 1. Primary Empirical Feature Table
    df_features = pd.DataFrame(all_feature_rows)
    feature_table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df_features.to_csv(feature_table_path, index=False)
    print(f"Saved primary feature table: {feature_table_path} ({len(df_features)} rows)")

    # 2. Long-Form RR Intervals
    df_rr = pd.DataFrame(all_rr_records)
    rr_path = os.path.join(OUTPUT_DIR, "08_rr_intervals.csv")
    df_rr.to_csv(rr_path, index=False)
    print(f"Saved long-form RR intervals: {rr_path} ({len(df_rr)} intervals)")

    # 3. Beat Templates NPZ
    templates_path = os.path.join(OUTPUT_DIR, "08_beat_templates.npz")
    np.savez_compressed(templates_path, **all_beat_templates)
    print(f"Saved aligned beat templates: {templates_path} ({len(all_beat_templates)} templates)")

    # 4. Data Quality Report
    quality_cols = [
        "dataset", "record_id", "seizure_id", "phase", "channel_name",
        "original_sampling_rate_hz", "window_duration_s", "Detected_Peaks", "Valid_RR_Count",
        "NaN_Fraction", "Clipping_Detected", "Clipping_Fraction", "Quality_OK", "Quality_Notes"
    ]
    avail_q_cols = [c for c in quality_cols if c in df_features.columns]
    df_quality = df_features[avail_q_cols]
    quality_path = os.path.join(OUTPUT_DIR, "08_data_quality_report.csv")
    df_quality.to_csv(quality_path, index=False)
    print(f"Saved data quality report: {quality_path}")

    # 5. Backward-Compatible Aggregate Table (derived strictly from real data)
    # Aggregates across records by phase/clinical class
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
    print(f"Saved backward-compatible aggregate: {compat_path} ({len(df_compat)} rows)")

    # -----------------------------------------------------------------------
    # Stage 10: Generate Figures
    # -----------------------------------------------------------------------

    # Figure 1: Real Seizure Phases (Waveform plots)
    fig, axes = plt.subplots(len(representative_signals), 1, figsize=(12, 11), sharex=False)
    fig.suptitle("Experiment 08: Real Seizure ECG & Hard-Negative Waveforms (sz01 & MIT-BIH 203)\n"
                 "Exploratory Empirical Characterization Across Seizure Phases (Zero Synthetic Fallback)",
                 fontsize=11, fontweight="bold", y=0.995)

    colors = ["steelblue", "darkorange", "purple", "seagreen", "crimson"]
    for idx, (p_name, (raw_sig, clean_sig, p_fs, t_start, t_end)) in enumerate(representative_signals.items()):
        ax = axes[idx]
        t_axis = np.arange(len(raw_sig)) / float(p_fs)
        ax.plot(t_axis, raw_sig, color=colors[idx % len(colors)], linewidth=0.7, alpha=0.9)
        rec_label = "MIT-BIH 203" if "Hard_Negative" in p_name else "sz01"
        ax.set_title(f"{rec_label} | {p_name.replace('_', ' ')} (Time: {t_start:.0f}s - {t_end:.0f}s, fs: {p_fs:.0f} Hz)",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Amplitude (mV)", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time within Phase Window (seconds)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig1_path = os.path.join(OUTPUT_DIR, "08_real_seizure_phases.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved waveform figure: {fig1_path}")

    # Figure 2: Empirical Feature Distributions across phases
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Experiment 08: Empirical Feature Distributions Across Real Seizure Phases\n"
                 "Exploratory Characterization (sz01, sz04, MIT-BIH 203; Note: Limited Exploratory Sample)",
                 fontsize=11, fontweight="bold")

    phases_list = [p for p in classes_order if p in df_features["phase"].values]
    x_pos = np.arange(len(phases_list))
    labels = [p.replace("_", "\n") for p in phases_list]

    # Panel 1: Mean HR
    hr_vals = [df_features[df_features["phase"] == p]["Mean_HR_BPM"].dropna().values for p in phases_list]
    axes[0, 0].boxplot(hr_vals, positions=x_pos, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    axes[0, 0].set_title("Mean Heart Rate (BPM)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("BPM")
    axes[0, 0].grid(True, linestyle="--", alpha=0.3)

    # Panel 2: SDNN
    sdnn_vals = [df_features[df_features["phase"] == p]["SDNN_ms"].dropna().values for p in phases_list]
    axes[0, 1].boxplot(sdnn_vals, positions=x_pos, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    axes[0, 1].set_title("Heart Rate Variability (SDNN, ms)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("ms")
    axes[0, 1].grid(True, linestyle="--", alpha=0.3)

    # Panel 3: High-Frequency Power Proxy
    hf_vals = [df_features[df_features["phase"] == p]["EMG_Associated_High_Frequency_Power"].dropna().values for p in phases_list]
    axes[1, 0].boxplot(hf_vals, positions=x_pos, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="lightsalmon"))
    axes[1, 0].set_title("High-Frequency Spectral Power (Proxy for Muscle Tremor)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Power Density (mV^2/Hz)")
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    # Panel 4: Morphology Variability
    morph_vals = [df_features[df_features["phase"] == p]["Morphology_Variability"].dropna().values for p in phases_list]
    # If empty or all NaN, handle gracefully
    if any(len(v) > 0 for v in morph_vals):
        axes[1, 1].boxplot(morph_vals, positions=x_pos, tick_labels=labels, patch_artist=True, boxprops=dict(facecolor="plum"))
    axes[1, 1].set_title("Morphology Variability (1 - Beat Correlation)", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("Variability Index [0-1]")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig2_path = os.path.join(OUTPUT_DIR, "08_real_feature_distributions.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature distribution figure: {fig2_path}")

    print("\n=== Experiment 08 Complete: Real-Data Extraction Successfully Validated ===")


if __name__ == "__main__":
    run_experiment_08()
