"""
Experiment 09: Empirically Calibrated Parametric Synthetic ECG Generation Pipeline
==================================================================================
Objective:
  Generate continuous synthetic ECG signals empirically calibrated to the full suite
  of real clinical seizure observations from PhysioNet SZDB (records sz01-sz07, 10 seizure
  events across 4 distinct seizure phases) and MIT-BIH 203 arrhythmia control extracted in Experiment 08.

Scientific Rigor & Integrity Principles:
  - NO collapsing across records: Each synthetic signal is anchored on a specific real
    empirical joint feature vector (record_id, seizure_id, phase), preserving observed biological variability.
  - Phase-conditional generation: Interictal, Preictal, Ictal, and Postictal distributions
    are strictly maintained without cross-phase mixing.
  - Arrhythmia control separation: The MIT-BIH 203 recording is classified explicitly as
    a "cardiac_arrhythmia_control" condition, separate from epileptic seizure phases.
  - Optimization-based calibration over explicitly defined simulator parameter search spaces:
    Replaces guesswork heuristics with optimization over bounded candidate grids
    minimizing empirical error against clinical targets (SDNN, Welch spectral features).
  - Explicit simulator search bounds: Search spaces are documented as numerical simulator configurations,
    not clinical physiological measurements.
  - Zero missing-data fallbacks: If empirical rhythm targets (HR, SDNN) are missing or non-positive,
    generation is skipped and documented in the quality report without fabricating synthetic values.
  - Continuous dynamical trajectories: Continuous 60-second generation in single calls
    eliminates artificial chunk cuts, boundary steps, and transient restarts.
  - Source sampling-rate provenance: Standardized synthetic generation is fixed to 200 Hz (FS = 200),
    while preserving source sampling rates (200 Hz for SZDB, 360 Hz for MIT-BIH) in the provenance schema.
  - Strict post-hoc morphology evaluation: Morphology is evaluated post-hoc against Exp 08 targets
    (storing QRS error, QTc error, R-amplitude error, and beat-template correlation) rather than
    erroneously claimed as direct simulator controls.

Outputs:
  - outputs/09_empirical_generation_parameters.csv (Per-signal provenance and calibrated parameters)
  - outputs/09_synthetic_ecg_signals.npz (Continuous synthetic ECG arrays keyed by synthetic_id)
  - outputs/09_synthetic_feature_table.csv (Re-extracted features using aligned Exp 08 pipeline)
  - outputs/09_empirical_vs_synthetic_parameters.csv (Direct target vs achieved comparison)
  - outputs/09_generation_quality_report.csv (Truthful diagnostics, continuity, calibration status)
  - outputs/09_fitted_synthetic_signals.csv (Backward-compatible summary table for Exp 10 & 11)
  - outputs/09_empirical_synthetic_fitting.png (Waveform inspection and distribution comparison)

Scientific Disclaimer:
  Synthetic rhythm and noise parameters are calibrated to clinical PhysioNet data over explicit
  simulator search ranges. Because kinematic ECG models (ECGSYN) do not expose independent
  morphological equations, morphology metrics represent post-hoc benchmark evaluations rather
  than direct simulator parameterizations.
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

# ---------------------------------------------------------------------------
# Numerical Stability Guard for NeuroKit2 ECGSYN RR Resampling
# ---------------------------------------------------------------------------
# In NeuroKit2's _ecg_simulate_ecgsyn, the 1 Hz RR time series is resampled to
# sfint via scipy.ndimage.zoom spline interpolation. Spline boundary effects
# can occasionally evaluate the final sample to 0.0, which causes the ODE
# time-stepping loop (while i < len(rr): tecg += rr[i]; i = ip) to stall in an
# infinite loop. We guard against non-positive boundary values by clipping
# resampled RR intervals to a physiological floor of 0.05 seconds.
# ---------------------------------------------------------------------------
try:
    from neurokit2.ecg.ecg_simulate import _ecg_simulate_ecgsyn
    _orig_nk_resample = _ecg_simulate_ecgsyn.__globals__.get("signal_resample")
    if _orig_nk_resample is not None:
        def _safe_nk_resample(signal, *args, **kwargs):
            res = _orig_nk_resample(signal, *args, **kwargs)
            if isinstance(res, np.ndarray) and len(res) > 0 and np.all(np.array(signal) > 0.05):
                res = np.maximum(res, 0.05)
            return res
        _ecg_simulate_ecgsyn.__globals__["signal_resample"] = _safe_nk_resample
except Exception:
    pass

# Guarantee outputs directory exists
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Centralized Simulator Parameter Search Configurations
# ---------------------------------------------------------------------------
# SCIENTIFIC CLARIFICATION ON SEARCH SPACES:
# These constants define SIMULATOR SEARCH BOUNDS and CANDIDATE GRIDS for the
# numerical ECG generator (NeuroKit2 / ECGSYN). They are NOT empirical clinical
# physiological measurements. The selected parameter is empirical only in the
# specific sense that it minimizes quantitative error against a real clinical
# target feature extracted from PhysioNet SZDB or MIT-BIH recordings.
# ---------------------------------------------------------------------------

HR_STD_SEARCH_BOUNDS = (0.5, 14.0)
"""Simulator heart_rate_std bounds (BPM).
- Lower bound (0.5 BPM) avoids degenerate periodic flatlines.
- Upper bound (14.0 BPM) prevents negative RR intervals in the ECGSYN ODE integrator,
  which otherwise cause non-convergence in the internal RK4 integration loop."""

HR_STD_LOCAL_SEARCH_FACTORS = [0.75, 0.85, 1.00, 1.15, 1.25]
"""Local search multiplier grid evaluated around the initial analytical approximation center.
Provides dense candidate coverage to find the parameter minimizing empirical SDNN error."""

NOISE_SEARCH_GRID = [0.005, 0.010, 0.025, 0.060, 0.120]
"""Simulator noise candidate grid for spectral matching.
Evaluated against empirical Welch PSD spectral features."""

FS = 200            # Standardized generation sampling frequency (Hz)
DURATION_SEC = 60   # Continuous signal duration (seconds)
N_SAMPLES = FS * DURATION_SEC


# ---------------------------------------------------------------------------
# Signal Processing & Re-extraction Helpers (Aligned with Exp 08)
# ---------------------------------------------------------------------------

def compute_spectral_features(signal, fs=FS):
    """
    Computes Welch PSD spectral power bands and centroid.
    Frequency bands and formulas are identical to Experiment 08:
      - Baseline Wander Band: [0.0, 0.5) Hz
      - ECG Dominant Band:    [0.5, min(20.0, Nyquist)] Hz
      - High Frequency Band:  [20.0, min(100.0, Nyquist)] Hz
    """
    if hasattr(signal, 'iloc'):
        signal = signal.iloc[:, 0].values
    elif isinstance(signal, np.ndarray) and signal.ndim > 1:
        signal = signal.flatten()

    nperseg = min(len(signal), int(4 * fs))
    freqs, psd = sp_signal.welch(signal, fs=fs, nperseg=nperseg)
    f_nyq = fs / 2.0

    bw_idx = (freqs >= 0.0) & (freqs < 0.5)
    ecg_idx = (freqs >= 0.5) & (freqs <= min(20.0, f_nyq))
    hf_top = min(100.0, f_nyq)
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
        "spectral_valid": spectral_valid
    }


def reextract_synthetic_features(signal, fs=FS):
    """
    Extracts rhythm, morphology, spectral, and amplitude features from a synthetic signal
    using feature definitions strictly compatible with Experiment 08.
    """
    if hasattr(signal, 'iloc'):
        signal = signal.iloc[:, 0].values
    elif isinstance(signal, np.ndarray) and signal.ndim > 1:
        signal = signal.flatten()

    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    rhythm_dict = {
        "Detected_Peaks": 0, "Valid_RR_Count": 0, "Mean_HR_BPM": np.nan, "Median_HR_BPM": np.nan,
        "Std_HR_BPM": np.nan, "Mean_RR_ms": np.nan, "Median_RR_ms": np.nan, "SDNN_ms": np.nan,
        "RMSSD_ms": np.nan, "RR_Min_ms": np.nan, "RR_Max_ms": np.nan, "RR_CV": np.nan,
        "RR_IQR_ms": np.nan, "peak_detection_ok": False
    }
    morph_dict = {
        "QRS_Duration_Median_ms": np.nan, "QRS_Duration_IQR_ms": np.nan, "QRS_Amplitude_Median_mV": np.nan,
        "R_Amplitude_Median_mV": np.nan, "QT_Interval_Median_ms": np.nan, "QTc_Median_ms": np.nan,
        "Beat_Correlation_Median": np.nan, "Beat_Correlation_IQR": np.nan, "Beat_RMSE": np.nan,
        "Morphology_Variability": np.nan, "morphology_valid": False
    }
    median_template = np.array([])

    # 1. Peak detection & Rhythm (Pan-Tompkins with neurokit fallback)
    try:
        _, p_info = nk.ecg_peaks(cleaned, sampling_rate=int(fs), method="pantompkins1985")
        peaks = p_info.get("ECG_R_Peaks", np.array([], dtype=int))
    except Exception:
        try:
            _, p_info = nk.ecg_peaks(cleaned, sampling_rate=int(fs), method="neurokit")
            peaks = p_info.get("ECG_R_Peaks", np.array([], dtype=int))
        except Exception:
            peaks = np.array([], dtype=int)

    detected_peaks = len(peaks)
    rhythm_dict["Detected_Peaks"] = detected_peaks

    if detected_peaks >= 3:
        raw_rr_ms = (np.diff(peaks) / float(fs)) * 1000.0
        # Filter physiological RR intervals: 250 ms to 2500 ms (identical to Exp 08)
        valid_rr = raw_rr_ms[(raw_rr_ms >= 250.0) & (raw_rr_ms <= 2500.0)]
        rhythm_dict["Valid_RR_Count"] = len(valid_rr)
        if len(valid_rr) >= 2:
            inst_hr = 60000.0 / valid_rr
            mean_rr = float(np.mean(valid_rr))
            sdnn = float(np.std(valid_rr, ddof=1))
            diff_rr = np.diff(valid_rr)
            rmssd = float(np.sqrt(np.mean(diff_rr**2))) if len(diff_rr) > 0 else np.nan
            rr_iqr = float(np.percentile(valid_rr, 75) - np.percentile(valid_rr, 25))

            rhythm_dict["Mean_HR_BPM"] = round(float(np.mean(inst_hr)), 2)
            rhythm_dict["Median_HR_BPM"] = round(float(np.median(inst_hr)), 2)
            rhythm_dict["Std_HR_BPM"] = round(float(np.std(inst_hr, ddof=1)), 2)
            rhythm_dict["Mean_RR_ms"] = round(mean_rr, 2)
            rhythm_dict["Median_RR_ms"] = round(float(np.median(valid_rr)), 2)
            rhythm_dict["SDNN_ms"] = round(sdnn, 2)
            rhythm_dict["RMSSD_ms"] = round(rmssd, 2) if np.isfinite(rmssd) else np.nan
            rhythm_dict["RR_Min_ms"] = round(float(np.min(valid_rr)), 2)
            rhythm_dict["RR_Max_ms"] = round(float(np.max(valid_rr)), 2)
            rhythm_dict["RR_CV"] = round(float(sdnn / mean_rr), 4) if mean_rr > 0 else np.nan
            rhythm_dict["RR_IQR_ms"] = round(rr_iqr, 2)
            rhythm_dict["peak_detection_ok"] = True

    # 2. Morphology & Template Extraction (Aligned with Exp 08)
    if detected_peaks >= 3:
        delineation_ok = False
        try:
            _, waves = nk.ecg_delineate(cleaned, peaks, sampling_rate=int(fs), method="dwt")
            q_peaks = np.array(waves.get("ECG_Q_Peaks", []))
            s_peaks = np.array(waves.get("ECG_S_Peaks", []))
            t_offsets = np.array(waves.get("ECG_T_Offsets", []))

            valid_qs = [i for i in range(min(len(q_peaks), len(s_peaks)))
                        if np.isfinite(q_peaks[i]) and np.isfinite(s_peaks[i])]
            if len(valid_qs) >= 2:
                q_i = np.array(q_peaks[valid_qs], dtype=int)
                s_i = np.array(s_peaks[valid_qs], dtype=int)
                qrs_dur = (s_i - q_i) / float(fs) * 1000.0
                qrs_dur = qrs_dur[(qrs_dur > 20.0) & (qrs_dur < 300.0)]
                if len(qrs_dur) >= 2:
                    morph_dict["QRS_Duration_Median_ms"] = round(float(np.median(qrs_dur)), 2)
                    morph_dict["QRS_Duration_IQR_ms"] = round(float(np.percentile(qrs_dur, 75) - np.percentile(qrs_dur, 25)), 2)
                    qrs_amps = np.abs(cleaned[s_i] - cleaned[q_i])
                    morph_dict["QRS_Amplitude_Median_mV"] = round(float(np.median(qrs_amps)), 4)
                    delineation_ok = True

            morph_dict["R_Amplitude_Median_mV"] = round(float(np.median(cleaned[peaks])), 4)

            valid_qt = [i for i in range(min(len(q_peaks), len(t_offsets)))
                        if np.isfinite(q_peaks[i]) and np.isfinite(t_offsets[i])]
            if len(valid_qt) >= 2:
                q_idx = np.array(q_peaks[valid_qt], dtype=int)
                t_idx = np.array(t_offsets[valid_qt], dtype=int)
                qt_ms = (t_idx - q_idx) / float(fs) * 1000.0
                qt_ms = qt_ms[(qt_ms > 150.0) & (qt_ms < 800.0)]
                if len(qt_ms) >= 2:
                    med_qt = float(np.median(qt_ms))
                    morph_dict["QT_Interval_Median_ms"] = round(med_qt, 2)
                    rr_sec = float(np.median(np.diff(peaks)) / fs) if len(peaks) > 1 else 0.8
                    if rr_sec > 0.2:
                        morph_dict["QTc_Median_ms"] = round(med_qt / np.sqrt(rr_sec), 2)

            morph_dict["morphology_valid"] = delineation_ok
        except Exception:
            morph_dict["morphology_valid"] = False

        # Template Extraction (-200 ms to +400 ms around R-peak)
        pre_s = int(0.200 * fs)
        post_s = int(0.400 * fs)
        epochs = [cleaned[p - pre_s : p + post_s] for p in peaks if p - pre_s >= 0 and p + post_s <= len(cleaned)]
        if len(epochs) >= 3:
            ep_arr = np.array(epochs)
            median_template = np.median(ep_arr, axis=0)
            corrs = []
            rmses = []
            tmpl_norm = np.linalg.norm(median_template - np.mean(median_template))
            for b in ep_arr:
                b_norm = np.linalg.norm(b - np.mean(b))
                if b_norm > 1e-6 and tmpl_norm > 1e-6:
                    r = np.dot(b - np.mean(b), median_template - np.mean(median_template)) / (b_norm * tmpl_norm)
                    corrs.append(np.clip(r, -1.0, 1.0))
                rmses.append(float(np.sqrt(np.mean((b - median_template)**2))))
            if len(corrs) >= 2:
                med_c = float(np.median(corrs))
                iqr_c = float(np.percentile(corrs, 75) - np.percentile(corrs, 25))
                morph_dict["Beat_Correlation_Median"] = round(med_c, 4)
                morph_dict["Beat_Correlation_IQR"] = round(iqr_c, 4)
                morph_dict["Beat_RMSE"] = round(float(np.median(rmses)), 4)
                morph_dict["Morphology_Variability"] = round(max(0.0, 1.0 - med_c), 4)

    # 3. Spectral Features
    spectral_dict = compute_spectral_features(cleaned, fs)

    # 4. Amplitude Features
    amp_dict = {
        "Signal_Mean_mV": round(float(np.mean(signal)), 4),
        "Signal_Std_mV": round(float(np.std(signal)), 4),
        "Signal_RMS_mV": round(float(np.sqrt(np.mean(signal**2))), 4),
        "Peak_to_Peak_mV": round(float(np.ptp(signal)), 4),
        "Robust_Amplitude_Range_mV": round(float(np.percentile(signal, 95) - np.percentile(signal, 5)), 4)
    }

    return {**rhythm_dict, **morph_dict, **spectral_dict, **amp_dict}, median_template


# ---------------------------------------------------------------------------
# Calibration Engine
# ---------------------------------------------------------------------------

def calibrate_rhythm_hrv(target_hr, target_sdnn, method="ecgsyn", seed=42, duration_sec=DURATION_SEC, fs=FS):
    """
    Calibrates simulator heart_rate_std against empirical target SDNN.

    Scientific Principle:
      The analytical approximation:
        initial_hr_std_estimate = (target_sdnn * (target_hr**2) / 60000.0) / 0.94
      is used ONLY as an INITIAL SEARCH CENTER, not as an exact physiological mapping.
      The final selected heart_rate_std is chosen from actual generated candidate signals by:
        1. Generating candidate ECG at each local search candidate
        2. Re-detecting R-peaks via Pan-Tompkins
        3. Filtering physiological RR intervals (250-2500 ms)
        4. Computing achieved SDNN
        5. Minimizing absolute error versus target SDNN.

    Strict Missing-Data Policy:
      If target_hr or target_sdnn is missing (NaN), non-finite, or <= 0:
        Returns calibrated=False without inventing fallback simulator defaults.
    """
    if not (np.isfinite(target_hr) and target_hr > 0 and np.isfinite(target_sdnn) and target_sdnn > 0):
        return {
            "calibrated": False,
            "status": "MISSING_REQUIRED_INPUTS",
            "initial_hr_std_estimate": np.nan,
            "selected_hr_std": np.nan,
            "target_sdnn": round(target_sdnn, 2) if np.isfinite(target_sdnn) else np.nan,
            "achieved_sdnn": np.nan,
            "sdnn_error": np.nan,
            "number_of_candidates_evaluated": 0,
            "best_signal": None,
            "failure_reason": "Target HR or SDNN is missing, NaN, or non-positive"
        }

    # 1. Compute initial search center using documented analytical approximation
    initial_estimate = (target_sdnn * (target_hr**2) / 60000.0) / 0.94
    bounded_initial = float(np.clip(initial_estimate, HR_STD_SEARCH_BOUNDS[0], HR_STD_SEARCH_BOUNDS[1]))

    # 2. Form dense candidate grid multiplying bounded center by local search factors
    candidates = sorted(list(set(
        float(np.clip(bounded_initial * f, HR_STD_SEARCH_BOUNDS[0], HR_STD_SEARCH_BOUNDS[1]))
        for f in HR_STD_LOCAL_SEARCH_FACTORS
    )))

    best_hr_std = None
    best_sdnn_err = float("inf")
    best_achieved_sdnn = np.nan
    best_signal = None
    candidates_evaluated = 0

    sim_hr = float(np.clip(target_hr, 35.0, 180.0))

    for cand_std in candidates:
        candidates_evaluated += 1
        try:
            sig = nk.ecg_simulate(
                duration=duration_sec,
                sampling_rate=fs,
                heart_rate=sim_hr,
                heart_rate_std=cand_std,
                noise=0.01,  # Baseline nominal noise for peak detection
                method=method,
                random_state=seed
            )
            if hasattr(sig, 'iloc'):
                sig = sig.iloc[:, 0].values
            elif isinstance(sig, np.ndarray) and sig.ndim > 1:
                sig = sig.flatten()

            cleaned = nk.ecg_clean(sig, sampling_rate=fs)
            _, p_info = nk.ecg_peaks(cleaned, sampling_rate=fs, method="pantompkins1985")
            peaks = p_info.get("ECG_R_Peaks", [])
            if len(peaks) >= 3:
                raw_rr_ms = (np.diff(peaks) / float(fs)) * 1000.0
                valid_rr = raw_rr_ms[(raw_rr_ms >= 250.0) & (raw_rr_ms <= 2500.0)]
                if len(valid_rr) >= 2:
                    ach_sdnn = float(np.std(valid_rr, ddof=1))
                    err = abs(ach_sdnn - target_sdnn)
                    if err < best_sdnn_err:
                        best_sdnn_err = err
                        best_hr_std = cand_std
                        best_achieved_sdnn = ach_sdnn
                        best_signal = sig
        except Exception:
            continue

    if best_hr_std is None or best_signal is None:
        return {
            "calibrated": False,
            "status": "CALIBRATION_FAILED",
            "initial_hr_std_estimate": round(bounded_initial, 3),
            "selected_hr_std": np.nan,
            "target_sdnn": round(target_sdnn, 2),
            "achieved_sdnn": np.nan,
            "sdnn_error": np.nan,
            "number_of_candidates_evaluated": candidates_evaluated,
            "best_signal": None,
            "failure_reason": "No candidate signal produced sufficient valid RR intervals"
        }

    return {
        "calibrated": True,
        "status": "CALIBRATED",
        "initial_hr_std_estimate": round(bounded_initial, 3),
        "selected_hr_std": round(best_hr_std, 3),
        "target_sdnn": round(target_sdnn, 2),
        "achieved_sdnn": round(best_achieved_sdnn, 2),
        "sdnn_error": round(best_sdnn_err, 2),
        "number_of_candidates_evaluated": candidates_evaluated,
        "best_signal": best_signal,
        "failure_reason": None
    }


def calibrate_spectral_noise(target_hr, selected_hr_std, target_spectral_features,
                             method="ecgsyn", seed=42, duration_sec=DURATION_SEC, fs=FS):
    """
    Calibrates simulator noise against real target Welch PSD spectral features
    using a normalized multi-feature objective.

    Evaluates candidate noise values from NOISE_SEARCH_GRID.
    For each candidate:
      1. Generates ECG at selected_hr_std and candidate noise
      2. Computes Welch PSD spectral features
      3. Calculates normalized multi-feature objective across valid finite target features:
         - Relative_High_Frequency_Power (normalized by target value or 0.01)
         - Baseline_Wander_Band_Power (normalized by target value or 1e-6)
         - Spectral_Centroid_Hz (normalized by target value or 1.0)

    Only features where BOTH target and candidate values are finite are included in the objective.
    If no valid target spectral features exist:
      Marks calibration as NOT_CALIBRATED rather than claiming an empirical fit.
    """
    target_rel_hf = target_spectral_features.get("Relative_High_Frequency_Power", np.nan)
    target_bw = target_spectral_features.get("Baseline_Wander_Band_Power", np.nan)
    target_centroid = target_spectral_features.get("Spectral_Centroid_Hz", np.nan)

    valid_targets = {}
    if np.isfinite(target_rel_hf) and target_rel_hf >= 0:
        valid_targets["Relative_High_Frequency_Power"] = (target_rel_hf, max(float(target_rel_hf), 0.01))
    if np.isfinite(target_bw) and target_bw >= 0:
        valid_targets["Baseline_Wander_Band_Power"] = (target_bw, max(float(target_bw), 1e-6))
    if np.isfinite(target_centroid) and target_centroid > 0:
        valid_targets["Spectral_Centroid_Hz"] = (target_centroid, max(float(target_centroid), 1.0))

    if not valid_targets:
        return {
            "calibrated": False,
            "status": "NOT_CALIBRATED",
            "selected_noise": np.nan,
            "spectral_objective": np.nan,
            "spectral_features_used": "None",
            "number_of_noise_candidates_evaluated": 0,
            "final_signal": None,
            "failure_reason": "No valid finite spectral targets available for calibration"
        }

    best_noise = None
    best_objective = float("inf")
    final_signal = None
    candidates_evaluated = 0
    sim_hr = float(np.clip(target_hr, 35.0, 180.0))

    for cand_noise in NOISE_SEARCH_GRID:
        candidates_evaluated += 1
        try:
            sig = nk.ecg_simulate(
                duration=duration_sec,
                sampling_rate=fs,
                heart_rate=sim_hr,
                heart_rate_std=selected_hr_std,
                noise=cand_noise,
                method=method,
                random_state=seed
            )
            if hasattr(sig, 'iloc'):
                sig = sig.iloc[:, 0].values
            elif isinstance(sig, np.ndarray) and sig.ndim > 1:
                sig = sig.flatten()

            spec = compute_spectral_features(sig, fs)

            feature_errors = []
            for feat_name, (tgt_val, normalizer) in valid_targets.items():
                cand_val = spec.get(feat_name, np.nan)
                if np.isfinite(cand_val):
                    rel_err = abs(cand_val - tgt_val) / normalizer
                    feature_errors.append(rel_err)

            if feature_errors:
                obj = float(np.mean(feature_errors))
                if obj < best_objective:
                    best_objective = obj
                    best_noise = cand_noise
                    final_signal = sig
        except Exception:
            continue

    if best_noise is None or final_signal is None:
        return {
            "calibrated": False,
            "status": "CALIBRATION_FAILED",
            "selected_noise": np.nan,
            "spectral_objective": np.nan,
            "spectral_features_used": ", ".join(valid_targets.keys()),
            "number_of_noise_candidates_evaluated": candidates_evaluated,
            "final_signal": None,
            "failure_reason": "Simulator failed to generate candidate signal for all noise levels"
        }

    return {
        "calibrated": True,
        "status": "CALIBRATED",
        "selected_noise": round(best_noise, 4),
        "spectral_objective": round(best_objective, 6),
        "spectral_features_used": ", ".join(valid_targets.keys()),
        "number_of_noise_candidates_evaluated": candidates_evaluated,
        "final_signal": final_signal,
        "failure_reason": None
    }


# ---------------------------------------------------------------------------
# Main Pipeline Execution
# ---------------------------------------------------------------------------

def run_experiment_09():
    print("=========================================================================", flush=True)
    print("  EXPERIMENT 09: EMPIRICALLY CALIBRATED SYNTHETIC ECG GENERATION ENGINE", flush=True)
    print("=========================================================================\n", flush=True)

    # 1. Audit & Ingest Experiment 08 Empirical Outputs
    primary_features_csv = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    rr_intervals_csv = os.path.join(OUTPUT_DIR, "08_rr_intervals.csv")
    templates_npz = os.path.join(OUTPUT_DIR, "08_beat_templates.npz")
    compat_csv = os.path.join(OUTPUT_DIR, "08_empirical_seizure_parameters.csv")

    if not os.path.exists(primary_features_csv):
        raise FileNotFoundError(f"Missing required Experiment 08 output: {primary_features_csv}. Run Exp 08 first.")

    df_emp_full = pd.read_csv(primary_features_csv)
    print(f"[Audit] Loaded primary empirical feature table: {primary_features_csv} ({len(df_emp_full)} observations)", flush=True)

    if "Quality_OK" in df_emp_full.columns:
        df_emp = df_emp_full[df_emp_full["Quality_OK"] == True].copy().reset_index(drop=True)
    else:
        df_emp = df_emp_full.copy().reset_index(drop=True)

    print(f"[Audit] Usable valid empirical observations: {len(df_emp)} across phases: {df_emp['phase'].value_counts().to_dict()}", flush=True)

    real_templates = np.load(templates_npz) if os.path.exists(templates_npz) else {}

    synthetic_signals = {}
    generation_params_records = []
    feature_table_records = []
    comparison_records = []
    quality_report_records = []
    compat_class_records = []

    seizure_phases = [
        "Interictal_Baseline",
        "Preictal_Prediction",
        "Ictal_Seizure",
        "Postictal_Recovery"
    ]

    all_classes_order = seizure_phases + ["Hard_Negative_Arrhythmia"]

    print(f"--> Beginning Empirically Calibrated Generation across {len(df_emp)} Real Observations...\n", flush=True)

    skipped_count = 0
    generated_count = 0

    for obs_idx, row in df_emp.iterrows():
        rec_id = str(row.get("record_id", "unknown"))
        sz_id = str(row.get("seizure_id", "none"))
        phase = str(row.get("phase", "unknown"))
        dataset_name = str(row.get("dataset", "unknown"))
        patient_id = str(row.get("patient_id", np.nan))
        syn_id = f"syn_{rec_id}_{sz_id}_{phase}"

        source_fs = float(row.get("original_sampling_rate_hz", row.get("analysis_sampling_rate_hz", row.get("fs", 200.0))))
        gen_fs = float(FS)

        is_arrhythmia_control = (phase == "Hard_Negative_Arrhythmia") or (dataset_name.lower() == "mitdb")
        condition_type = "cardiac_arrhythmia_control" if is_arrhythmia_control else "seizure_phase"

        target_mean_hr = float(row["Mean_HR_BPM"]) if "Mean_HR_BPM" in row and pd.notna(row["Mean_HR_BPM"]) else np.nan
        target_std_hr = float(row["Std_HR_BPM"]) if "Std_HR_BPM" in row and pd.notna(row["Std_HR_BPM"]) else np.nan
        target_sdnn = float(row["SDNN_ms"]) if "SDNN_ms" in row and pd.notna(row["SDNN_ms"]) else np.nan
        target_rmssd = float(row["RMSSD_ms"]) if "RMSSD_ms" in row and pd.notna(row["RMSSD_ms"]) else np.nan
        target_qrs_dur = float(row["QRS_Duration_Median_ms"]) if "QRS_Duration_Median_ms" in row and pd.notna(row["QRS_Duration_Median_ms"]) else np.nan
        target_qtc = float(row["QTc_Median_ms"]) if "QTc_Median_ms" in row and pd.notna(row["QTc_Median_ms"]) else np.nan
        target_r_amp = float(row["R_Amplitude_Median_mV"]) if "R_Amplitude_Median_mV" in row and pd.notna(row["R_Amplitude_Median_mV"]) else np.nan
        target_sig_std = float(row["Signal_Std_mV"]) if "Signal_Std_mV" in row and pd.notna(row["Signal_Std_mV"]) else np.nan

        target_spectral = {
            "Relative_High_Frequency_Power": float(row["Relative_High_Frequency_Power"]) if "Relative_High_Frequency_Power" in row and pd.notna(row["Relative_High_Frequency_Power"]) else np.nan,
            "Baseline_Wander_Band_Power": float(row["Baseline_Wander_Band_Power"]) if "Baseline_Wander_Band_Power" in row and pd.notna(row["Baseline_Wander_Band_Power"]) else np.nan,
            "Spectral_Centroid_Hz": float(row["Spectral_Centroid_Hz"]) if "Spectral_Centroid_Hz" in row and pd.notna(row["Spectral_Centroid_Hz"]) else np.nan
        }

        # Strict Missing-Data Rule: Skip generation if required core rhythm targets are missing
        if not (np.isfinite(target_mean_hr) and target_mean_hr > 0 and np.isfinite(target_sdnn) and target_sdnn > 0):
            skipped_count += 1
            missing_inputs = []
            if not (np.isfinite(target_mean_hr) and target_mean_hr > 0):
                missing_inputs.append("Mean_HR_BPM")
            if not (np.isfinite(target_sdnn) and target_sdnn > 0):
                missing_inputs.append("SDNN_ms")
            missing_str = ", ".join(missing_inputs)
            print(f"  [SKIP] {syn_id}: Missing required empirical rhythm inputs ({missing_str}). Skipping generation.", flush=True)

            quality_report_records.append({
                "synthetic_id": syn_id,
                "source_record_id": rec_id,
                "source_seizure_id": sz_id,
                "phase": phase,
                "condition_type": condition_type,
                "source_sampling_rate_hz": source_fs,
                "generation_sampling_rate_hz": gen_fs,
                "generation_status": "SKIPPED",
                "rhythm_calibration_status": "MISSING_REQUIRED_INPUTS",
                "spectral_calibration_status": "NOT_ATTEMPTED",
                "is_continuous": False,
                "no_chunk_boundaries": True,
                "boundary_discontinuity_status": "N/A (Skipped)",
                "max_sample_step_diff": np.nan,
                "mean_sample_step_diff": np.nan,
                "signal_std": np.nan,
                "detected_peaks": 0,
                "missing_required_inputs": missing_str,
                "failure_reason": f"Empirical rhythm targets unavailable ({missing_str}); skipped without synthetic fallback",
                "limitations": "Cannot calibrate simulator without valid empirical targets"
            })
            continue

        sim_method = "ecgsyn"
        sim_seed = 42 + obs_idx * 17

        # 1. Calibrate Rhythm HRV
        rhythm_cal = calibrate_rhythm_hrv(
            target_hr=target_mean_hr,
            target_sdnn=target_sdnn,
            method=sim_method,
            seed=sim_seed,
            duration_sec=DURATION_SEC,
            fs=FS
        )

        if not rhythm_cal["calibrated"]:
            skipped_count += 1
            print(f"  [ERROR] Rhythm calibration failed for {syn_id}: {rhythm_cal['failure_reason']}", flush=True)
            quality_report_records.append({
                "synthetic_id": syn_id,
                "source_record_id": rec_id,
                "source_seizure_id": sz_id,
                "phase": phase,
                "condition_type": condition_type,
                "source_sampling_rate_hz": source_fs,
                "generation_sampling_rate_hz": gen_fs,
                "generation_status": "FAILED",
                "rhythm_calibration_status": rhythm_cal["status"],
                "spectral_calibration_status": "NOT_ATTEMPTED",
                "is_continuous": False,
                "no_chunk_boundaries": True,
                "boundary_discontinuity_status": "N/A (Failed)",
                "max_sample_step_diff": np.nan,
                "mean_sample_step_diff": np.nan,
                "signal_std": np.nan,
                "detected_peaks": 0,
                "missing_required_inputs": "None",
                "failure_reason": rhythm_cal["failure_reason"],
                "limitations": "Rhythm candidate search failed to achieve valid RR intervals"
            })
            continue

        selected_hr_std = rhythm_cal["selected_hr_std"]

        # 2. Calibrate Spectral Noise
        spectral_cal = calibrate_spectral_noise(
            target_hr=target_mean_hr,
            selected_hr_std=selected_hr_std,
            target_spectral_features=target_spectral,
            method=sim_method,
            seed=sim_seed,
            duration_sec=DURATION_SEC,
            fs=FS
        )

        if spectral_cal["calibrated"]:
            cal_noise = spectral_cal["selected_noise"]
            spec_obj = spectral_cal["spectral_objective"]
            spec_status = "CALIBRATED"
            spec_features_used = spectral_cal["spectral_features_used"]
            syn_signal = spectral_cal["final_signal"]
        else:
            cal_noise = 0.010
            spec_obj = np.nan
            spec_status = "NOT_CALIBRATED"
            spec_features_used = "None"
            syn_signal = rhythm_cal["best_signal"]

        if syn_signal is None or len(syn_signal) == 0:
            skipped_count += 1
            print(f"  [ERROR] Signal synthesis produced empty array for {syn_id}.", flush=True)
            quality_report_records.append({
                "synthetic_id": syn_id,
                "source_record_id": rec_id,
                "source_seizure_id": sz_id,
                "phase": phase,
                "condition_type": condition_type,
                "source_sampling_rate_hz": source_fs,
                "generation_sampling_rate_hz": gen_fs,
                "generation_status": "FAILED",
                "rhythm_calibration_status": rhythm_cal["status"],
                "spectral_calibration_status": spec_status,
                "is_continuous": False,
                "no_chunk_boundaries": True,
                "boundary_discontinuity_status": "N/A (Failed)",
                "max_sample_step_diff": np.nan,
                "mean_sample_step_diff": np.nan,
                "signal_std": np.nan,
                "detected_peaks": 0,
                "missing_required_inputs": "None",
                "failure_reason": "Simulator returned empty signal array",
                "limitations": "Simulator output empty"
            })
            continue

        generated_count += 1

        step_diffs = np.abs(np.diff(syn_signal))
        max_step_diff = float(np.max(step_diffs))
        mean_step_diff = float(np.mean(step_diffs))

        synthetic_signals[syn_id] = syn_signal

        syn_features, syn_template = reextract_synthetic_features(syn_signal, fs=FS)

        tmpl_corr = np.nan
        target_tmpl_key = f"{rec_id}_{sz_id}_{phase}"
        alt_tmpl_key = f"{rec_id}_{phase}"
        real_tmpl = None
        if target_tmpl_key in real_templates:
            real_tmpl = real_templates[target_tmpl_key]
        elif alt_tmpl_key in real_templates:
            real_tmpl = real_templates[alt_tmpl_key]

        if real_tmpl is not None and len(real_tmpl) > 0 and len(syn_template) > 0:
            if len(real_tmpl) == len(syn_template):
                r_norm = np.linalg.norm(real_tmpl - np.mean(real_tmpl))
                s_norm = np.linalg.norm(syn_template - np.mean(syn_template))
                if r_norm > 1e-6 and s_norm > 1e-6:
                    tmpl_corr = round(float(np.dot(real_tmpl - np.mean(real_tmpl), syn_template - np.mean(syn_template)) / (r_norm * s_norm)), 4)

        generation_params_records.append({
            "synthetic_id": syn_id,
            "phase": phase,
            "condition_type": condition_type,
            "source_record_id": rec_id,
            "source_patient_id": patient_id,
            "source_seizure_id": sz_id,
            "source_empirical_row": obs_idx,
            "source_sampling_rate_hz": source_fs,
            "generation_sampling_rate_hz": gen_fs,
            "target_mean_hr": target_mean_hr,
            "target_std_hr": target_std_hr,
            "target_sdnn": target_sdnn,
            "target_rmssd": target_rmssd,
            "target_relative_hf": target_spectral["Relative_High_Frequency_Power"],
            "target_qrs_dur": target_qrs_dur,
            "target_qtc": target_qtc,
            "target_r_amp": target_r_amp,
            "target_signal_std": target_sig_std,
            "generator": "neurokit2",
            "generator_method": sim_method,
            "generator_hr": round(target_mean_hr, 2),
            "initial_hr_std_estimate": rhythm_cal["initial_hr_std_estimate"],
            "selected_hr_std": selected_hr_std,
            "generator_hr_variability": selected_hr_std,
            "number_of_hr_candidates_evaluated": rhythm_cal["number_of_candidates_evaluated"],
            "achieved_sdnn": rhythm_cal["achieved_sdnn"],
            "sdnn_error_ms": rhythm_cal["sdnn_error"],
            "selected_noise": cal_noise,
            "generator_noise": cal_noise,
            "spectral_objective": spec_obj,
            "spectral_features_used": spec_features_used,
            "number_of_noise_candidates_evaluated": spectral_cal["number_of_noise_candidates_evaluated"],
            "duration_s": DURATION_SEC,
            "random_seed": sim_seed,
            "generation_status": "SUCCESS",
            "rhythm_calibration_status": "CALIBRATED",
            "spectral_calibration_status": spec_status,
            "limitations": "Morphology evaluated post-hoc; generator lacks independent QRS/QT programmatic equations"
        })

        feature_table_records.append({
            "synthetic_id": syn_id,
            "phase": phase,
            "condition_type": condition_type,
            "source_record_id": rec_id,
            "source_seizure_id": sz_id,
            "source_sampling_rate_hz": source_fs,
            "generation_sampling_rate_hz": gen_fs,
            "duration_s": DURATION_SEC,
            **syn_features
        })

        ach_hr = syn_features.get("Mean_HR_BPM", np.nan)
        ach_sdnn_re = syn_features.get("SDNN_ms", np.nan)
        ach_std_hr = syn_features.get("Std_HR_BPM", np.nan)
        ach_qrs = syn_features.get("QRS_Duration_Median_ms", np.nan)
        ach_qtc = syn_features.get("QTc_Median_ms", np.nan)
        ach_r_amp = syn_features.get("R_Amplitude_Median_mV", np.nan)
        ach_rel_hf_re = syn_features.get("Relative_High_Frequency_Power", np.nan)

        hr_abs_err = round(abs(ach_hr - target_mean_hr), 2) if np.isfinite(ach_hr) and np.isfinite(target_mean_hr) else np.nan
        hr_rel_err = round((hr_abs_err / target_mean_hr) * 100.0, 2) if np.isfinite(hr_abs_err) and target_mean_hr > 0 else np.nan
        sdnn_abs_err = round(abs(ach_sdnn_re - target_sdnn), 2) if np.isfinite(ach_sdnn_re) and np.isfinite(target_sdnn) else np.nan
        sdnn_rel_err = round((sdnn_abs_err / target_sdnn) * 100.0, 2) if np.isfinite(sdnn_abs_err) and target_sdnn > 0 else np.nan
        qrs_err = round(abs(ach_qrs - target_qrs_dur), 2) if np.isfinite(ach_qrs) and np.isfinite(target_qrs_dur) else np.nan
        qtc_err = round(abs(ach_qtc - target_qtc), 2) if np.isfinite(ach_qtc) and np.isfinite(target_qtc) else np.nan
        r_amp_err = round(abs(ach_r_amp - target_r_amp), 4) if np.isfinite(ach_r_amp) and np.isfinite(target_r_amp) else np.nan
        spec_abs_err = round(abs(ach_rel_hf_re - target_spectral["Relative_High_Frequency_Power"]), 6) if np.isfinite(ach_rel_hf_re) and np.isfinite(target_spectral["Relative_High_Frequency_Power"]) else np.nan

        comparison_records.append({
            "synthetic_id": syn_id,
            "phase": phase,
            "condition_type": condition_type,
            "source_record_id": rec_id,
            "source_seizure_id": sz_id,
            "source_sampling_rate_hz": source_fs,
            "generation_sampling_rate_hz": gen_fs,
            "target_mean_hr": target_mean_hr,
            "achieved_mean_hr": ach_hr,
            "hr_abs_error_bpm": hr_abs_err,
            "hr_rel_error_pct": hr_rel_err,
            "target_sdnn_ms": target_sdnn,
            "achieved_sdnn_ms": ach_sdnn_re,
            "sdnn_abs_error_ms": sdnn_abs_err,
            "sdnn_rel_error_pct": sdnn_rel_err,
            "initial_hr_std_estimate": rhythm_cal["initial_hr_std_estimate"],
            "selected_hr_std": selected_hr_std,
            "target_std_hr_bpm": target_std_hr,
            "achieved_std_hr_bpm": ach_std_hr,
            "target_qrs_dur_ms": target_qrs_dur,
            "achieved_qrs_dur_ms": ach_qrs,
            "qrs_abs_error_ms": qrs_err,
            "target_qtc_ms": target_qtc,
            "achieved_qtc_ms": ach_qtc,
            "qtc_abs_error_ms": qtc_err,
            "target_r_amp_mv": target_r_amp,
            "achieved_r_amp_mv": ach_r_amp,
            "r_amp_abs_error_mv": r_amp_err,
            "beat_template_correlation": tmpl_corr,
            "target_relative_hf": target_spectral["Relative_High_Frequency_Power"],
            "achieved_relative_hf": ach_rel_hf_re,
            "spectral_abs_error": spec_abs_err,
            "selected_noise": cal_noise,
            "spectral_objective": spec_obj,
            "spectral_features_used": spec_features_used
        })

        quality_report_records.append({
            "synthetic_id": syn_id,
            "source_record_id": rec_id,
            "source_seizure_id": sz_id,
            "phase": phase,
            "condition_type": condition_type,
            "source_sampling_rate_hz": source_fs,
            "generation_sampling_rate_hz": gen_fs,
            "generation_status": "SUCCESS",
            "rhythm_calibration_status": "CALIBRATED",
            "spectral_calibration_status": spec_status,
            "is_continuous": True,
            "no_chunk_boundaries": True,
            "boundary_discontinuity_status": "N/A (Continuous 60s Generation)",
            "max_sample_step_diff": round(max_step_diff, 5),
            "mean_sample_step_diff": round(mean_step_diff, 5),
            "signal_std": round(float(np.std(syn_signal)), 4),
            "detected_peaks": syn_features.get("Detected_Peaks", 0),
            "missing_required_inputs": "None",
            "failure_reason": "None",
            "limitations": "Morphology evaluated post-hoc against Exp 08 targets"
        })

        print(f"  [OK] {syn_id:38s} | HR: tgt={target_mean_hr:.1f} -> syn={ach_hr:.1f} BPM | "
              f"SDNN: tgt={target_sdnn:.1f} -> syn={ach_sdnn_re:.1f} ms | Noise: {cal_noise:.3f} | Corr: {tmpl_corr}", flush=True)

    # Backward-Compatible Class Summary Table (for Exp 10 & 11)
    for c_name in all_classes_order:
        sub_comp = [c for c in comparison_records if c["phase"] == c_name]
        sub_params = [p for p in generation_params_records if p["phase"] == c_name]
        if sub_comp:
            mean_tgt_hr = float(np.mean([c["target_mean_hr"] for c in sub_comp]))
            mean_tgt_sdnn = float(np.mean([c["target_sdnn_ms"] for c in sub_comp]))
            mean_ach_hr_std = float(np.mean([p["selected_hr_std"] for p in sub_params]))
            mean_noise = float(np.mean([p["selected_noise"] for p in sub_params]))
            src_recs = sorted(list(set(c["source_record_id"] for c in sub_comp)))
            src_fs_list = sorted(list(set(c["source_sampling_rate_hz"] for c in sub_comp)))
            is_ctrl = (c_name == "Hard_Negative_Arrhythmia")

            compat_class_records.append({
                "Clinical_Class": c_name,
                "Condition_Type": "cardiac_arrhythmia_control" if is_ctrl else "seizure_phase",
                "Target_Mean_HR_BPM": round(mean_tgt_hr, 2),
                "Empirical_SDNN_ms": round(mean_tgt_sdnn, 2),
                "Empirical_Std_HR_BPM": round(mean_ach_hr_std, 2),
                "Fitted_hr_std_param": round(mean_ach_hr_std, 2),
                "Fitted_noise_param": round(mean_noise, 4),
                "Synthetic_Signal_Samples": N_SAMPLES,
                "No_Chunk_Boundaries": True,
                "Boundary_Discontinuity_Aligned": "N/A (Continuous 60s Generation)",
                "Source_Sampling_Rate_Hz": ", ".join(str(f) for f in src_fs_list),
                "Generation_Sampling_Rate_Hz": float(FS),
                "Amplitude_Units": "Arbitrary Units (a.u.)",
                "Empirical_Source_Records": ", ".join(src_recs),
                "Generator_Status": "Empirically Calibrated (Optimization over Defined Search Spaces)"
            })

    # Save Artifacts
    df_gen_params = pd.DataFrame(generation_params_records)
    out_gen_params = os.path.join(OUTPUT_DIR, "09_empirical_generation_parameters.csv")
    df_gen_params.to_csv(out_gen_params, index=False)
    print(f"\n[Artifact 1] Empirical generation parameters saved: {out_gen_params} ({len(df_gen_params)} rows)", flush=True)

    out_npz = os.path.join(OUTPUT_DIR, "09_synthetic_ecg_signals.npz")
    np.savez_compressed(out_npz, **synthetic_signals)
    print(f"[Artifact 2] Synthetic ECG signals saved: {out_npz} ({len(synthetic_signals)} continuous signals)", flush=True)

    df_syn_feat = pd.DataFrame(feature_table_records)
    out_syn_feat = os.path.join(OUTPUT_DIR, "09_synthetic_feature_table.csv")
    df_syn_feat.to_csv(out_syn_feat, index=False)
    print(f"[Artifact 3] Synthetic feature table saved: {out_syn_feat} ({len(df_syn_feat)} rows)", flush=True)

    df_comp = pd.DataFrame(comparison_records)
    out_comp = os.path.join(OUTPUT_DIR, "09_empirical_vs_synthetic_parameters.csv")
    df_comp.to_csv(out_comp, index=False)
    print(f"[Artifact 4] Empirical vs synthetic comparison saved: {out_comp} ({len(df_comp)} rows)", flush=True)

    df_qual = pd.DataFrame(quality_report_records)
    out_qual = os.path.join(OUTPUT_DIR, "09_generation_quality_report.csv")
    df_qual.to_csv(out_qual, index=False)
    print(f"[Artifact 5] Generation quality report saved: {out_qual} ({len(df_qual)} rows)", flush=True)

    df_compat = pd.DataFrame(compat_class_records)
    out_compat = os.path.join(OUTPUT_DIR, "09_fitted_synthetic_signals.csv")
    df_compat.to_csv(out_compat, index=False)
    print(f"[Artifact 6] Backward-compatible class summary saved: {out_compat} ({len(df_compat)} classes)", flush=True)

    # Visualizations
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.4])
    fig.suptitle("Experiment 09: Empirically Calibrated Synthetic ECG Generation Engine\n"
                 "Calibrated Across Real PhysioNet SZDB Observations (Continuous 60s Dynamics)",
                 fontsize=12, fontweight="bold", y=0.995)

    colors = {
        "Interictal_Baseline": "steelblue",
        "Preictal_Prediction": "darkorange",
        "Ictal_Seizure": "purple",
        "Postictal_Recovery": "seagreen",
        "Hard_Negative_Arrhythmia": "crimson"
    }

    t_window = np.arange(int(6 * FS)) / float(FS)

    rep_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for ax, p_name in zip(rep_axes, seizure_phases):
        rep_key = next((k for k in synthetic_signals if p_name in k and "sz01" in k),
                       next((k for k in synthetic_signals if p_name in k), None))
        if rep_key:
            sig_chunk = synthetic_signals[rep_key][:int(6 * FS)]
            p_info = next(r for r in generation_params_records if r["synthetic_id"] == rep_key)
            ax.plot(t_window, sig_chunk, color=colors[p_name], linewidth=0.85)
            ax.set_title(f"Synthetic {p_name.replace('_', ' ')} ({p_info['source_record_id']} {p_info['source_seizure_id']})\n"
                         f"Target HR: {p_info['target_mean_hr']:.1f} BPM | Gen HR: {p_info['generator_hr']:.1f} | Calib Noise: {p_info['generator_noise']:.3f}",
                         fontsize=8.5, fontweight="bold")
            ax.set_ylabel("Amplitude (a.u.)", fontsize=8)
            ax.set_xlabel("Time (seconds)", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

    ax_hr = fig.add_subplot(gs[2, 0])
    all_phases_plot = [p for p in all_classes_order if p in df_comp["phase"].values]
    x_pos = np.arange(len(all_phases_plot))
    bar_width = 0.35

    tgt_hr_means = [df_comp[df_comp["phase"] == p]["target_mean_hr"].mean() for p in all_phases_plot]
    ach_hr_means = [df_comp[df_comp["phase"] == p]["achieved_mean_hr"].mean() for p in all_phases_plot]

    ax_hr.bar(x_pos - bar_width/2, tgt_hr_means, bar_width, label="Empirical Target HR", color="steelblue", alpha=0.85)
    ax_hr.bar(x_pos + bar_width/2, ach_hr_means, bar_width, label="Achieved Synthetic HR", color="darkorange", alpha=0.85)
    ax_hr.set_xticks(x_pos)

    x_labels = []
    for p in all_phases_plot:
        if p == "Hard_Negative_Arrhythmia":
            x_labels.append("Arrhythmia\n(Control)")
        else:
            x_labels.append(p.replace("_", "\n"))

    ax_hr.set_xticklabels(x_labels, fontsize=7.5)
    ax_hr.set_ylabel("Mean Heart Rate (BPM)", fontsize=9)
    ax_hr.set_title("Heart Rate Fidelity (Seizure Phases vs Arrhythmia Control)", fontsize=9.5, fontweight="bold")
    ax_hr.legend(fontsize=8)
    ax_hr.grid(True, linestyle="--", alpha=0.3)

    ax_sdnn = fig.add_subplot(gs[2, 1])
    tgt_sdnn_means = [df_comp[df_comp["phase"] == p]["target_sdnn_ms"].mean() for p in all_phases_plot]
    ach_sdnn_means = [df_comp[df_comp["phase"] == p]["achieved_sdnn_ms"].mean() for p in all_phases_plot]

    ax_sdnn.bar(x_pos - bar_width/2, tgt_sdnn_means, bar_width, label="Empirical Target SDNN", color="seagreen", alpha=0.85)
    ax_sdnn.bar(x_pos + bar_width/2, ach_sdnn_means, bar_width, label="Achieved Synthetic SDNN", color="plum", alpha=0.85)
    ax_sdnn.set_xticks(x_pos)
    ax_sdnn.set_xticklabels(x_labels, fontsize=7.5)
    ax_sdnn.set_ylabel("SDNN (ms)", fontsize=9)
    ax_sdnn.set_title("HRV SDNN Fidelity (Seizure Phases vs Arrhythmia Control)", fontsize=9.5, fontweight="bold")
    ax_sdnn.legend(fontsize=8)
    ax_sdnn.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_img = os.path.join(OUTPUT_DIR, "09_empirical_synthetic_fitting.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Artifact 7] Visualizations saved: {out_img}\n", flush=True)

    print("=== Experiment 09 Complete: Empirically Calibrated Generation Successfully Finished ===", flush=True)


if __name__ == "__main__":
    run_experiment_09()
