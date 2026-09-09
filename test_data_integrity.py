"""
Test Suite: ECG Research Repository Data Integrity & Scientific Correctness Audit
===================================================================================
Verifies the critical integrity and provenance rules across Experiments 08-11:
  1. Experiment 08 contains zero calls to nk.ecg_simulate.
  2. Both sz01 and sz04 are attempted and represented in outputs.
  3. No fallback HR/HRV constants (75 bpm, 30 ms, 25 ms) are used as measurement fillers.
  4. Real rows include data_origin = 'real_clinical_physionet'.
  5. Missing/failed feature extraction strictly produces NaN and failure flags.
  6. window_end_s > window_start_s for every extracted phase window.
  7. n_samples strictly matches window_duration_s * sampling_rate.
  8. Sampling rate metadata is dynamically present and preserved.
  9. Experiment 09 reads the primary Experiment 08 empirical output table.
  10. All required output files exist after pipeline execution.
  11. MIT-BIH records are verified with annotation evidence rather than blindly labeled.
"""

import os
import re
import pytest
import numpy as np
import pandas as pd

OUTPUT_DIR = "outputs"
EXP08_SCRIPT = "Experiments/08_real_seizure_ecg_extraction.py"
EXP09_SCRIPT = "Experiments/09_empirical_synthetic_fitting.py"
EXP10_SCRIPT = "Experiments/10_quantitative_real_vs_synthetic_validation.py"
EXP11_SCRIPT = "Experiments/11_downstream_seizure_classifier.py"


def test_rule_1_exp08_no_ecg_simulate():
    """Rule 1: Experiment 08 must contain zero calls to nk.ecg_simulate."""
    with open(EXP08_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    # Check for any call to ecg_simulate
    matches = re.findall(r"ecg_simulate\s*\(", content)
    assert len(matches) == 0, f"Found {len(matches)} calls to ecg_simulate in {EXP08_SCRIPT}!"


def test_rule_2_both_sz01_and_sz04_attempted():
    """Rule 2: Both sz01 and sz04 must be attempted and present in the primary feature table."""
    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    assert os.path.exists(table_path), f"Missing {table_path}"
    df = pd.read_csv(table_path)
    records = set(df["record_id"].dropna().unique())
    assert "sz01" in records, "Record sz01 is missing from feature table!"
    assert "sz04" in records, "Record sz04 is missing from feature table!"


def test_rule_3_no_fallback_constants():
    """Rule 3: No default fallback constants (75.0, 30.0, 25.0) used as replacement fillers."""
    with open(EXP08_SCRIPT, "r", encoding="utf-8") as f:
        code = f.read()
    assert "75.0, 30.0, 25.0" not in code, "Found legacy default replacement tuple (75.0, 30.0, 25.0) in code!"

    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df = pd.read_csv(table_path)
    for _, row in df.iterrows():
        # Ensure that no row has exactly the triplet (75.0, 30.0, 25.0)
        triplet = (row.get("Mean_HR_BPM"), row.get("SDNN_ms"), row.get("RMSSD_ms"))
        assert triplet != (75.0, 30.0, 25.0), f"Row {row.get('record_id')} has fabricated default triplet!"


def test_rule_4_real_rows_data_origin():
    """Rule 4: All real rows must include data_origin = 'real_clinical_physionet'."""
    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df = pd.read_csv(table_path)
    assert "data_origin" in df.columns, "Column 'data_origin' missing!"
    for origin in df["data_origin"]:
        assert origin == "real_clinical_physionet", f"Invalid data_origin found: {origin}"


def test_rule_5_missing_feature_produces_nan():
    """Rule 5: Missing or failed feature extraction must produce NaN and validity flags, never fake numbers."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp08", EXP08_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Test with empty signal or flatline signal
    cleaned_flat = np.zeros(2000, dtype=np.float64)
    rhythm, rr_recs, peaks = mod.extract_rhythm_and_hrv(cleaned_flat, 200, {"dataset": "test"})
    assert np.isnan(rhythm["Mean_HR_BPM"]), "Mean_HR_BPM should be NaN on failure!"
    assert np.isnan(rhythm["SDNN_ms"]), "SDNN_ms should be NaN on failure!"
    assert np.isnan(rhythm["RMSSD_ms"]), "RMSSD_ms should be NaN on failure!"
    assert rhythm["peak_detection_ok"] is False, "peak_detection_ok should be False!"

    # Morphology on zero peaks
    morph, template, n_beats = mod.extract_morphology_and_templates(cleaned_flat, [], 200)
    assert np.isnan(morph["QRS_Duration_Median_ms"]), "QRS_Duration should be NaN on failure!"
    assert morph["morphology_valid"] is False, "morphology_valid should be False on failure!"


def test_rule_6_window_bounds_valid():
    """Rule 6: window_end_s > window_start_s for all extracted rows."""
    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df = pd.read_csv(table_path)
    assert "window_start_s" in df.columns and "window_end_s" in df.columns
    for _, row in df.iterrows():
        assert row["window_end_s"] > row["window_start_s"], f"Invalid window in row: {row['window_start_s']} >= {row['window_end_s']}"


def test_rule_7_n_samples_matches_window():
    """Rule 7: n_samples matches the selected window duration * sampling rate."""
    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df = pd.read_csv(table_path)
    for _, row in df.iterrows():
        expected_samples = int(round(row["window_duration_s"] * row["original_sampling_rate_hz"]))
        assert abs(row["n_samples"] - expected_samples) <= 1, f"n_samples mismatch: {row['n_samples']} vs {expected_samples}"


def test_rule_8_sampling_rate_metadata_present():
    """Rule 8: Dynamic sampling-rate metadata is present and valid."""
    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df = pd.read_csv(table_path)
    assert "original_sampling_rate_hz" in df.columns
    assert "analysis_sampling_rate_hz" in df.columns
    for fs in df["original_sampling_rate_hz"]:
        assert fs in [200.0, 360.0], f"Unexpected fs value: {fs}"


def test_rule_9_exp09_reads_exp08_output():
    """Rule 9: Experiment 09 reads the intended Experiment 08 output table."""
    with open(EXP09_SCRIPT, "r", encoding="utf-8") as f:
        code = f.read()
    assert "08_empirical_ecg_feature_table.csv" in code, "Experiment 09 does not reference 08_empirical_ecg_feature_table.csv!"


def test_rule_10_output_filenames_exist():
    """Rule 10: All required output files exist after pipeline execution."""
    expected_files = [
        "outputs/08_empirical_ecg_feature_table.csv",
        "outputs/08_rr_intervals.csv",
        "outputs/08_beat_templates.npz",
        "outputs/08_data_quality_report.csv",
        "outputs/08_real_seizure_phases.png",
        "outputs/08_real_feature_distributions.png",
        "outputs/08_empirical_seizure_parameters.csv",
        "outputs/09_fitted_synthetic_signals.csv",
        "outputs/09_empirical_synthetic_fitting.png",
        "outputs/10_quantitative_validation_metrics.csv",
        "outputs/10_real_vs_synthetic_validation.png",
        "outputs/11_downstream_seizure_classifier_summary.csv",
        "outputs/11_downstream_seizure_classifier_benchmark.png"
    ]
    for p in expected_files:
        assert os.path.exists(p), f"Missing required output file: {p}"


def test_rule_11_no_silent_unverified_mitbih_label():
    """Rule 11: No function silently labels MIT-BIH without annotation evidence."""
    table_path = os.path.join(OUTPUT_DIR, "08_empirical_ecg_feature_table.csv")
    df = pd.read_csv(table_path)
    mit_rows = df[df["dataset"] == "mitdb"]
    assert len(mit_rows) > 0, "No MIT-BIH rows found!"
    for _, r in mit_rows.iterrows():
        assert "Annotation_Evidence" in r and pd.notna(r["Annotation_Evidence"]), "MIT-BIH row missing annotation evidence!"
        assert len(str(r["Annotation_Evidence"])) > 5, "MIT-BIH annotation evidence string empty or trivial!"


# ===========================================================================
# Experiment 09 Scientific Integrity & Calibration Audit Suite
# ===========================================================================

def test_exp09_no_fallback_defaults_in_code():
    """Exp 09 Integrity Rule 1: No fallback heart_rate_std = 3.0 or [1, 3, 6] in Exp 09 code."""
    with open(EXP09_SCRIPT, "r", encoding="utf-8") as f:
        code = f.read()
    assert "best_hr_std = 3.0" not in code, "Found legacy fallback 'best_hr_std = 3.0' in Exp 09!"
    assert "[1.0, 3.0, 6.0]" not in code, "Found legacy fallback candidate grid '[1.0, 3.0, 6.0]' in Exp 09!"
    assert "[1, 3, 6]" not in code and "[1,3,6]" not in code, "Found fallback list [1, 3, 6] in Exp 09!"


def test_exp09_missing_rhythm_inputs_handling():
    """Exp 09 Integrity Rule 2: Missing rhythm inputs cause skip/not-calibrated, never fake defaults."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp09", EXP09_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Test with NaN target_hr
    res1 = mod.calibrate_rhythm_hrv(target_hr=np.nan, target_sdnn=50.0)
    assert res1["calibrated"] is False, "Should not calibrate when target_hr is NaN!"
    assert res1["status"] == "MISSING_REQUIRED_INPUTS"
    assert np.isnan(res1["selected_hr_std"]), "Should not fabricate selected_hr_std when target_hr is NaN!"

    # Test with NaN target_sdnn
    res2 = mod.calibrate_rhythm_hrv(target_hr=70.0, target_sdnn=np.nan)
    assert res2["calibrated"] is False, "Should not calibrate when target_sdnn is NaN!"
    assert res2["status"] == "MISSING_REQUIRED_INPUTS"
    assert np.isnan(res2["selected_hr_std"]), "Should not fabricate selected_hr_std when target_sdnn is NaN!"

    # Test with non-positive inputs
    res3 = mod.calibrate_rhythm_hrv(target_hr=-10.0, target_sdnn=50.0)
    assert res3["calibrated"] is False
    assert res3["status"] == "MISSING_REQUIRED_INPUTS"


def test_exp09_search_bounds_configuration():
    """Exp 09 Integrity Rule 3: Search spaces are explicit configuration constants, not physiological claims."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp09", EXP09_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "HR_STD_SEARCH_BOUNDS"), "HR_STD_SEARCH_BOUNDS must be defined at module level!"
    assert hasattr(mod, "HR_STD_LOCAL_SEARCH_FACTORS"), "HR_STD_LOCAL_SEARCH_FACTORS must be defined at module level!"
    assert hasattr(mod, "NOISE_SEARCH_GRID"), "NOISE_SEARCH_GRID must be defined at module level!"

    bounds = mod.HR_STD_SEARCH_BOUNDS
    assert isinstance(bounds, tuple) and len(bounds) == 2
    assert bounds[0] >= 0.1 and bounds[1] <= 25.0

    # Ensure comments clarify these are simulator search ranges, not clinical measurements
    with open(EXP09_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read()
    assert "simulator search bounds" in content.lower() or "simulator parameter search" in content.lower(), \
        "Exp 09 must document that search bounds are simulator parameter configurations, not clinical measurements!"


def test_exp09_no_unsupported_zero_heuristic_claims():
    """Exp 09 Integrity Rule 4: 'Zero heuristic' and 'heuristic-free' claims must be removed."""
    with open(EXP09_SCRIPT, "r", encoding="utf-8") as f:
        content = f.read().lower()
    assert "zero heuristic" not in content, "Found unsupported claim 'zero heuristic' in Exp 09!"
    assert "zero-heuristic" not in content, "Found unsupported claim 'zero-heuristic' in Exp 09!"
    assert "heuristic-free" not in content, "Found unsupported claim 'heuristic-free' in Exp 09!"


def test_exp09_sampling_rate_provenance():
    """Exp 09 Integrity Rule 5: Both source_sampling_rate_hz and generation_sampling_rate_hz are preserved."""
    params_path = os.path.join(OUTPUT_DIR, "09_empirical_generation_parameters.csv")
    report_path = os.path.join(OUTPUT_DIR, "09_generation_quality_report.csv")
    assert os.path.exists(params_path), f"Missing {params_path}"
    assert os.path.exists(report_path), f"Missing {report_path}"

    df_p = pd.read_csv(params_path)
    df_q = pd.read_csv(report_path)

    assert "source_sampling_rate_hz" in df_p.columns, "source_sampling_rate_hz missing from parameters table!"
    assert "generation_sampling_rate_hz" in df_p.columns, "generation_sampling_rate_hz missing from parameters table!"
    assert "source_sampling_rate_hz" in df_q.columns, "source_sampling_rate_hz missing from quality report!"
    assert "generation_sampling_rate_hz" in df_q.columns, "generation_sampling_rate_hz missing from quality report!"

    # Standardized generation is 200 Hz
    for fs in df_p["generation_sampling_rate_hz"]:
        assert fs == 200.0, f"Unexpected generation_sampling_rate_hz: {fs}"

    # Source rates contain both 200 Hz (SZDB) and 360 Hz (MIT-BIH)
    src_rates = set(df_p["source_sampling_rate_hz"].unique())
    assert 200.0 in src_rates, "Source sampling rate 200.0 Hz missing!"


def test_exp09_continuous_generation_continuity_reporting():
    """Exp 09 Integrity Rule 6: Continuous generation does not claim repaired chunk boundaries."""
    report_path = os.path.join(OUTPUT_DIR, "09_generation_quality_report.csv")
    compat_path = os.path.join(OUTPUT_DIR, "09_fitted_synthetic_signals.csv")
    assert os.path.exists(report_path)
    assert os.path.exists(compat_path)

    df_q = pd.read_csv(report_path)
    df_c = pd.read_csv(compat_path)

    for _, r in df_q.iterrows():
        if r.get("generation_status") == "SUCCESS":
            assert r["is_continuous"] is True or r["is_continuous"] == "True"
            assert r["no_chunk_boundaries"] is True or r["no_chunk_boundaries"] == "True"
            assert "repaired" not in str(r.get("boundary_discontinuity_status", "")).lower()

    for _, r in df_c.iterrows():
        assert r["No_Chunk_Boundaries"] is True or r["No_Chunk_Boundaries"] == "True"
        align_str = str(r["Boundary_Discontinuity_Aligned"])
        assert "N/A" in align_str or "Continuous" in align_str, \
            f"Boundary_Discontinuity_Aligned should reflect N/A or Continuous, found: {align_str}"


def test_exp09_arrhythmia_separated_from_seizure_phases():
    """Exp 09 Integrity Rule 7: Arrhythmia control is labeled as cardiac_arrhythmia_control, not a seizure phase."""
    params_path = os.path.join(OUTPUT_DIR, "09_empirical_generation_parameters.csv")
    report_path = os.path.join(OUTPUT_DIR, "09_generation_quality_report.csv")
    compat_path = os.path.join(OUTPUT_DIR, "09_fitted_synthetic_signals.csv")

    df_p = pd.read_csv(params_path)
    df_q = pd.read_csv(report_path)
    df_c = pd.read_csv(compat_path)

    assert "condition_type" in df_p.columns
    assert "condition_type" in df_q.columns
    assert "Condition_Type" in df_c.columns

    # Verify MIT-BIH / Hard_Negative_Arrhythmia rows
    arrhythmia_p = df_p[df_p["phase"] == "Hard_Negative_Arrhythmia"]
    for _, r in arrhythmia_p.iterrows():
        assert r["condition_type"] == "cardiac_arrhythmia_control", \
            f"Arrhythmia row has invalid condition_type: {r['condition_type']}"

    # Verify seizure phase rows
    seizure_p = df_p[df_p["phase"] != "Hard_Negative_Arrhythmia"]
    for _, r in seizure_p.iterrows():
        assert r["condition_type"] == "seizure_phase", \
            f"Seizure row has invalid condition_type: {r['condition_type']}"


def test_exp09_truthful_calibration_status():
    """Exp 09 Integrity Rule 8: Calibration status is CALIBRATED only when empirical calibration succeeded."""
    report_path = os.path.join(OUTPUT_DIR, "09_generation_quality_report.csv")
    params_path = os.path.join(OUTPUT_DIR, "09_empirical_generation_parameters.csv")

    df_q = pd.read_csv(report_path)
    df_p = pd.read_csv(params_path)

    for _, r in df_q.iterrows():
        if r.get("generation_status") == "SUCCESS":
            assert r["rhythm_calibration_status"] == "CALIBRATED"
        else:
            assert r["rhythm_calibration_status"] != "CALIBRATED"

    for _, r in df_p.iterrows():
        if r["rhythm_calibration_status"] == "CALIBRATED":
            assert np.isfinite(r["target_mean_hr"]) and r["target_mean_hr"] > 0
            assert np.isfinite(r["target_sdnn"]) and r["target_sdnn"] > 0
            assert np.isfinite(r["selected_hr_std"]) and r["selected_hr_std"] > 0
            assert np.isfinite(r["achieved_sdnn"]) and r["achieved_sdnn"] > 0
            assert np.isfinite(r["sdnn_error_ms"])


def test_exp08_exp09_feature_extraction_consistency():
    """Exp 09 Integrity Rule 9: Feature extraction definitions in Exp 09 match Exp 08."""
    import importlib.util

    spec08 = importlib.util.spec_from_file_location("exp08", EXP08_SCRIPT)
    mod08 = importlib.util.module_from_spec(spec08)
    spec08.loader.exec_module(mod08)

    spec09 = importlib.util.spec_from_file_location("exp09", EXP09_SCRIPT)
    mod09 = importlib.util.module_from_spec(spec09)
    spec09.loader.exec_module(mod09)

    # Test spectral features consistency on identical test signal
    np.random.seed(42)
    t = np.linspace(0, 10, 2000)
    test_sig = np.sin(2 * np.pi * 1.5 * t) + 0.2 * np.sin(2 * np.pi * 35.0 * t) + 0.05 * np.random.randn(2000)

    spec08_res = mod08.compute_spectral_features(test_sig, 200)
    spec09_res = mod09.compute_spectral_features(test_sig, 200)

    for key in ["Relative_High_Frequency_Power", "Spectral_Centroid_Hz", "Baseline_Wander_Band_Power", "ECG_Dominant_Band_Power"]:
        val08 = spec08_res[key]
        val09 = spec09_res[key]
        assert abs(val08 - val09) < 1e-4, f"Spectral mismatch for {key}: Exp08={val08} vs Exp09={val09}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

