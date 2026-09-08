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


if __name__ == "__main__":
    pytest.main(["-v", __file__])
