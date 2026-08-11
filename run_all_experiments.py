"""
Master Runner Script: Comprehensive ECG Benchmark (11 Experiments)
====================================================================
Executes all 11 experimental benchmarks in sequence, generating publication-ready
visualizations and structured summary CSV tables under outputs/.

Usage:
  python run_all_experiments.py
"""

import os
import sys
import subprocess
import time

experiments = [
    ("01_generation_parameter_sweeps.py", "Exp 01: Systematic Generation Parameter Sweeps (NeuroKit2 & physioKIT)"),
    ("02_physiokit_arrhythmia_presets.py", "Exp 02: Evaluation of physioKIT Presets & Terminology Audit"),
    ("03_dynamic_ictal_surge_simulation.py", "Exp 03: Proof-of-Concept Seizure Simulation with Boundary Alignment"),
    ("04_torch_ecg_augmentations.py", "Exp 04: Native torch_ecg Multi-Signal CutMix & Task-Dependent Risk"),
    ("05_signal_acquisition_comparison.py", "Exp 05: Signal Acquisition & Data Representation Audit (mV vs a.u.)"),
    ("06_noise_filtering_snr_benchmark.py", "Exp 06: Quantitative Filtering Benchmark across Target SNRs"),
    ("07_qrs_detection_multi_record.py", "Exp 07: R-Peak Detection Benchmark across 10 MIT-BIH Records"),
    ("08_real_seizure_ecg_extraction.py", "Exp 08: Real Seizure ECG Stage Parameter Extraction Pipeline"),
    ("09_empirical_synthetic_fitting.py", "Exp 09: Empirical Real-Fitted Synthetic ECG Generation"),
    ("10_quantitative_real_vs_synthetic_validation.py", "Exp 10: Quantitative Feature Distance Metrics (Wasserstein / PSD MSE)"),
    ("11_downstream_seizure_classifier.py", "Exp 11: Downstream AI Seizure Detection Classifier Benchmark")
]

print("=========================================================================")
print("          STARTING COMPREHENSIVE ECG BENCHMARK RUNNER (11 EXPS)         ")
print("=========================================================================\n")

start_total = time.time()
successful = 0

for idx, (script_name, description) in enumerate(experiments, 1):
    script_path = os.path.join("Experiments", script_name)
    print(f"[{idx}/11] Running {description}...")
    
    t0 = time.time()
    res = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    elapsed = time.time() - t0
    
    if res.returncode == 0:
        print(f"  [PASS] PASSED ({elapsed:.2f}s)\n")
        successful += 1
    else:
        print(f"  [FAIL] FAILED ({elapsed:.2f}s)")
        print(f"    Error output:\n{res.stderr}\n")

total_elapsed = time.time() - start_total

print("=========================================================================")
print(f"BENCHMARK COMPLETE: {successful}/{len(experiments)} Experiments Executed Successfully.")
print(f"Total Elapsed Time: {total_elapsed:.2f} seconds")
print("Outputs directory: outputs/")
print("=========================================================================")
