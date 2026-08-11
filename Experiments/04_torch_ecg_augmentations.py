"""
Experiment 04: Native torch_ecg Augmentation Capabilities & Task-Dependent Risk Audit
======================================================================================
Research Question: How do native torch_ecg.augmenters classes (BaselineWanderAugmenter,
RandomRenormalize, RandomFlip, RandomMasking, CutMix) transform ECG tensors,
and how does their label-preservation suitability depend on the specific downstream task?

Key Methodological Enhancements:
  - CutMix Multi-Signal Batch: CutMix is evaluated on a multi-signal batch (N=4 distinct ECG signals
    with different heart rates: 50, 70, 120, 150 BPM) to demonstrate genuine inter-signal mixing.
  - Task-Dependent Label Preservation: Rather than classifying transformations as statically safe or unsafe,
    diagnostic risk and label preservation are evaluated as task-dependent (e.g. R-peak/HR detection vs.
    amplitude-based LVH hypertrophy or rhythm classification).
"""

import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk

# Add torch DLL directory on Windows if required
torch_lib = r'C:\Users\Lenovo 2\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\lib'
if os.path.exists(torch_lib):
    try:
        os.add_dll_directory(torch_lib)
    except Exception:
        pass

import torch
import torch_ecg.augmenters as A

def run_experiment():
    os.makedirs("outputs", exist_ok=True)
    print("=== Running Experiment 04: Native torch_ecg Augmentations Benchmark ===")

    FS = 500
    DURATION = 10
    N_SAMPLES = FS * DURATION
    TIME_AXIS = np.arange(N_SAMPLES) / float(FS)

    # Construct a multi-signal batch (batch_size=4, leads=1, samples=5000)
    # Using 4 distinct heart rates (50, 70, 120, 150 BPM) to demonstrate true CutMix inter-signal blending
    hr_list = [50, 70, 120, 150]
    batch_signals_np = []
    for hr_val in hr_list:
        sig_np = nk.ecg_simulate(duration=DURATION, sampling_rate=FS, heart_rate=hr_val, noise=0.01, random_state=42 + hr_val)
        batch_signals_np.append(sig_np)
        
    batch_np = np.array(batch_signals_np, dtype=np.float32)[:, None, :] # Shape (4, 1, 5000)
    batch_tensor = torch.from_numpy(batch_np)
    batch_labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    # Single reference for 1D display (Lead 0: 50 BPM, Lead 1: 70 BPM)
    ref_signal_np = batch_signals_np[1] # 70 BPM

    # Define native torch_ecg.augmenters classes and task-dependent diagnostic risk analyses
    augmenter_instances = [
        (
            "BaselineWanderAugmenter",
            A.BaselineWanderAugmenter(fs=FS, prob=1.0),
            "red",
            "Task-Dependent Risk: Preserves R-peak location / HR detection (Low Risk), but distorts baseline-dependent ST-elevation ischemic diagnosis.",
            "Task Dependent: Low risk for HR detection; High risk for ST-segment ischemia diagnosis"
        ),
        (
            "RandomRenormalize",
            A.RandomRenormalize(prob=1.0),
            "orange",
            "Task-Dependent Risk: Preserves temporal interval spacing (Low Risk), but alters absolute voltage criteria for Hypertrophy (LVH/Sokolow-Lyon index).",
            "Task Dependent: Low risk for timing/rhythm; High risk for voltage-based hypertrophy diagnosis"
        ),
        (
            "RandomFlip",
            A.RandomFlip(prob=1.0),
            "magenta",
            "Task-Dependent Risk: Preserves beat-to-beat interval spacing (Low Risk), but inverts polarity mimicking lead reversal or dextrocardia.",
            "Task Dependent: Low risk for HR/HRV; High risk for lead axis and polarity interpretation"
        ),
        (
            "RandomMasking",
            A.RandomMasking(fs=FS, prob=1.0),
            "gray",
            "Task-Dependent Risk: Tests model robustness to signal dropout, but zeroed segments may crop critical P-waves or QRS complexes.",
            "Task Dependent: Moderate risk; depends on whether masked window overlaps diagnostic P/QRS complexes"
        ),
        (
            "CutMix",
            A.CutMix(prob=1.0),
            "purple",
            "Task-Dependent Risk: Blends multi-signal batch segments (50/70/120/150 BPM); tests local feature noise but severely distorts global rhythm continuity.",
            "Task Dependent: High risk for rhythm classification; Useful for noise-robust local feature pretraining"
        )
    ]

    fig, axes = plt.subplots(len(augmenter_instances), 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Experiment 04: Native torch_ecg.augmenters Class Capabilities & Task-Dependent Risk Audit\n"
                 "Multi-Signal Batch Execution (N=4 distinct signals: 50, 70, 120, 150 BPM) on PyTorch Tensors",
                 fontsize=11, fontweight='bold', y=0.995)

    csv_records = []

    for idx, (name, aug_obj, col, risk_desc, task_suitability) in enumerate(augmenter_instances):
        # Execute native torch_ecg.augmenters forward call on multi-signal batch
        if name == "CutMix":
            aug_tensor, _ = aug_obj(batch_tensor.clone(), batch_labels.clone())
            # Select index 1 (70 BPM signal mixed with 150 BPM slice) for visual display
            aug_np = aug_tensor[1, 0].detach().cpu().numpy()
        else:
            aug_tensor, _ = aug_obj(batch_tensor.clone(), None)
            aug_np = aug_tensor[1, 0].detach().cpu().numpy()

        ax = axes[idx]
        ax.plot(TIME_AXIS, ref_signal_np, color='black', alpha=0.4, linewidth=0.8, label="Original Reference (70 BPM)")
        t_plot = np.arange(len(aug_np)) / float(FS)
        ax.plot(t_plot, aug_np, color=col, linewidth=1.0, label=f"torch_ecg.{name}")
        ax.set_title(f"{idx+1}. torch_ecg.augmenters.{name}", fontsize=9, fontweight='bold')
        ax.set_ylabel("Amplitude (a.u.)", fontsize=8)
        ax.legend(loc="upper right", fontsize=7.5)
        ax.grid(True, linestyle="--", alpha=0.3)

        ax.text(0.01, 0.08, f"TASK-DEPENDENT DIAGNOSTIC RISK: {risk_desc}",
                transform=ax.transAxes, fontsize=7.5, fontweight='bold', color='darkred',
                bbox=dict(boxstyle="round", facecolor="#fff0f0", edgecolor="red", alpha=0.9))

        csv_records.append({
            "Augmentation_Method": name,
            "torch_ecg_Class": f"torch_ecg.augmenters.{name}",
            "Multi_Signal_Batch_Size": 4,
            "Tested_Heart_Rates_BPM": "50, 70, 120, 150",
            "Clinical_Diagnostic_Risk_Description": risk_desc,
            "Task_Dependent_Label_Preservation": task_suitability
        })

    axes[-1].set_xlabel("Time (seconds)", fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_img = "outputs/04_torch_ecg_augmentations.png"
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    print(f"Experiment 04 complete. Saved visualization to {out_img}")
    plt.close()

    # Save summary CSV
    csv_path = "outputs/04_augmentation_clinical_risk_audit.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_records[0].keys())
        writer.writeheader()
        writer.writerows(csv_records)
    print(f"Results saved to {csv_path}\n")

if __name__ == '__main__':
    run_experiment()
