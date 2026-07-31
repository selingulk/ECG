"""
Experiment 04: Native torch_ecg Augmentation Capabilities & Clinical Risk Audit
=================================================================================
Research Question: How do native torch_ecg.augmenters classes (BaselineWanderAugmenter,
RandomRenormalize, RandomFlip, RandomMasking, CutMix) transform ECG tensors,
and what clinical diagnostic risks do they pose?

Fixes Applied (Addressing Credibility Risk 1):
  - Uses native torch_ecg.augmenters classes directly imported from torch_ecg.
  - Transforms PyTorch Tensors natively using torch_ecg's forward augmentation pipeline.
  - Wrapped in if __name__ == '__main__': to support Windows multiprocessing.

Classes Tested:
  1. torch_ecg.augmenters.BaselineWanderAugmenter
  2. torch_ecg.augmenters.RandomRenormalize
  3. torch_ecg.augmenters.RandomFlip
  4. torch_ecg.augmenters.RandomMasking
  5. torch_ecg.augmenters.CutMix
"""

import os
import sys
import ctypes
import csv
import numpy as np
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
    N = FS * DURATION
    time = np.linspace(0, DURATION, N)

    # Base 10-second reference signal converted to PyTorch tensor (batch_size=1, leads=1, samples=5000)
    clean_ecg_np = nk.ecg_simulate(duration=DURATION, sampling_rate=FS, heart_rate=70, noise=0.01, random_state=42)
    sig_tensor = torch.from_numpy(clean_ecg_np.astype(np.float32))[None, None, :]
    dummy_label = torch.tensor([[1.0, 0.0]])

    # Instantiate native torch_ecg.augmenters classes
    augmenter_instances = [
        ("BaselineWanderAugmenter", A.BaselineWanderAugmenter(fs=FS, prob=1.0), "red",
         "Risk: Obscures low-amplitude P-waves and small Q/S deflections; may cause false peak detections.", False),
        ("RandomRenormalize", A.RandomRenormalize(prob=1.0), "orange",
         "HIGH RISK: Rescales signal amplitude; alters voltage criteria for Left Ventricular Hypertrophy (Sokolow-Lyon index).", False),
        ("RandomFlip", A.RandomFlip(prob=1.0), "magenta",
         "HIGH RISK: Inverts signal polarity (+/-); mimics dextrocardia or reversed lead placement.", False),
        ("RandomMasking", A.RandomMasking(fs=FS, prob=1.0), "gray",
         "Moderate Risk: Zeroes out signal segments; simulates electrode disconnection to test model robustness.", False),
        ("CutMix", A.CutMix(prob=1.0), "purple",
         "HIGH RISK: Slices and mixes segments from different signals; distorts beat continuity and rhythm.", True)
    ]

    fig, axes = plt.subplots(len(augmenter_instances), 1, figsize=(12, 14), sharex=True)
    fig.suptitle("Experiment 04: Native torch_ecg.augmenters Class Capabilities & Clinical Risk Audit\n"
                 "Transformations executed via native torch_ecg.augmenters module on PyTorch Tensors",
                 fontsize=12, fontweight='bold', y=0.995)

    csv_records = []

    for idx, (name, aug_obj, col, risk_desc, needs_label) in enumerate(augmenter_instances):
        # Execute native torch_ecg.augmenters forward call
        if needs_label:
            aug_tensor, _ = aug_obj(sig_tensor.clone(), dummy_label)
        else:
            aug_tensor, _ = aug_obj(sig_tensor.clone(), None)
            
        aug_np = aug_tensor[0, 0].detach().cpu().numpy()

        ax = axes[idx]
        ax.plot(time, clean_ecg_np, color='black', alpha=0.4, linewidth=0.8, label="Original Reference")
        t_plot = np.linspace(0, DURATION, len(aug_np))
        ax.plot(t_plot, aug_np, color=col, linewidth=1.0, label=f"torch_ecg.{name}")
        ax.set_title(f"{idx+1}. torch_ecg.augmenters.{name}", fontsize=9, fontweight='bold')
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=7.5)
        ax.grid(True, linestyle="--", alpha=0.3)

        ax.text(0.01, 0.08, f"CLINICAL DIAGNOSTIC RISK: {risk_desc}",
                transform=ax.transAxes, fontsize=7.5, fontweight='bold', color='darkred',
                bbox=dict(boxstyle="round", facecolor="#fff0f0", edgecolor="red", alpha=0.9))

        csv_records.append({
            "Augmentation_Method": name,
            "torch_ecg_Class": f"torch_ecg.augmenters.{name}",
            "Diagnostic_Risk_Level": "HIGH" if "HIGH RISK" in risk_desc else ("Moderate" if "Moderate" in risk_desc else "Low"),
            "Clinical_Impact": risk_desc,
            "Preserves_Diagnostic_Labels": "No" if "HIGH RISK" in risk_desc else "Yes"
        })

    axes[-1].set_xlabel("Time (seconds)")
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
