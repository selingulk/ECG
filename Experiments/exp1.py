import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np

# The three heart rates required by your experiment instructions
heart_rates = [50, 70, 100]

# Setup a figure with 3 subplots stacked vertically
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

for i, hr in enumerate(heart_rates):
    # Simulate the ECG changing ONLY the heart_rate parameter
    ecg = nk.ecg_simulate(
        duration=10,
        sampling_rate=500,
        heart_rate=hr,
        noise=0.01,
        random_state=42
    )
    
    # Convert samples to time (seconds)
    time = np.linspace(0, 10, len(ecg))
    
    # Plotting each one
    axes[i].plot(time, ecg, color='tab:blue')
    axes[i].set_title(f"Heart Rate: {hr} BPM")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid(True, linestyle='--', alpha=0.5)

# X-axis label only on the bottom plot since they share the axis
axes[-1].set_xlabel("Time (seconds)")

plt.tight_layout()

# Save the comparative figure for your report
plt.savefig("experiment_1_heart_rate.png", dpi=300, bbox_inches="tight")
plt.show()