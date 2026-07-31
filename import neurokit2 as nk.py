import neurokit2 as nk
import matplotlib.pyplot as plt
import numpy as np

# Baseline configuration
ecg = nk.ecg_simulate(
    duration=10,
    sampling_rate=500,
    heart_rate=70,
    noise=0.01,
    random_state=42
)

# Convert samples to time (seconds) for easier reading
time = np.linspace(0, 10, len(ecg)) 

plt.figure(figsize=(10, 4))
plt.plot(time, ecg)
plt.title("Baseline Simulated ECG (70 BPM)")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

# --- SAVE THE PLOT HERE ---
plt.savefig("baseline_ecg.png", dpi=300, bbox_inches="tight")

plt.show()