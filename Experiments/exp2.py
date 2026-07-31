import neurokit2 as nk
import matplotlib.pyplot as plt

# 1. Low HRV (Steady, robotic heartbeat)
ecg_steady = nk.ecg_simulate(duration=10, heart_rate=70, heart_rate_std=1, random_state=42)

# 2. High HRV (Irregular, variable heartbeat)
ecg_variable = nk.ecg_simulate(duration=10, heart_rate=70, heart_rate_std=20, random_state=42)

# Plot them stacked for easy comparison
fig, axes = plt.subplots(2, 1, figsize=(10, 6))

axes[0].plot(ecg_steady)
axes[0].set_title("Low HRV (heart_rate_std=1)")

axes[1].plot(ecg_variable, color='red')
axes[1].set_title("High HRV (heart_rate_std=20)")

plt.tight_layout()
plt.savefig("experiment_2_hrv.png", dpi=300, bbox_inches="tight")
plt.show()