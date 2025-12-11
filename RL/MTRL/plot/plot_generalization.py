import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 1. Data
# ===========================

methods = [
    "RawHA-RL",
    "GroundHA-RL",
    "MotionHA-RL",
    "CrossAgent(w/o SSRL)",
    "CrossAgent"
]

# In-domain
FT_in = np.array([90.0, 93.3, 96.7, 86.6, 93.3])
ASR_in = np.array([70.1, 52.6, 61.9, 39.7, 68.8])
STD_in = np.array([33.6, 31.5, 27.5, 48.1, 30.5])

# Out-of-domain
FT_ood = np.array([51.9, 41.0, 43.8, 41.4, 58.3])
ASR_ood = np.array([42.4, 39.4, 39.1, 39.7, 49.1])
STD_ood = np.array([45.1, 44.3, 42.0, 48.1, 46.6])

# ===========================
# 2. Plotting style
# ===========================

colors = ["gray", "gray", "red", "gray", "blue"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.subplots_adjust(hspace=0.3, wspace=0.2)

x = np.arange(len(methods))

# ===========================
# In-domain FT
# ===========================
ax = axes[0, 0]
bars = ax.bar(x, FT_in, color=colors)
ax.set_title("In-Domain FT (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha="right")

# ===========================
# In-domain ASR
# ===========================
ax = axes[0, 1]
bars = ax.bar(x, ASR_in, yerr=STD_in, capsize=5, color=colors)
ax.set_title("In-Domain ASR (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha="right")

# ===========================
# OOD FT
# ===========================
ax = axes[1, 0]
bars = ax.bar(x, FT_ood, color=colors)
ax.set_title("Out-of-Domain FT (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha="right")

# ===========================
# OOD ASR
# ===========================
ax = axes[1, 1]
bars = ax.bar(x, ASR_ood, yerr=STD_ood, capsize=5, color=colors)
ax.set_title("Out-of-Domain ASR (%)")
ax.set_ylim(0, 110)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=30, ha="right")

# ===========================
# Legend
# ===========================
fig.legend(
    ["Baseline", "MotionHA-RL (SOTA)", "CrossAgent"],
    loc="lower center",
    ncol=3,
    fontsize=12,
    frameon=False
)

plt.savefig("output/toy.png")
