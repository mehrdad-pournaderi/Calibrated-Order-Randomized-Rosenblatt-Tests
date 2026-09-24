"""Figure 3: calibrated power versus the shape of the departure, at known F."""
import numpy as np
from plot_style import setup, C, LBL
plt = setup()

d = np.load("results_shape_sparse_dense.npz")          # labels: ["sparse", "dense"]
keys = ["single", "pmerge", "bonf", "eavg", "e1", "chi2", "sym", "fisher"]   # v2: + e1

fig, ax = plt.subplots(figsize=(7.6, 4.4))
x = np.arange(2); w = 0.10
for i, k in enumerate(keys):
    ax.bar(x + (i - 3.5) * w, d[f"cpw_known_{k}"], w, color=C[k], label=LBL[k])
ax.set_xticks(x)
ax.set_xticklabels(["sparse departure\n(one coordinate)", "dense departure\n(all coordinates)"])
ax.set_ylabel("calibrated power")
ax.set_ylim(0.3, 0.88)
ax.legend(fontsize=7.2, loc="upper center", ncol=4, framealpha=0.95, columnspacing=0.9)
fig.tight_layout()
fig.savefig("power_summary.png", bbox_inches="tight")
print("saved power_summary.png")
