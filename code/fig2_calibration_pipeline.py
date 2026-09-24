"""Fig (boot replacement): one clean setting (n=20, rho=0.5, k=2 same-sign shift, ncp=12)
showing naive miscalibration, restored size, and pooled-method power domination.
v2: adds e1 (calibrated single-ordering mixture e-value) so the e-avg vs e1 gap isolates aggregation."""
import numpy as np
from plot_style import setup, C, LBL
plt = setup()

d = np.load("results_calibration_pipeline.npz")
alpha = float(d["alpha"]); xt = [r"$4n$", r"$8n$", "known"]; x = np.arange(3)
keys = ["single", "pmerge", "bonf", "eavg", "e1", "chi2", "sym", "fisher"]
LS = {k: "-" for k in keys}; LS["e1"] = "--"      # calibrated single-ordering e-value (v2)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))

A = ax[0]
for k in keys:
    A.plot(x, d[f"{k}_ns"], LS[k], marker="o", color=C[k], lw=1.7, ms=3.5, label=LBL[k])
A.axhline(alpha, color="0.3", lw=1.1, ls="--")
A.text(1.9, alpha + 0.007, "α = 0.05", color="0.3", fontsize=8.5)
A.set_xticks(x); A.set_xticklabels(xt)
A.set_title("A. Naive null size:\nmiscalibrated in both directions")
A.set_xlabel(r"$N_{\mathrm{tr}}$"); A.set_ylabel("null size")
A.set_ylim(0, 0.3); A.legend(fontsize=6.8, loc="upper right")

B = ax[1]
for k in keys:
    B.plot(x, d[f"{k}_cs"], LS[k], marker="o", color=C[k], lw=1.7, ms=3.5, label=LBL[k])
B.axhline(alpha, color="0.3", lw=1.1, ls="--")
B.text(1.9, alpha + 0.007, "α = 0.05", color="0.3", fontsize=8.5)
B.set_xticks(x); B.set_xticklabels(xt)
B.set_ylim(0, 0.3)
B.set_title("B. Calibrated null size:\nevery method at α (same scale as A)")
B.set_xlabel(r"$N_{\mathrm{tr}}$"); B.set_ylabel("null size")

Cx = ax[2]
for k in keys:
    Cx.plot(x, d[f"{k}_cp"], LS[k], marker="o", color=C[k], lw=1.7, ms=3.5, label=LBL[k])
Cx.set_xticks(x); Cx.set_xticklabels(xt)
Cx.set_title("C. Calibrated power: pooled combiners on top\nat every training size")
Cx.set_xlabel(r"$N_{\mathrm{tr}}$"); Cx.set_ylabel("power")
Cx.set_ylim(top=0.585); Cx.legend(fontsize=6.8, loc="upper left", ncol=2, columnspacing=0.8)

fig.tight_layout()
fig.savefig("bootstrap_recalibrated.png", bbox_inches="tight")
print("saved bootstrap_recalibrated.png")
