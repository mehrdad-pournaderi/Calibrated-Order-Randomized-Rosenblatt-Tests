"""Fig (boot replacement): one clean setting (n=20, rho=0.5, k=2 same-sign shift, ncp=12)
showing naive miscalibration, restored size, and pooled-method power domination.
v2: adds e1 (calibrated single-ordering mixture e-value) so the e-avg vs e1 gap isolates aggregation."""
import numpy as np
from plot_style import setup, C, LBL, with_extras, log_ntr_axis, draw_series, finish_log_ntr_axis
plt = setup()

d = with_extras("results_calibration_pipeline.npz", ["results_calibration_pipeline_extra.npz", "results_calibration_pipeline_extra2.npz"])
alpha = float(d["alpha"]); x, fin, xk = log_ntr_axis(d)
tx = 3.75                                           # x position of the alpha label
keys = ["single", "pmerge", "bonf", "eavg", "e1", "chi2", "sym", "fisher"]
LS = {k: "-" for k in keys}; LS["e1"] = "--"      # calibrated single-ordering e-value (v2)

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))

A = ax[0]
for k in keys:
    draw_series(A, x, fin, d[f"{k}_ns"], ls=LS[k], color=C[k], lw=1.7, ms=3.5, label=LBL[k])
A.axhline(alpha, color="0.3", lw=1.1, ls="--")
A.text(tx + 0.25, alpha - 0.014, "α = 0.05", color="0.3", fontsize=8.5)
finish_log_ntr_axis(A, x, fin, xk)
A.set_title("A. Naive null size:\nmiscalibrated in both directions")
A.set_ylabel("null size")
A.set_ylim(0, 0.3); A.legend(fontsize=6.8, loc="upper right")

B = ax[1]
for k in keys:
    draw_series(B, x, fin, d[f"{k}_cs"], ls=LS[k], color=C[k], lw=1.7, ms=3.5, label=LBL[k])
B.axhline(alpha, color="0.3", lw=1.1, ls="--")
B.text(tx, alpha + 0.012, "α = 0.05", color="0.3", fontsize=8.5)
finish_log_ntr_axis(B, x, fin, xk)
B.set_ylim(0, 0.3)
B.set_title("B. Calibrated null size:\nevery method at α (same scale as A)")
B.set_ylabel("null size")

Cx = ax[2]
for k in keys:
    draw_series(Cx, x, fin, d[f"{k}_cp"], ls=LS[k], color=C[k], lw=1.7, ms=3.5, label=LBL[k])
finish_log_ntr_axis(Cx, x, fin, xk)
Cx.set_title("C. Calibrated power: pooled combiners on top\nat every training size")
Cx.set_ylabel("power")
Cx.set_ylim(top=0.585); Cx.legend(fontsize=6.8, loc="upper left", ncol=2, columnspacing=0.8)

fig.tight_layout()
fig.savefig("bootstrap_recalibrated.png", bbox_inches="tight")
print("saved bootstrap_recalibrated.png")
