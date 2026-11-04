"""Fig (FDR layer): heterogeneous screening, naive vs bootstrap-calibrated.
Naive power is deliberately not plotted -- it is bought with uncontrolled FDR."""
import numpy as np
from plot_style import setup, C, LBL, with_extras, log_ntr_axis, draw_series, finish_log_ntr_axis
plt = setup()

d = with_extras("results_multiplicity_screening.npz", ["results_multiplicity_screening_extra.npz", "results_multiplicity_screening_extra2.npz"])
q = float(d["q"]); x, fin, xk = log_ntr_axis(d)
tx = 3.75                                           # x position of the target label
keys = ["single", "pmerge", "bonf", "eavg", "chi2", "sym"]

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))

A = ax[0]
for k in keys:
    draw_series(A, x, fin, d[f"naive_{k}_fdr"], color=C[k], lw=1.7, ms=3.5, label=LBL[k])
A.axhline(q, color="0.3", lw=1.1, ls="--", label="target $q=0.10$")
finish_log_ntr_axis(A, x, fin, xk); A.set_ylim(-0.02, 0.6)
A.set_title("A. Naive: realized FDR uncontrolled\n(or, for the e-value average, near zero)")
A.set_ylabel("realized FDR")
A.legend(fontsize=6.8, loc="upper right")

B = ax[1]
for k in keys:
    draw_series(B, x, fin, d[f"cal_{k}_fdr"], color=C[k], lw=1.7, ms=3.5, label=LBL[k])
B.axhline(q, color="0.3", lw=1.1, ls="--")
B.text(tx, q + 0.035, "target $q=0.10$", color="0.3", fontsize=8.5)
finish_log_ntr_axis(B, x, fin, xk); B.set_ylim(-0.02, 0.6)
B.set_title("B. Bootstrap-calibrated: FDR at or near target\n(same scale as A)")
B.set_ylabel("realized FDR")

Cx = ax[2]
for k in keys:
    draw_series(Cx, x, fin, d[f"cal_{k}_pow"], color=C[k], lw=1.7, ms=3.5, label=LBL[k])
finish_log_ntr_axis(Cx, x, fin, xk)
Cx.set_title("C. Calibrated power: pooled combiners lead\nwhen departures differ in shape")
Cx.set_ylabel("power")
Cx.set_ylim(top=0.8); Cx.legend(fontsize=6.8, loc="upper left", ncol=2, columnspacing=0.8)

fig.tight_layout()
fig.savefig("estimated_F.png", bbox_inches="tight"); print("saved estimated_F.png")
