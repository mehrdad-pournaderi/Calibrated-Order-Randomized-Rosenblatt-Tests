"""Fig (FDR layer): heterogeneous screening, naive vs bootstrap-calibrated.
Naive power is deliberately not plotted -- it is bought with uncontrolled FDR."""
import numpy as np
from plot_style import setup, C, LBL
plt = setup()

d = np.load("results_multiplicity_screening.npz")
q = float(d["q"]); x = np.arange(3); xt = ["80", "160", "known"]
keys = ["single", "pmerge", "bonf", "eavg", "chi2", "sym"]

fig, ax = plt.subplots(1, 3, figsize=(13, 4.1))

A = ax[0]
for k in keys:
    A.plot(x, d[f"naive_{k}_fdr"], "-o", color=C[k], lw=1.7, ms=3.5, label=LBL[k])
A.axhline(q, color="0.3", lw=1.1, ls="--")
A.text(1.75, q + 0.015, "target $q=0.10$", color="0.3", fontsize=8.5)
A.set_xticks(x); A.set_xticklabels(xt); A.set_ylim(-0.02, 0.6)
A.set_title("A. Naive: realized FDR uncontrolled\n(or, for the e-value average, near zero)")
A.set_xlabel(r"$N_{\mathrm{tr}}$"); A.set_ylabel("realized FDR")
A.legend(fontsize=6.8, loc="upper right")

B = ax[1]
for k in keys:
    B.plot(x, d[f"cal_{k}_fdr"], "-o", color=C[k], lw=1.7, ms=3.5, label=LBL[k])
B.axhline(q, color="0.3", lw=1.1, ls="--")
B.text(1.75, q + 0.015, "target $q=0.10$", color="0.3", fontsize=8.5)
B.set_xticks(x); B.set_xticklabels(xt); B.set_ylim(-0.02, 0.6)
B.set_title("B. Bootstrap-calibrated: FDR at or near target\n(same scale as A)")
B.set_xlabel(r"$N_{\mathrm{tr}}$"); B.set_ylabel("realized FDR")

Cx = ax[2]
for k in keys:
    Cx.plot(x, d[f"cal_{k}_pow"], "-o", color=C[k], lw=1.7, ms=3.5, label=LBL[k])
Cx.set_xticks(x); Cx.set_xticklabels(xt)
Cx.set_title("C. Calibrated power: pooled combiners lead\nwhen departures differ in shape")
Cx.set_xlabel(r"$N_{\mathrm{tr}}$"); Cx.set_ylabel("power")
Cx.legend(fontsize=6.8, loc="upper left")

fig.tight_layout()
fig.savefig("estimated_F.png", bbox_inches="tight"); print("saved estimated_F.png")
