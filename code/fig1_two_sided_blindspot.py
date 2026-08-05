"""Fig (Sec. 5): why the base statistic must be two-sided (n=20, independent null,
where all orderings coincide).
A: signed single-coordinate shift — one-sided is blind below level for delta<0.
B: mixed-sign shift +/- c/sqrt(n) on half the coordinates — no one-sided
   orientation helps; two-sided dominates uniformly."""
import numpy as np
from plot_style import setup, C
plt = setup()

d = np.load("results_base_statistic.npz")
alpha = float(d["alpha"]); n = 20

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.1))

A = ax[0]
A.plot(d["deltas_B"], d["pow_one"], "-", color=C["one"], lw=2, label="one-sided Simes")
A.plot(d["deltas_B"], d["pow_two"], "-", color=C["two"], lw=2, label="two-sided Simes")
A.axhline(alpha, color="0.45", lw=0.9, ls=":")
A.axhline(alpha * (n - 1) / n, color=C["one"], lw=0.9, ls="--", alpha=0.7)
A.text(-3.9, alpha + 0.015, r"$\alpha$", color="0.35", fontsize=10)
A.text(-1.7, alpha * (n - 1) / n - 0.055, r"$\alpha(n-1)/n$", color=C["one"], fontsize=9)
A.axvline(0, color="0.85", lw=0.7)
A.set_title("A. One shifted coordinate: the one-sided test is\nblind (below level) for $\\delta<0$")
A.set_xlabel(r"mean shift $\delta$ of one coordinate")
A.set_ylabel("power")
A.set_ylim(-0.02, 1.02)
A.legend(loc="upper center")

B = ax[1]
B.plot(d["cs"], d["pow_mix_one"], "-", color=C["one"], lw=2,
       label="one-sided Simes (either orientation)")
B.plot(d["cs"], d["pow_mix_two"], "-", color=C["two"], lw=2, label="two-sided Simes")
B.axhline(alpha, color="0.45", lw=0.9, ls=":")
B.set_title("B. Mixed signs, $\\pm c/\\sqrt{n}$ on half the coordinates each:\n"
            "no orientation helps — two-sided dominates uniformly")
B.set_xlabel(r"total signal size $c$  (energy $c^2$)")
B.set_ylabel("power")
B.set_ylim(-0.02, 1.02)
B.legend(loc="upper left")

fig.tight_layout()
fig.savefig("onesided_blindspot.png", bbox_inches="tight")
print("saved onesided_blindspot.png")
