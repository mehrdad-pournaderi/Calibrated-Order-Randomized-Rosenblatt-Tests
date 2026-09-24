"""Figure: the exactly solvable orbit (equicorrelated Sigma, sparse departure, ncp = 12).

(A) exact per-slot power of the within-ordering Bonferroni test, n = 10.
(B) max-min spread across the orbit as a function of rho, for several n: the spread peaks
    near rho ~ 0.75 and then falls, at every dimension.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from check_equicorrelated_orbit import beta, delta_for_ncp, gain_and_spread

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.6))

n = 10
for rho, c in ((0.2, "#8ecae6"), (0.5, "#219ebc"), (0.8, "#023047"), (0.95, "#fb8500")):
    d = delta_for_ncp(n, rho, 12.0)
    b = [beta(n, rho, d, j) for j in range(1, n + 1)]
    ax1.plot(range(1, n + 1), b, "o-", ms=4, color=c, label=r"$\rho=%.2f$" % rho)
ax1.axhline(0.7557, color="gray", lw=0.8, ls=":")
ax1.text(1.02, 0.759, "oracle (slot $n$): $\\rho$-free", fontsize=8, color="gray")
ax1.set_xlabel("slot $j$ of the departing coordinate")
ax1.set_ylabel("exact power")
ax1.set_title("(A) power along the orbit ($n=10$, ncp $=12$)", fontsize=10)
ax1.legend(fontsize=8, loc="lower right")
ax1.set_xticks(range(1, n + 1))

grid = np.arange(0.02, 0.996, 0.005)
for nn, c in ((5, "#8ecae6"), (10, "#219ebc"), (20, "#023047"), (50, "#fb8500")):
    sp = [gain_and_spread(nn, r, 12.0)[1] for r in grid]
    ax2.plot(grid, sp, color=c, label="$n=%d$" % nn)
    i = int(np.argmax(sp))
    ax2.plot(grid[i], sp[i], "o", ms=5, color=c)
ax2.axvline(0.75, color="gray", lw=0.8, ls=":")
ax2.set_xlabel(r"equicorrelation $\rho$")
ax2.set_ylabel("spread  $\\beta(n)-\\beta(1)$")
ax2.set_title(r"(B) orbit spread peaks near $\rho\approx0.75$ at every $n$", fontsize=10)
ax2.legend(fontsize=8, loc="upper left")

plt.tight_layout()
plt.savefig("orbit_exact.png", dpi=200)
print("wrote orbit_exact.png")
