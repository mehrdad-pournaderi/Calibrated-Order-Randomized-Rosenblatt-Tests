"""
Why the ranking of combiners depends on the decision threshold.

Bonferroni-over-orders is fifth of six in Figure 2 and first in Figure 4.  That looks like a
contradiction and is not: the two experiments interrogate the combiners at very different
thresholds.  Figure 2 tests one hypothesis at alpha = 0.05.  Figure 4 runs Benjamini-Hochberg
over N = 200 hypotheses at q = 0.10 with ten non-nulls, so a p-value must reach q*k/N -- that
is, 0.0005 to 0.005 -- before it is reported.  The screen therefore asks each statistic for
p-values one to two orders of magnitude smaller than the single test does.

That distinction matters because the combiners differ in *where* their evidence sits.
Bonferroni over orderings, M * min_m P_m, keys on the single most favourable ordering, so its
calibrated p-value has a heavier left tail than the averaging rules but a worse typical value.
p-merge and the e-value average use all M views, which pays at a moderate threshold and costs
at an extreme one.

This script isolates that mechanism.  The design is held completely fixed -- the covariance,
dimension, alternative and number of orderings never change -- and only alpha is swept.  If the
explanation is right, the ranking should invert as alpha falls, with no change of design at all.

Three alternatives are used: the two-coordinate departure of Figure 2, and the sparse and dense
halves of the heterogeneous non-nulls of Figure 4.

v2 (Sep 2026): decisions by the rank-based Monte-Carlo rule of Procedure 1 on the uncapped merger
summaries, so this script follows the same calibration as every other experiment (it previously
used interpolated np.quantile thresholds, whose bias at 2e5 draws is of order 1e-5).
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(31337)
n, M, RHO = 20, 12, 0.5
TAUS = np.array([1.0, 2.0, 3.0])
LOG2 = np.log(2.0)
NDRAW, CHUNK = 200_000, 20_000
ALPHAS = (0.05, 0.02, 0.005, 0.001, 0.0005)

idx = np.arange(1, n + 1)
S = np.full((n, n), RHO); np.fill_diagonal(S, 1.0)
Sinv = np.linalg.inv(S); L = np.linalg.cholesky(S)


def logcosh(x):
    return np.logaddexp(x, -x) - LOG2


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (n * np.sort(p2, axis=-1) / idx).min(-1)


PERMS = np.stack([rng.permutation(n) for _ in range(M)])
LINV = np.stack([np.linalg.inv(np.linalg.cholesky(S[np.ix_(p, p)])) for p in PERMS])
KEYS = ["pmerge", "bonf", "eavg", "single"]


def stats_of(V):
    """All statistics oriented so that SMALL means evidence against the null."""
    out = {k: [] for k in KEYS}
    for s0 in range(0, len(V), CHUNK):
        Z = np.einsum('mij,kmj->kmi', LINV, V[s0:s0 + CHUNK][:, PERMS])
        Ps = simes2(Z)
        t = np.stack([-0.5 * a * a + logcosh(a * Z) for a in TAUS])
        lE = logsumexp(t, axis=(0, -1)) - (np.log(n) + np.log(len(TAUS)))
        out["pmerge"].append(2 * Ps.mean(-1))          # uncapped summaries (v2)
        out["bonf"].append(M * Ps.min(-1))
        out["eavg"].append(-(logsumexp(lE, -1) - np.log(M)))
        out["single"].append(Ps[:, 0])
    return {k: np.concatenate(v) for k, v in out.items()}


null = stats_of(rng.standard_normal((NDRAW, n)) @ L.T)
SORTED_NULL = {x: np.sort(null[x]) for x in KEYS}

CASES = [("fig2", 12.0, 2, "Figure 2 alternative: two coordinates, ncp=12"),
         ("sparse", 20.0, 1, "Figure 4, sparse half: one coordinate, ncp=20"),
         ("dense", 20.0, n, "Figure 4, dense half: all coordinates, ncp=20")]

print("Design fixed throughout: n=%d, rho=%.1f, M=%d, %d null and %d alternative draws.\n"
      % (n, RHO, M, NDRAW, NDRAW))
POW = {}
for tag, ncp, k_shift, lab in CASES:
    v = np.zeros(n); v[:k_shift] = 1.0
    mu = v * np.sqrt(ncp / (v @ Sinv @ v))
    alt = stats_of(mu + rng.standard_normal((NDRAW, n)) @ L.T)
    print(lab)
    print("  %-9s %s" % ("alpha", "  ".join("%8s" % x for x in KEYS)))
    for a in ALPHAS:
        # rank-based Monte-Carlo decision (Procedure 1 at known F): reject iff
        # (1 + #{null <= obs}) / (NDRAW + 1) <= alpha   (v2; was an interpolated quantile)
        row = {x: float((1 + np.searchsorted(SORTED_NULL[x], alt[x], side='right')
                         <= a * (NDRAW + 1)).mean()) for x in KEYS}
        for x in KEYS:
            POW["%s_%s" % (tag, x)] = POW.get("%s_%s" % (tag, x), []) + [row[x]]
        print("  %-9.4f %s   <- %s" % (a, "  ".join("%8.3f" % row[x] for x in KEYS),
                                       max(KEYS, key=lambda x: row[x])))
    print()

np.savez("results_threshold_dependence.npz", alphas=np.array(ALPHAS),
         methods=np.array(KEYS), cases=np.array([c[0] for c in CASES]),
         **{k: np.array(v) for k, v in POW.items()},
         n=n, M=M, rho=RHO, ndraw=NDRAW)
print("saved results_threshold_dependence.npz")
