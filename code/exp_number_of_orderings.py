"""
CALIBRATED power versus the number of orderings M (Section 7.2), at rho=0.6 and rho=0, known F.

v2 (Nov 2026).  Every realization draws a FRESH bank of MMAX orderings and uses its first M for
each M in MGRID, so M=1 is the expected power of a randomly chosen single ordering, differences
between values of M are paired within a realization, and their Monte-Carlo standard errors are
reported; M=12, the value used throughout the paper, is on the grid.  Calibration is exact
Monte-Carlo at known F, conditional on the realization's bank (Procedure 1 with F known): B null
draws are whitened with the same orderings, and the rank rule is applied to the uncapped
summaries mean_m P_m (p-merge), min_m P_m (Bonferroni) and log mean_m E_m (e-value average).
(v1 drew a single bank for all replications, so its M=1 column was one particular ordering.)

The capped nominal p-values min{1, M min P} and min{1, 2 mean P} would put an atom at 1 that a
nonrandomized rank rule cannot break; calibrating the uncapped summaries avoids it.
Equicorrelated Sigma is invariant under permutation, so every ordering shares one Cholesky factor.
At rho=0 every ordering gives the same statistic, so every combiner is flat in M by construction.
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(1212)
ALPHA = 0.05; LOG2 = np.log(2.0)
N, NCP = 10, 12.0
TAUS = np.array([1.0, 2.0, 3.0])
MGRID = np.array([1, 2, 4, 8, 12, 16, 32, 64, 128])
MMAX = int(MGRID[-1])
B, NTE = 1999, 200                       # null draws per realization (calibration), test draws
RLZ = {0.6: 400, 0.0: 40}                # realizations (rho=0 is flat by construction)
COMB = ("pmerge", "bonf", "eavg")


def equicorr(n, rho):
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); return S


def per_order(X, perms, Li):
    """X (r, n), perms (MMAX, n) -> Simes p-values and log mixture e-values, each (r, MMAX)."""
    Z = np.einsum('ij,rmj->rmi', Li, X[:, perms])
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    P = (N * np.sort(p2, axis=2) / np.arange(1, N + 1)).min(axis=2)
    lc = np.logaddexp(TAUS[:, None, None, None] * Z, -TAUS[:, None, None, None] * Z) - LOG2
    lE = logsumexp(-0.5 * TAUS[:, None, None, None] ** 2 + lc, axis=(0, 3)) - np.log(N * len(TAUS))
    return P, lE


def summaries(P, lE, M):
    return {"pmerge": P[:, :M].mean(1), "bonf": P[:, :M].min(1), "eavg": logsumexp(lE[:, :M], 1)}


def power(null_s, alt_s, low):
    """Rank-based Monte-Carlo decision, p = (1 + #{null at least as extreme}) / (B + 1) <= alpha."""
    sb = np.sort(null_s); K = len(null_s)
    if low:
        cnt = np.searchsorted(sb, alt_s, side="right")
    else:
        cnt = K - np.searchsorted(sb, alt_s, side="left")
    return float((1 + cnt <= ALPHA * (K + 1)).mean())


out = {}
for rho in (0.6, 0.0):
    S = equicorr(N, rho); Sinv = np.linalg.inv(S); L = np.linalg.cholesky(S); Li = np.linalg.inv(L)
    mu = np.zeros(N); mu[0] = np.sqrt(NCP / Sinv[0, 0])
    pw = np.zeros((RLZ[rho], len(COMB), len(MGRID)))
    for r in range(RLZ[rho]):
        perms = np.argsort(rng.random((MMAX, N)), axis=1)              # fresh bank
        Pn, lEn = per_order(rng.standard_normal((B, N)) @ L.T, perms, Li)
        Pa, lEa = per_order(mu + rng.standard_normal((NTE, N)) @ L.T, perms, Li)
        for j, M in enumerate(MGRID):
            sn, sa = summaries(Pn, lEn, int(M)), summaries(Pa, lEa, int(M))
            for c, k in enumerate(COMB):
                pw[r, c, j] = power(sn[k], sa[k], low=(k != "eavg"))
    tag = "dep" if rho else "ind"
    for c, k in enumerate(COMB):
        out[f"{tag}_{k}"] = pw[:, c, :].mean(0)
        out[f"{tag}_{k}_se"] = pw[:, c, :].std(0, ddof=1) / np.sqrt(RLZ[rho])
        d = pw[:, c, :] - pw[:, c, [list(MGRID).index(12)]]               # paired gap to M=12
        out[f"{tag}_{k}_gap12"] = d.mean(0)
        out[f"{tag}_{k}_gap12_se"] = d.std(0, ddof=1) / np.sqrt(RLZ[rho])
    print("rho=%.1f  (%d realizations x %d test draws, B=%d)" % (rho, RLZ[rho], NTE, B))
    print("   M        " + " ".join("%7d" % m for m in MGRID))
    for k in COMB:
        print("   %-7s  " % k + " ".join("%7.3f" % v for v in out[f"{tag}_{k}"]))
        print("   %-7s  " % "  (se)" + " ".join("%7.3f" % v for v in out[f"{tag}_{k}_se"]))
        print("   %-7s  " % "  -M12" + " ".join("%+7.3f" % v for v in out[f"{tag}_{k}_gap12"]))
        print("   %-7s  " % "  (se)" + " ".join("%7.3f" % v for v in out[f"{tag}_{k}_gap12_se"]))

np.savez("results_number_of_orderings.npz", Mgrid=MGRID, alpha=ALPHA, ncp=NCP, n=N, B=B, NTE=NTE,
         R_dep=RLZ[0.6], R_ind=RLZ[0.0], **out)
print("saved results_number_of_orderings.npz")
