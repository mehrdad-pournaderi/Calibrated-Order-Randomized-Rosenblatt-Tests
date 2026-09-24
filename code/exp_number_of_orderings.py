"""
CALIBRATED power vs number of orderings M, at rho=0.6 and rho=0 (known F).
Replaces the earlier naive-threshold study: every combiner is calibrated by exact
Monte-Carlo null quantiles (known F, so Procedure 1 reduces to this), the same
method set appears in both panels, and the alternative is energy-matched (ncp=12)
to the rest of the paper.

At rho=0 the Cholesky factor is I, so every ordering yields the same Simes p-value
and the same e-value; all combiners are then monotone transforms of one statistic
and, after calibration, have identical power (Prop. 'main'(iii) in the paper).
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(1212)
ALPHA = 0.05; LOG2 = np.log(2.0)
N, NCP = 10, 12.0
TAUS = np.array([1.0, 2.0, 3.0])
MGRID = np.array([1, 2, 4, 8, 16, 32, 64, 128])
MMAX = int(MGRID[-1])
REPS = 60_000
CHUNK = 16


def equicorr(n, rho):
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); return S


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def per_order(X, perms, Linv):
    """-> (P_simes, logE), each (reps, M). Chunked over orderings to bound memory."""
    reps = X.shape[0]; M = perms.shape[0]
    idx = np.arange(1, N + 1)
    P = np.empty((reps, M)); lE = np.empty((reps, M))
    norm = np.log(N) + np.log(len(TAUS))
    for s in range(0, M, CHUNK):
        sl = slice(s, min(s + CHUNK, M))
        Z = np.einsum('cij,rcj->rci', Linv[sl], X[:, perms[sl]])
        p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
        P[:, sl] = (N * np.sort(p2, axis=2) / idx).min(axis=2)
        terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
        lE[:, sl] = logsumexp(terms, axis=(0, 3)) - norm
    return P, lE


def cal_power(null_s, alt_s, low=True):
    # rank-based MC decision, p = (1+#{null at least as extreme})/(K+1) <= alpha
    sb = np.sort(null_s); K = len(null_s)
    if low:
        return float((1 + np.searchsorted(sb, alt_s, side='right') <= ALPHA * (K + 1)).mean())
    return float((1 + (K - np.searchsorted(sb, alt_s, side='left')) <= ALPHA * (K + 1)).mean())


out = {}
for rho in (0.6, 0.0):
    S = equicorr(N, rho); Sinv = np.linalg.inv(S); L = np.linalg.cholesky(S)
    mu = np.zeros(N); mu[0] = np.sqrt(NCP / Sinv[0, 0])
    perms = np.stack([rng.permutation(N) for _ in range(MMAX)])
    Linv = np.stack([np.linalg.inv(np.linalg.cholesky(S[np.ix_(p, p)])) for p in perms])
    Xn = rng.standard_normal((REPS, N)) @ L.T
    Xa = mu + rng.standard_normal((REPS, N)) @ L.T
    Pn, lEn = per_order(Xn, perms, Linv)
    Pa, lEa = per_order(Xa, perms, Linv)
    # NOTE: calibrate on the UNCAPPED statistics (mean p, min p, sum log e). The capped
    # nominal p-values min{1, M min p} and min{1, 2 mean p} put an atom at 1; under a
    # nonrandomized rank rule an observation on the atom can never reject, so the calibrated
    # test's rejection probability is bounded by the null mass below the cap once that mass
    # falls under alpha (at rho=0, M=128: P(128 P < 1) = 1/128, so the capped Bonferroni test
    # could reject at most 0.8% of the time). The uncapped summaries
    # are the statistics Procedure 1 is applied to throughout the paper; capping is reserved
    # for reporting a nominal merged p-value.
    res = {k: [] for k in ("pmerge", "bonf", "eavg")}
    for M in MGRID:
        res["pmerge"].append(cal_power(Pn[:, :M].mean(1), Pa[:, :M].mean(1)))
        res["bonf"].append(cal_power(Pn[:, :M].min(1), Pa[:, :M].min(1)))
        res["eavg"].append(cal_power(logsumexp(lEn[:, :M], 1), logsumexp(lEa[:, :M], 1), low=False))
    single = cal_power(Pn[:, 0], Pa[:, 0])
    out[rho] = {k: np.array(v) for k, v in res.items()}
    out[rho]["single"] = single
    print("rho=%.1f  single=%.3f" % (rho, single))
    for k in ("pmerge", "bonf", "eavg"):
        print("   %-7s " % k + " ".join("%.3f" % v for v in out[rho][k]))

np.savez("results_number_of_orderings.npz", Mgrid=MGRID,
         **{f"{'dep' if rho else 'ind'}_{k}": v for rho in (0.6, 0.0)
            for k, v in out[rho].items()},
         alpha=ALPHA, ncp=NCP, n=N)
print("saved sim12.npz")
