"""Does the naive-threshold Bonferroni dome survive the TWO-SIDED base statistic?
Hub covariance, n=20, one-coordinate (hub) shift, nominal alpha/M thresholds."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(303)
ALPHA = 0.05
n, RHO = 20, 0.75
MGRID = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
REPS = 20_000
idx = np.arange(1, n + 1)

S = np.full((n, n), RHO * RHO); S[0, :] = RHO; S[:, 0] = RHO
np.fill_diagonal(S, 1.0)
L = np.linalg.cholesky(S); Sinv = np.linalg.inv(S)
MMAX = int(MGRID[-1])
perms = np.stack([np.arange(n)] + [rng.permutation(n) for _ in range(MMAX - 1)])
Linv = np.stack([np.linalg.inv(np.linalg.cholesky(S[np.ix_(p, p)])) for p in perms])


def per_order_simes2(X, chunk=32):
    P = np.empty((X.shape[0], MMAX))
    for s in range(0, MMAX, chunk):
        sl = slice(s, min(s + chunk, MMAX))
        Z = np.einsum('cij,rcj->rci', Linv[sl], X[:, perms[sl]])
        p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
        P[:, sl] = (n * np.sort(p2, axis=2) / idx).min(axis=2)
    return P


print("Two-sided Bonferroni-over-orders at NOMINAL thresholds (hub, n=20):")
for d in (1.0, 1.5, 2.55):
    mu = np.zeros(n); mu[0] = d
    P = per_order_simes2(mu + rng.standard_normal((REPS, n)) @ L.T)
    pw = [float(np.mean(np.minimum(M * P[:, :M].min(1), 1) < ALPHA)) for M in MGRID]
    star = MGRID[int(np.argmax(pw))]
    print("  delta=%.2f (ncp=%.1f): " % (d, (mu @ Sinv @ mu)) +
          " ".join("%.3f" % v for v in pw) + "   peak at M=%d, end=%.3f" % (star, pw[-1]))
print("M grid:", MGRID)
