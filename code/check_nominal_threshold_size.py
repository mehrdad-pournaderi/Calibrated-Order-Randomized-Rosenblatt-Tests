"""Naive-threshold null sizes of the pooled combiners with the TWO-SIDED base statistic
(known F, n=10, rho=0.6, M=128) -- the numbers quoted in the combiner-guidance section."""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(404)
ALPHA = 0.05; LOG2 = np.log(2.0)
n, RHO, M = 10, 0.6, 128
TAUS = np.array([1.0, 2.0, 3.0])
REPS = 60_000
idx = np.arange(1, n + 1)

S = np.full((n, n), RHO); np.fill_diagonal(S, 1.0)
L = np.linalg.cholesky(S)
perms = np.stack([rng.permutation(n) for _ in range(M)])
Linv = np.stack([np.linalg.inv(np.linalg.cholesky(S[np.ix_(p, p)])) for p in perms])


def logcosh(x): return np.logaddexp(x, -x) - LOG2


X = rng.standard_normal((REPS, n)) @ L.T
P = np.empty((REPS, M)); lE = np.empty((REPS, M))
norm = np.log(n) + np.log(len(TAUS))
for s in range(0, M, 16):
    sl = slice(s, min(s + 16, M))
    Z = np.einsum('cij,rcj->rci', Linv[sl], X[:, perms[sl]])
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)          # TWO-SIDED
    P[:, sl] = (n * np.sort(p2, axis=2) / idx).min(axis=2)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    lE[:, sl] = logsumexp(terms, axis=(0, 3)) - norm

print("Naive (nominal-threshold) null sizes, two-sided base, n=%d rho=%.1f M=%d:" % (n, RHO, M))
print("  single ordering  %.4f" % np.mean(P[:, 0] < ALPHA))
print("  p-merge          %.5f" % np.mean(np.minimum(2 * P.mean(1), 1) < ALPHA))
print("  Bonferroni       %.5f" % np.mean(np.minimum(M * P.min(1), 1) < ALPHA))
print("  e-value average  %.5f" % np.mean(logsumexp(lE, 1) - np.log(M) >= np.log(1 / ALPHA)))
