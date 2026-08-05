"""(a) Why slot power increases: energy concentration by slot.
   (b) Is a DATA-DEPENDENT ordering (sort by |x|) valid? -> null size check."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(2024)
ALPHA = 0.05
n, RHO, NCP = 10, 0.8, 12.0
S = np.full((n, n), RHO); np.fill_diagonal(S, 1.0)
Sinv = np.linalg.inv(S); L = np.linalg.cholesky(S); Li = np.linalg.inv(L)
mu = np.zeros(n); mu[0] = np.sqrt(NCP / Sinv[0, 0])
idx = np.arange(1, n + 1)

print("(a) whitened signal profile by slot of the shifted coordinate (rho=%.1f, ncp=%.0f)" % (RHO, NCP))
print(" slot  cond.sd s_k  |m| at signal slot  share of energy there  #coords with |m|>0.1")
others = list(range(1, n))
for pos in range(n):
    p = np.array(others[:pos] + [0] + others[pos:])
    m = Li @ mu[p]                      # whitened mean profile
    share = m[pos] ** 2 / (m @ m)
    s_k = 1.0 / Li[pos, pos]            # conditional sd at slot pos
    print("  %2d      %.3f          %.2f                %.2f                 %d"
          % (pos + 1, s_k, abs(m[pos]), share, int((np.abs(m) > 0.1).sum())))

print("\n(b) null size of a DATA-DEPENDENT ordering (sort coordinates by |x|), nominal alpha=0.05")
REPS = 100_000
X = rng.standard_normal((REPS, n)) @ L.T          # NULL data


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (n * np.sort(p2, axis=-1) / idx).min(-1)


# fixed (natural) ordering: valid reference
print("   fixed natural ordering        size = %.4f" % (simes2(X @ Li.T) < ALPHA).mean())
# random ordering independent of data: valid
P = np.argsort(rng.random((REPS, n)), axis=1)
print("   random ordering (indep.)      size = %.4f"
      % (simes2(np.einsum('ij,kj->ki', Li, np.take_along_axis(X, P, 1))) < ALPHA).mean())
# data-dependent: sort by |x| ascending (largest last)
Pd = np.argsort(np.abs(X), axis=1)
print("   sorted by |x| ascending       size = %.4f"
      % (simes2(np.einsum('ij,kj->ki', Li, np.take_along_axis(X, Pd, 1))) < ALPHA).mean())
Pd2 = np.argsort(-np.abs(X), axis=1)
print("   sorted by |x| descending      size = %.4f"
      % (simes2(np.einsum('ij,kj->ki', Li, np.take_along_axis(X, Pd2, 1))) < ALPHA).mean())

print("\n(c) can the data-dependent ordering be rescued by calibrating it? (known F, alpha=0.05)")
from scipy.special import logsumexp
LOG2 = np.log(2.0); TAUS = np.array([1.0, 2.0, 3.0]); M = 12
Xa = mu + rng.standard_normal((REPS, n)) @ L.T


def sorted_stat(Y, desc=True):
    Pp = np.argsort(-np.abs(Y) if desc else np.abs(Y), axis=1)
    return simes2(np.einsum('ij,kj->ki', Li, np.take_along_axis(Y, Pp, 1)))


def pooled_stat(Y):
    Pr = np.argsort(rng.random((Y.shape[0], M, n)), axis=-1)
    Z = np.einsum('ij,kmj->kmi', Li, np.take_along_axis(Y[:, None, :], Pr, axis=-1))
    return np.minimum(2 * simes2(Z).mean(-1), 1.0)


def cal(nst, ast):
    c = np.quantile(nst, ALPHA); return float((ast <= c).mean())


for lab, desc in [("sorted |x| descending", True), ("sorted |x| ascending", False)]:
    print("   %-22s calibrated power = %.3f" % (lab, cal(sorted_stat(X, desc), sorted_stat(Xa, desc))))
print("   %-22s calibrated power = %.3f" % ("pooled p-merge (M=12)", cal(pooled_stat(X), pooled_stat(Xa))))
print("   %-22s calibrated power = %.3f" % ("single fixed ordering", cal(simes2(X @ Li.T), simes2(Xa @ Li.T))))
