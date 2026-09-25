"""Is the SLOT a sufficient description of an ordering?
Compare orderings that place the signal in the same slot but arrange the rest differently.
(i)  equicorrelated Sigma, 1-coordinate signal  -> should be identical (exchangeability)
(ii) AR(1) Sigma (non-exchangeable), 1-coord signal -> should differ
(iii) equicorrelated, 2-coordinate signal -> should differ (predecessor set matters)
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(31415)
ALPHA = 0.05; n, NCP = 10, 12.0
REPS = 200_000
idx = np.arange(1, n + 1)


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (n * np.sort(p2, axis=-1) / idx).min(-1)


def power_of(perm, S, mu):
    """calibrated power of the fixed ordering `perm` (exact MC calibration)."""
    Sp = S[np.ix_(perm, perm)]
    Li = np.linalg.inv(np.linalg.cholesky(Sp))
    L = np.linalg.cholesky(S)
    Xn = rng.standard_normal((REPS, n)) @ L.T
    Xa = mu + rng.standard_normal((REPS, n)) @ L.T
    sn = simes2(Xn[:, perm] @ Li.T); sa = simes2(Xa[:, perm] @ Li.T)
    # rank-based Monte-Carlo decision (Procedure 1 at known F), uncapped statistics
    sb = np.sort(sn); K = len(sb)
    return float((1 + np.searchsorted(sb, sa, side='right') <= ALPHA * (K + 1)).mean())


def equicorr(rho):
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); return S


def ar1(phi):
    i = np.arange(n)
    return phi ** np.abs(i[:, None] - i[None, :])


print("(i) EQUICORRELATED rho=0.8, signal on coordinate 0, both orderings put it in slot 10")
S = equicorr(0.8); Sinv = np.linalg.inv(S)
mu = np.zeros(n); mu[0] = np.sqrt(NCP / Sinv[0, 0])
pa = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
pb = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0])
print("    ordering A (1..9 natural, signal last): power = %.4f" % power_of(pa, S, mu))
print("    ordering B (1..9 shuffled, signal last): power = %.4f" % power_of(pb, S, mu))

print("\n(ii) AR(1) phi=0.8 (NON-exchangeable), signal on coordinate 0, both put it in slot 10")
S = ar1(0.8); Sinv = np.linalg.inv(S)
mu = np.zeros(n); mu[0] = np.sqrt(NCP / Sinv[0, 0])
print("    ordering A (1..9 natural, signal last): power = %.4f" % power_of(pa, S, mu))
print("    ordering B (1..9 shuffled, signal last): power = %.4f" % power_of(pb, S, mu))

print("\n(iii) EQUICORRELATED rho=0.8, signal on coordinates 0 AND 1, coord 0 in slot 10")
S = equicorr(0.8); Sinv = np.linalg.inv(S)
v = np.zeros(n); v[0] = v[1] = 1.0
mu = v * np.sqrt(NCP / (v @ Sinv @ v))
pc = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])      # other signal coord (1) in slot 1
pd = np.array([2, 3, 4, 5, 6, 7, 8, 9, 1, 0])      # other signal coord (1) in slot 9
print("    partner signal coord early (slot 1): power = %.4f" % power_of(pc, S, mu))
print("    partner signal coord late  (slot 9): power = %.4f" % power_of(pd, S, mu))

print("\n(iv) AR(1) phi=0.8, signal on coordinate 5, placed in slot 5 but with DIFFERENT predecessors")
S = ar1(0.8); Sinv = np.linalg.inv(S)
mu = np.zeros(n); mu[5] = np.sqrt(NCP / Sinv[5, 5])
pe = np.array([4, 6, 3, 7, 5, 0, 1, 2, 8, 9])   # neighbours of coord 5 precede it
pf = np.array([0, 1, 2, 9, 5, 3, 4, 6, 7, 8])   # distant coordinates precede it
print("    predecessors = near neighbours {4,6,3,7}: power = %.4f" % power_of(pe, S, mu))
print("    predecessors = distant coords  {0,1,2,9}: power = %.4f" % power_of(pf, S, mu))
