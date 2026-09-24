"""
SPARSITY PATH: k same-sign shifted coordinates out of n=20 (equicorrelated, rho=0.5),
energy matched at ncp=12 for every k, known F, exact MC calibration.
Shows where pooling beats the symmetric-root test: sym is competitive only at k=1 and
degrades monotonically toward its dense collapse, while pooled e-avg dominates at every k.
chi2 is constant in k -- the empirical face of orbit-flatness (Prop. 1).
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(1010)
ALPHA = 0.05; LOG2 = np.log(2.0)
N, M, NCP, RHO = 20, 12, 12.0, 0.5
TAUS = np.array([1.0, 2.0, 3.0])
REPS = 50_000
KS = [1, 2, 5, 10, 20]

S = np.full((N, N), RHO); np.fill_diagonal(S, 1.0)
Sinv = np.linalg.inv(S); L = np.linalg.cholesky(S); Li = np.linalg.inv(L)
w, V = np.linalg.eigh(S); W = (V * (w ** -0.5)) @ V.T
idx = np.arange(1, N + 1)


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (N * np.sort(p2, axis=-1) / idx).min(-1)


def stats_all(X):
    """Fresh M orderings per replicate (equicorr: chol fixed, only X permuted)."""
    P = np.argsort(rng.random((X.shape[0], M, N)), axis=-1)
    Z = np.einsum('ij,kmj->kmi', Li, np.take_along_axis(X[:, None, :], P, axis=-1))
    Ps = simes2(Z)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    logE = logsumexp(terms, axis=(0, -1)) - (np.log(N) + np.log(len(TAUS)))
    return dict(single=Ps[..., 0], pmerge=np.minimum(2 * Ps.mean(-1), 1.0),
                bonf=np.minimum(M * Ps.min(-1), 1.0),
                eavg=logsumexp(logE, -1) - np.log(M))


Xn = rng.standard_normal((REPS, N)) @ L.T
sn = stats_all(Xn); symn = simes2(Xn @ W.T)
chin = np.einsum('ki,ij,kj->k', Xn, Sinv, Xn)


def cal(nst, ast, low=True):
    # rank-based MC decision, p = (1+#{null at least as extreme})/(K+1) <= alpha
    sb = np.sort(nst); K = len(nst)
    if low:
        return float((1 + np.searchsorted(sb, ast, side='right') <= ALPHA * (K + 1)).mean())
    return float((1 + (K - np.searchsorted(sb, ast, side='left')) <= ALPHA * (K + 1)).mean())


KEYS = ["single", "pmerge", "bonf", "eavg", "sym", "chi2"]
out = {k: [] for k in KEYS}
print("Sparsity path (n=%d, rho=%.1f, ncp=%.0f, known F, calibrated, M=%d)" % (N, RHO, NCP, M))
print("   k   single  pmerge  bonf   eavg   sym    chi2")
for k in KS:
    v = np.zeros(N); v[:k] = 1.0
    mu = v * np.sqrt(NCP / (v @ Sinv @ v))
    Xa = mu + rng.standard_normal((REPS, N)) @ L.T
    sa = stats_all(Xa); syma = simes2(Xa @ W.T)
    chia = np.einsum('ki,ij,kj->k', Xa, Sinv, Xa)
    row = dict(single=cal(sn["single"], sa["single"]), pmerge=cal(sn["pmerge"], sa["pmerge"]),
               bonf=cal(sn["bonf"], sa["bonf"]), eavg=cal(sn["eavg"], sa["eavg"], low=False),
               sym=cal(symn, syma), chi2=cal(chin, chia, low=False))
    for kk in KEYS: out[kk].append(row[kk])
    print("  %2d   %.3f   %.3f  %.3f  %.3f  %.3f  %.3f" %
          (k, row["single"], row["pmerge"], row["bonf"], row["eavg"], row["sym"], row["chi2"]))

np.savez("results_sparsity_path.npz", ks=np.array(KS), **{k: np.array(v) for k, v in out.items()},
         alpha=ALPHA, ncp=NCP, n=N, M=M, rho=RHO)
print("saved sim10.npz")
