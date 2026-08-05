"""
ONE CLEAN SETTING for calibration + power (Figure 'boot' replacement):
n=20, rho=0.5 equicorrelated, k=2 same-sign shifted coordinates, ncp=12,
N_tr in {4n, 8n, known}. For every method: naive size, calibrated size,
calibrated power. Pooled p-merge dominates all order-invariant references here.
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(1111)
ALPHA = 0.05; LOG2 = np.log(2.0); TAUS = np.array([1.0, 2.0, 3.0])
N, M, NCP, RHO, K = 20, 12, 12.0, 0.5, 2
R, B, NTE = 150, 199, 160
NTR_GRID = [80, 160, 0]                       # 0 = known F

S = np.full((N, N), RHO); np.fill_diagonal(S, 1.0)
Sinv_true = np.linalg.inv(S)
v = np.zeros(N); v[:K] = 1.0
MU = v * np.sqrt(NCP / (v @ Sinv_true @ v))
idx = np.arange(1, N + 1); ridge = 1e-3 * np.eye(N)
FCRIT = stats.chi2.isf(ALPHA, 2 * N); CCRIT = stats.chi2.isf(ALPHA, N)


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (N * np.sort(p2, axis=-1) / idx).min(-1)


def invsqrt(Sb):
    w, V = np.linalg.eigh(Sb)
    return (V * (w ** -0.5)[..., None, :]) @ np.swapaxes(V, -1, -2)


def om(Z):
    Ps = simes2(Z); Mn = Z.shape[-2]
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    logE = logsumexp(terms, axis=(0, -1)) - (np.log(N) + np.log(len(TAUS)))
    pf = stats.chi2.sf(-2 * np.log(np.maximum(p2, 1e-300)).sum(-1), 2 * N)
    return dict(single=Ps[..., 0], pmerge=np.minimum(2 * Ps.mean(-1), 1.0),
                bonf=np.minimum(Mn * Ps.min(-1), 1.0),
                eavg=np.exp(logsumexp(logE, -1) - np.log(Mn)), fisher=pf[..., 0])


def draw(mu, Sm, k): return mu + rng.standard_normal((k, N)) @ np.linalg.cholesky(Sm).T


PK = ["single", "pmerge", "bonf", "fisher", "sym"]; EK = ["eavg", "chi2"]
ALLK = PK + EK
OUT = {k: {"ns": [], "cs": [], "cp": []} for k in ALLK}
print("Clean setting: n=%d, rho=%.1f, k=%d, ncp=%.0f, M=%d, R=%d, B=%d" % (N, RHO, K, NCP, M, R, B))
for Ntr in NTR_GRID:
    acc = {k: {"ns": 0., "cs": 0., "cp": 0.} for k in ALLK}
    for _ in range(R):
        perms = np.stack([rng.permutation(N) for _ in range(M)])
        if Ntr == 0:
            Shat = S
        else:
            Xt = draw(np.zeros(N), S, Ntr); Shat = Xt.T @ Xt / Ntr + ridge
        Li = np.linalg.inv(np.linalg.cholesky(Shat[perms[:, :, None], perms[:, None, :]]))
        Sih = np.linalg.inv(Shat); Wsy = invsqrt(Shat)
        if Ntr == 0:
            Vc = draw(np.zeros(N), Shat, B)
            boot = om(np.einsum('mij,bmj->bmi', Li, Vc[:, perms]))
            boot["chi2"] = np.einsum('ki,ij,kj->k', Vc, Sih, Vc)
            boot["sym"] = simes2(Vc @ Wsy.T)
        else:
            Xtb = draw(np.zeros(N), Shat, B * Ntr).reshape(B, Ntr, N)
            Sst = np.einsum('bki,bkj->bij', Xtb, Xtb) / Ntr + ridge
            Lst = np.linalg.inv(np.linalg.cholesky(Sst[:, perms[:, :, None], perms[:, None, :]]))
            Vc = draw(np.zeros(N), Shat, B)
            boot = om(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]))
            boot["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
            boot["sym"] = simes2(np.einsum('bij,bj->bi', invsqrt(Sst), Vc))
        def st(X):
            s = om(np.einsum('mij,kmj->kmi', Li, X[:, perms]))
            s["chi2"] = np.einsum('ki,ij,kj->k', X, Sih, X); s["sym"] = simes2(X @ Wsy.T)
            return s
        nul = st(draw(np.zeros(N), S, NTE)); alt = st(draw(MU, S, NTE))
        for k in PK:
            c = np.quantile(boot[k], ALPHA)
            acc[k]["cs"] += (nul[k] <= c).mean(); acc[k]["cp"] += (alt[k] <= c).mean()
            acc[k]["ns"] += (nul[k] < ALPHA).mean()
        for k in EK:
            c = np.quantile(boot[k], 1 - ALPHA)
            acc[k]["cs"] += (nul[k] >= c).mean(); acc[k]["cp"] += (alt[k] >= c).mean()
        acc["eavg"]["ns"] += (nul["eavg"] >= 1 / ALPHA).mean()
        acc["chi2"]["ns"] += (nul["chi2"] >= CCRIT).mean()
    lab = "known" if Ntr == 0 else str(Ntr)
    print("  N_tr=%s" % lab)
    for k in ALLK:
        for m in ("ns", "cs", "cp"): OUT[k][m].append(acc[k][m] / R)
        print("    %-7s naive=%.3f cal-size=%.3f cal-POWER=%.3f"
              % (k, acc[k]["ns"] / R, acc[k]["cs"] / R, acc[k]["cp"] / R))

np.savez("results_calibration_pipeline.npz", Ntr=np.array([g if g else 100000 for g in NTR_GRID]),
         **{f"{k}_{m}": np.array(OUT[k][m]) for k in ALLK for m in ("ns", "cs", "cp")},
         alpha=ALPHA, ncp=NCP, n=N, M=M, rho=RHO, k=K)
print("saved sim11.npz")
