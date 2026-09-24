"""
ONE CLEAN SETTING for calibration + power (Figure 'boot' replacement):
n=20, rho=0.5 equicorrelated, k=2 same-sign shifted coordinates, ncp=12,
N_tr in {4n, 8n, known}. For every method: naive size, calibrated size,
calibrated power. Pooled p-merge dominates all order-invariant references here.

v3: calibration statistics are the UNCAPPED merger summaries mean_m P_m and M*min_m P_m
 (the capped p-values min{1,.} are monotone in these below the cap and identical for every
 decision here; capping only matters when it creates ties at 1, i.e. when more than a 1-alpha
 fraction of null draws hit the cap, which happens at large M under weak dependence).
 Naive decisions are unchanged: min{1,x} < alpha iff x < alpha.
v2: rank-based Monte-Carlo p-value decisions, p = (1+#{T* at least as extreme})/(B+1) <= alpha
    (interpolated quantiles are anti-conservative at finite B: ((B-1)a+1)/(B+1) = 0.0545 at
    B=199); adds 'e1', the calibrated single-ordering mixture e-value, to isolate aggregation
    from the base statistic. (Review credit: GPT Astra.)
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
    return dict(single=Ps[..., 0], pmerge=2 * Ps.mean(-1),   # uncapped (v3)
                bonf=Mn * Ps.min(-1),
                eavg=np.exp(logsumexp(logE, -1) - np.log(Mn)),
                e1=np.exp(logE[..., 0]), fisher=pf[..., 0])


def draw(mu, Sm, k): return mu + rng.standard_normal((k, N)) @ np.linalg.cholesky(Sm).T


def mc_low(bootv, obs, alpha):
    sb = np.sort(bootv)
    return 1 + np.searchsorted(sb, obs, side='right') <= alpha * (len(bootv) + 1)


def mc_high(bootv, obs, alpha):
    sb = np.sort(bootv)
    return 1 + (len(bootv) - np.searchsorted(sb, obs, side='left')) <= alpha * (len(bootv) + 1)


PK = ["single", "pmerge", "bonf", "fisher", "sym"]; EK = ["eavg", "e1", "chi2"]
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
            acc[k]["cs"] += mc_low(boot[k], nul[k], ALPHA).mean()
            acc[k]["cp"] += mc_low(boot[k], alt[k], ALPHA).mean()
            acc[k]["ns"] += (nul[k] < ALPHA).mean()
        for k in EK:
            acc[k]["cs"] += mc_high(boot[k], nul[k], ALPHA).mean()
            acc[k]["cp"] += mc_high(boot[k], alt[k], ALPHA).mean()
        acc["eavg"]["ns"] += (nul["eavg"] >= 1 / ALPHA).mean()
        acc["e1"]["ns"] += (nul["e1"] >= 1 / ALPHA).mean()
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
