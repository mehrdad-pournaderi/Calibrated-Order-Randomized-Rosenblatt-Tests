"""
MULTIPLICITY LAYER with HETEROGENEOUS departures -- the realistic screening problem:
the non-null hypotheses do not share a shape (half fail on one coordinate, half on all),
so no shape-tuned test is right for all of them.

Design notes:
 * B must exceed N_units/q, otherwise the bootstrap p-value cannot reach the smallest
   BH threshold q/N_units and every method is capped by calibration granularity.
   Here N_units=200, q=0.10 -> need B > 2000; we use B=2500.
 * ncp=20 so that the calibrated powers sit in an informative range.
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

import sys
_seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(1414 + 1000 * _seed)
Q = 0.10; LOG2 = np.log(2.0)
N, M, NCP, RHO = 20, 12, 20.0, 0.5
TAUS = np.array([1.0, 2.0, 3.0])
NUNITS, PI0 = 200, 0.95     # sparse screen, matching the regime of Sec. 7.2
R, B, BCHUNK = 25, 2500, 500  # replications are pooled across seeded chunks; with only
                              # 10 non-nulls the realized FDP is very variable
NTR_GRID = [80, 160, 0]
idx = np.arange(1, N + 1); ridge = 1e-3 * np.eye(N)

S = np.full((N, N), RHO); np.fill_diagonal(S, 1.0)
Sinv_true = np.linalg.inv(S); Lt = np.linalg.cholesky(S)
n_null = int(round(NUNITS * PI0)); n_alt = NUNITS - n_null
truth_alt = np.zeros(NUNITS, bool); truth_alt[n_null:] = True

v_sp = np.zeros(N); v_sp[0] = 1.0
v_dn = np.ones(N)
MU = np.zeros((NUNITS, N))
MU[n_null:n_null + n_alt // 2] = v_sp * np.sqrt(NCP / (v_sp @ Sinv_true @ v_sp))
MU[n_null + n_alt // 2:] = v_dn * np.sqrt(NCP / (v_dn @ Sinv_true @ v_dn))


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (N * np.sort(p2, axis=-1) / idx).min(-1)


def invsqrt(Sb):
    w, V = np.linalg.eigh(Sb)
    return (V * (w ** -0.5)[..., None, :]) @ np.swapaxes(V, -1, -2)


def order_stats(Z):
    Ps = simes2(Z); Mn = Z.shape[-2]
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    lE = logsumexp(terms, axis=(0, -1)) - (np.log(N) + np.log(len(TAUS)))
    return dict(single=Ps[..., 0], pmerge=np.minimum(2 * Ps.mean(-1), 1.0),
                bonf=np.minimum(Mn * Ps.min(-1), 1.0),
                eavg=np.exp(logsumexp(lE, -1) - np.log(Mn)))


def draw(mu, Sm, k): return mu + rng.standard_normal((k, N)) @ np.linalg.cholesky(Sm).T


def bh(p, q):
    m = len(p); o = np.argsort(p); ok = np.where(p[o] <= q * np.arange(1, m + 1) / m)[0]
    r = np.zeros(m, bool)
    if len(ok): r[o[:ok.max() + 1]] = True
    return r


def ebh(e, q):
    m = len(e); o = np.argsort(-e)
    ok = np.where(np.arange(1, m + 1) * e[o] >= m / q)[0]
    r = np.zeros(m, bool)
    if len(ok): r[o[:ok.max() + 1]] = True
    return r


PSMALL = ["single", "pmerge", "bonf", "sym"]; ELARGE = ["eavg", "chi2"]
ALLK = PSMALL + ELARGE
if len(sys.argv) > 1:
    R = int(sys.argv[1])
# raw per-replication proportions, so that chunks pool exactly and standard errors are exact
RAW = {m: {k: {"fdr": [], "pow": []} for k in ALLK} for m in ("naive", "cal")}

print("Heterogeneous screening (n=%d rho=%.1f M=%d ncp=%.0f q=%.2f, %d units, pi0=%.1f,"
      " half-sparse/half-dense, R=%d B=%d)" % (N, RHO, M, NCP, Q, NUNITS, PI0, R, B))
for Ntr in NTR_GRID:
    acc = {m: {k: {"fdr": [], "pow": []} for k in ALLK} for m in ("naive", "cal")}
    for _ in range(R):
        perms = np.stack([rng.permutation(N) for _ in range(M)])
        Shat = S if Ntr == 0 else (lambda X: X.T @ X / Ntr + ridge)(draw(np.zeros(N), S, Ntr))
        Li = np.linalg.inv(np.linalg.cholesky(Shat[perms[:, :, None], perms[:, None, :]]))
        Sih = np.linalg.inv(Shat); Wsy = invsqrt(Shat)
        # ---- bootstrap null pool, chunked over B ----
        pool = {k: [] for k in ALLK}
        for s in range(0, B, BCHUNK):
            b = min(BCHUNK, B - s)
            Vc = draw(np.zeros(N), Shat, b)
            if Ntr == 0:
                bs = order_stats(np.einsum('mij,bmj->bmi', Li, Vc[:, perms]))
                bs["chi2"] = np.einsum('bi,ij,bj->b', Vc, Sih, Vc)
                bs["sym"] = simes2(Vc @ Wsy.T)
            else:
                Xtb = draw(np.zeros(N), Shat, b * Ntr).reshape(b, Ntr, N)
                Sst = np.einsum('bki,bkj->bij', Xtb, Xtb) / Ntr + ridge
                Lst = np.linalg.inv(np.linalg.cholesky(Sst[:, perms[:, :, None], perms[:, None, :]]))
                bs = order_stats(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]))
                bs["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
                bs["sym"] = simes2(np.einsum('bij,bj->bi', invsqrt(Sst), Vc))
            for k in ALLK: pool[k].append(bs[k])
        boot = {k: np.concatenate(v) for k, v in pool.items()}
        # ---- the units ----
        X = MU + rng.standard_normal((NUNITS, N)) @ Lt.T
        st = order_stats(np.einsum('mij,kmj->kmi', Li, X[:, perms]))
        st["chi2"] = np.einsum('ki,ij,kj->k', X, Sih, X); st["sym"] = simes2(X @ Wsy.T)
        naive_rej = {k: bh(st[k], Q) for k in PSMALL}
        naive_rej["eavg"] = ebh(st["eavg"], Q)
        naive_rej["chi2"] = bh(stats.chi2.sf(st["chi2"], N), Q)
        cal_rej = {}
        for k in ALLK:
            bs = np.sort(boot[k])
            if k in PSMALL:
                cnt = np.searchsorted(bs, st[k], side="right")
            else:
                cnt = B - np.searchsorted(bs, st[k], side="left")
            cal_rej[k] = bh((1 + cnt) / (B + 1), Q)
        for mlab, rejd in (("naive", naive_rej), ("cal", cal_rej)):
            for k in ALLK:
                r = rejd[k]
                acc[mlab][k]["fdr"].append(np.sum(r & ~truth_alt) / max(r.sum(), 1))
                acc[mlab][k]["pow"].append(np.sum(r & truth_alt) / n_alt)
    lab = "known" if Ntr == 0 else str(Ntr)
    print("  N_tr=%s" % lab)
    for k in ALLK:
        for mlab in ("naive", "cal"):
            RAW[mlab][k]["fdr"].append(acc[mlab][k]["fdr"])
            RAW[mlab][k]["pow"].append(acc[mlab][k]["pow"])
        print("    %-7s naive[FDR=%.3f pow=%.3f]  calibrated[FDR=%.3f pow=%.3f]"
              % (k, np.mean(acc["naive"][k]["fdr"]), np.mean(acc["naive"][k]["pow"]),
                 np.mean(acc["cal"][k]["fdr"]), np.mean(acc["cal"][k]["pow"])))

fn = "chunk_multiplicity_s%d.npz" % _seed
np.savez(fn, Ntr=np.array([g if g else 100000 for g in NTR_GRID]), R=R,
         **{f"{m}_{k}_{s}": np.array(RAW[m][k][s])       # shape (n_Ntr, R)
            for m in ("naive", "cal") for k in ALLK for s in ("fdr", "pow")},
         q=Q, n=N, M=M, ncp=NCP, rho=RHO, nunits=NUNITS, pi0=PI0, B=B)
print("saved " + fn)
