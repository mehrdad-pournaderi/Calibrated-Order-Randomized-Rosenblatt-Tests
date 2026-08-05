"""
Robustness sweep for the bootstrap-calibrated comparison.
Usage: python3 sim7.py {n|rho|alt}

CORRECTED VERSION (v2):
 - fresh random orderings drawn in EVERY realization (the 'single' baseline is the
   expected power of an arbitrarily chosen ordering, per the paper's definition);
 - deterministic per-mode seeds (no hash(), which is process-randomized);
 - order-invariant references chi2 and sym (Sigma^{-1/2} whitening + two-sided Simes).

For each config, at N_train = 4n (heavy estimation) and known F:
  - calibrated null size per method (should be ~alpha)
  - calibrated power per method
  - paired power gaps (p-merge - single) and (e-avg - single), mean +/- SE
Alternatives matched on Mahalanobis energy ncp=12 so detectability is comparable.
"""
import sys
import numpy as np
from scipy import stats
from scipy.special import logsumexp

ALPHA = 0.05; LOG2 = np.log(2.0)
TAUS = np.array([1.0, 2.0, 3.0])
R, B, NTE, NCP = 150, 199, 160, 12.0
MFIX = 12                                   # fixed # orderings (remove M-vs-n confound)
SEEDS = {"n": 7001, "rho": 7002, "alt": 7003}


def equicorr(n, rho):
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); return S


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def linv1(S, perms):
    Sp = S[perms[:, :, None], perms[:, None, :]]
    return np.linalg.inv(np.linalg.cholesky(Sp))


def linvB(Sb, perms):
    Sp = Sb[:, perms[:, :, None], perms[:, None, :]]
    return np.linalg.inv(np.linalg.cholesky(Sp))


def invsqrt(Sb):
    w, V = np.linalg.eigh(Sb)
    return (V * (w ** -0.5)[..., None, :]) @ np.swapaxes(V, -1, -2)


def simes2(Z, n):
    ix = np.arange(1, n + 1)
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (n * np.sort(p2, axis=-1) / ix).min(-1)


def order_methods(Z, n):
    M = Z.shape[-2]
    ix = np.arange(1, n + 1)
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    Psimes = (n * np.sort(p2, axis=-1) / ix).min(-1)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    logE = logsumexp(terms, axis=(0, -1)) - (np.log(n) + np.log(len(TAUS)))
    pfish = stats.chi2.sf(-2.0 * np.log(np.maximum(p2, 1e-300)).sum(-1), 2 * n)
    return dict(single=Psimes[..., 0],
                pmerge=np.minimum(2 * Psimes.mean(-1), 1.0),
                bonf=np.minimum(M * Psimes.min(-1), 1.0),
                eavg=np.exp(logsumexp(logE, -1) - np.log(M)),
                fisher=pfish[..., 0],
                pmerge_f=np.minimum(2 * pfish.mean(-1), 1.0),
                bonf_f=np.minimum(M * pfish.min(-1), 1.0))


def draw(mu, S, k, rng): return mu + rng.standard_normal((k, S.shape[0])) @ np.linalg.cholesky(S).T
PKEYS = ["single", "pmerge", "bonf", "sym", "fisher", "pmerge_f", "bonf_f"]
EKEYS = ["eavg", "chi2"]
ALLK = ["single", "pmerge", "bonf", "eavg", "chi2", "sym", "fisher", "pmerge_f", "bonf_f"]


def compare(Sigma, Ntr, alt_type, rng):
    n = Sigma.shape[0]; Sinv = np.linalg.inv(Sigma); ridge = 1e-3 * np.eye(n)
    if alt_type == "sparse":
        mu = np.zeros(n); mu[0] = np.sqrt(NCP / Sinv[0, 0])
    else:                                   # dense: shift all coords, matched ncp
        one = np.ones(n); mu = np.sqrt(NCP / (one @ Sinv @ one)) * one
    csize = {k: 0.0 for k in ALLK}; cpow = {k: 0.0 for k in ALLK}
    pr = {k: [] for k in ALLK}
    for _ in range(R):
        perms = np.stack([rng.permutation(n) for _ in range(MFIX)])   # FRESH orderings
        if Ntr == 0:
            Shat = Sigma; Linv_r = linv1(Shat, perms)
            Sinv_hat = np.linalg.inv(Shat); Wsym = invsqrt(Shat)
            Vc = draw(np.zeros(n), Shat, B, rng)
            boot = order_methods(np.einsum('mij,bmj->bmi', Linv_r, Vc[:, perms]), n)
            boot["chi2"] = np.einsum('ki,ij,kj->k', Vc, Sinv_hat, Vc)
            boot["sym"] = simes2(Vc @ Wsym.T, n)
        else:
            Shat = (lambda X: X.T @ X / Ntr + ridge)(draw(np.zeros(n), Sigma, Ntr, rng))
            Linv_r = linv1(Shat, perms)
            Sinv_hat = np.linalg.inv(Shat); Wsym = invsqrt(Shat)
            Xtb = draw(np.zeros(n), Shat, B * Ntr, rng).reshape(B, Ntr, n)
            Sst = np.einsum('bki,bkj->bij', Xtb, Xtb) / Ntr + ridge
            Lst = linvB(Sst, perms)
            Vc = draw(np.zeros(n), Shat, B, rng)
            boot = order_methods(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]), n)
            boot["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
            boot["sym"] = simes2(np.einsum('bij,bj->bi', invsqrt(Sst), Vc), n)
        Xn = draw(np.zeros(n), Sigma, NTE, rng); Xa = draw(mu, Sigma, NTE, rng)
        def allst(X):
            st = order_methods(np.einsum('mij,kmj->kmi', Linv_r, X[:, perms]), n)
            st["chi2"] = np.einsum('ki,ij,kj->k', X, Sinv_hat, X)
            st["sym"] = simes2(X @ Wsym.T, n)
            return st
        nul, alt = allst(Xn), allst(Xa)
        for k in PKEYS:
            c = np.quantile(boot[k], ALPHA)
            csize[k] += (nul[k] <= c).mean(); v = (alt[k] <= c).mean(); cpow[k] += v; pr[k].append(v)
        for k in EKEYS:
            c = np.quantile(boot[k], 1 - ALPHA)
            csize[k] += (nul[k] >= c).mean(); v = (alt[k] >= c).mean(); cpow[k] += v; pr[k].append(v)
    def gap(kk):
        d = np.array(pr[kk]) - np.array(pr["single"])
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(R))
    return ({k: csize[k] / R for k in ALLK}, {k: cpow[k] / R for k in ALLK},
            gap("pmerge"), gap("eavg"))


mode = sys.argv[1] if len(sys.argv) > 1 else "n"
rng = np.random.default_rng(SEEDS.get(mode, 7000))
if mode == "n":
    configs = [("n=5", 5, 0.5, "sparse"), ("n=10", 10, 0.5, "sparse"), ("n=20", 20, 0.5, "sparse")]
elif mode == "rho":
    configs = [("rho=0.2", 10, 0.2, "sparse"), ("rho=0.5", 10, 0.5, "sparse"), ("rho=0.8", 10, 0.8, "sparse")]
else:
    configs = [("sparse", 10, 0.5, "sparse"), ("dense", 10, 0.5, "dense")]

labels = []; out = {f"{m}_{k}": [] for m in ["csz", "cpw", "cpw_known"] for k in ALLK}
for L in ["gm_s", "ge_s", "gm_k", "ge_k", "gme_s", "gee_s", "gme_k", "gee_k"]: out[L] = []
print("MODE=%s  (R=%d B=%d NTE=%d ncp=%.0f, fresh orderings per realization)" % (mode, R, B, NTE, NCP))
for name, n, rho, alt in configs:
    Nsmall = int(round(4 * n))
    cs_s, cp_s, gp_s, ge_s2 = compare(equicorr(n, rho), Nsmall, alt, rng)
    cs_k, cp_k, gp_k, ge_k2 = compare(equicorr(n, rho), 0, alt, rng)
    labels.append(name)
    for k in ALLK:
        out[f"csz_{k}"].append(cs_s[k]); out[f"cpw_{k}"].append(cp_s[k])
        out[f"cpw_known_{k}"].append(cp_k[k])
    out["gm_s"].append(gp_s[0]); out["ge_s"].append(gp_s[1])
    out["gm_k"].append(gp_k[0]); out["ge_k"].append(gp_k[1])
    out["gme_s"].append(ge_s2[0]); out["gee_s"].append(ge_s2[1])
    out["gme_k"].append(ge_k2[0]); out["gee_k"].append(ge_k2[1])
    print("  %-9s (M=%d,Nsmall=%d): size[" % (name, MFIX, Nsmall) +
          " ".join("%s=%.3f" % (k, cs_s[k]) for k in ALLK) + "]")
    print("     power small[" + " ".join("%s=%.3f" % (k, cp_s[k]) for k in ALLK) + "]")
    print("     power known[" + " ".join("%s=%.3f" % (k, cp_k[k]) for k in ALLK) + "]")
    print("     gap pmerge: small=%.4f±%.4f known=%.4f±%.4f | gap eavg: small=%.4f±%.4f known=%.4f±%.4f"
          % (gp_s[0], gp_s[1], gp_k[0], gp_k[1], ge_s2[0], ge_s2[1], ge_k2[0], ge_k2[1]))

np.savez("results_shape_%s.npz" % mode,
         labels=np.array(labels), **{k: np.array(v) for k, v in out.items()},
         alpha=ALPHA, ncp=NCP)
print("saved results_shape_%s.npz" % mode)
