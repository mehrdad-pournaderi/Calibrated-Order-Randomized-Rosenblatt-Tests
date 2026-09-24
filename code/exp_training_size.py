"""
APPLES-TO-APPLES under estimated F: parametric-bootstrap-calibrate every method to
size alpha, then compare POWER.  The bootstrap re-estimates Sigma INSIDE each replicate
(double bootstrap) so the calibration absorbs the same estimation noise the real test has.

v3: calibration statistics are the UNCAPPED merger summaries mean_m P_m and M*min_m P_m
 (the capped p-values min{1,.} are monotone in these below the cap and identical for every
 decision here; capping only matters when it creates ties at 1, i.e. when more than a 1-alpha
 fraction of null draws hit the cap, which happens at large M under weak dependence).
 Naive decisions are unchanged: min{1,x} < alpha iff x < alpha.

CORRECTED VERSION (v2):
 - fresh random orderings are drawn in EVERY realization, so the 'single' baseline
   estimates the EXPECTED power of an arbitrarily chosen ordering (the paper's
   definition), not the power of one fixed lucky/unlucky permutation;
 - the alternative is matched on Mahalanobis energy ncp=12 (protocol), not raw delta;
 - two order-invariant references included: chi2 = X'Sigma^-1 X, and SYM = two-sided
   Simes on the symmetric-root whitening Sigma^{-1/2} X;
 - deterministic seeding throughout.

Pipeline (per realization):
  draw M fresh orderings; real training (N_train null vectors) -> Sigma_hat;
  bootstrap null: B replicates, each re-estimates Sigma*_b from a fresh training sample
     drawn from N(0,Sigma_hat) and whitens a fresh calibration vector by Sigma*_b
     -> calibrated critical value per method;  evaluate on fresh null/alt test batches.
Methods (two-sided base): single random order (Simes), p-merge, Bonferroni/orders,
e-value avg, chi2 (order-invariant), sym (order-invariant).
"""
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(505)
ALPHA = 0.05; LOG2 = np.log(2.0)
N, MORD = 8, 12
TAUS = np.array([1.0, 2.0, 3.0])
idx = np.arange(1, N + 1)
NCP = 12.0


def equicorr(n, rho):
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); return S


Sigma_true = equicorr(N, 0.5)
Sinv_true = np.linalg.inv(Sigma_true)
MU = np.zeros(N); MU[0] = np.sqrt(NCP / Sinv_true[0, 0])   # ncp = mu' Sinv mu = 12


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def linv1(S, perms):                            # (n,n) -> (M,n,n)
    Sp = S[perms[:, :, None], perms[:, None, :]]
    return np.linalg.inv(np.linalg.cholesky(Sp))


def linvB(Sb, perms):                           # (B,n,n) -> (B,M,n,n)
    Sp = Sb[:, perms[:, :, None], perms[:, None, :]]
    return np.linalg.inv(np.linalg.cholesky(Sp))


def invsqrt(Sb):                                # batched symmetric inverse root
    w, V = np.linalg.eigh(Sb)
    return (V * (w ** -0.5)[..., None, :]) @ np.swapaxes(V, -1, -2)


def simes2(Z, n, ix):                           # two-sided Simes on last axis
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    Ps = np.sort(p2, axis=-1)
    return (n * Ps / ix).min(-1)


def order_methods(Z):                           # Z (...,M,n) -> dict of (...,) stats
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    Ps = np.sort(p2, axis=-1)
    Psimes = (N * Ps / idx).min(-1)                                     # (...,M)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])     # (T,...,M,n)
    logE = logsumexp(terms, axis=(0, -1)) - (np.log(N) + np.log(len(TAUS)))
    M = Psimes.shape[-1]
    pfish = stats.chi2.sf(-2.0 * np.log(np.maximum(p2, 1e-300)).sum(-1), 2 * N)  # (...,M)
    return dict(single=Psimes[..., 0],
                pmerge=2 * Psimes.mean(-1),          # uncapped (v3); min{1,.} only for reporting
                bonf=M * Psimes.min(-1),
                eavg=np.exp(logsumexp(logE, -1) - np.log(M)),
                fisher=pfish[..., 0],
                pmerge_f=2 * pfish.mean(-1),
                bonf_f=M * pfish.min(-1))


PKEYS = ["single", "pmerge", "bonf", "sym", "fisher", "pmerge_f", "bonf_f"]  # small = extreme
EKEYS = ["eavg", "chi2"]                                                     # large = extreme
ALLK = ["single", "pmerge", "bonf", "eavg", "chi2", "sym", "fisher", "pmerge_f", "bonf_f"]
CHI2_CRIT = stats.chi2.isf(ALPHA, N)


def naive_reject(st):
    return dict(single=st["single"] < ALPHA, pmerge=st["pmerge"] < ALPHA,
                bonf=st["bonf"] < ALPHA, sym=st["sym"] < ALPHA,
                fisher=st["fisher"] < ALPHA, pmerge_f=st["pmerge_f"] < ALPHA,
                bonf_f=st["bonf_f"] < ALPHA,
                eavg=st["eavg"] >= 1 / ALPHA, chi2=st["chi2"] >= CHI2_CRIT)


def mc_low(bootv, obs, alpha):
    """Rank-based MC p-value decision (finite-B exact under exchangeability); interpolated
    quantiles are anti-conservative: expected rejection ((B-1)a+1)/(B+1)."""
    sb = np.sort(bootv)
    return 1 + np.searchsorted(sb, obs, side='right') <= alpha * (len(bootv) + 1)


def mc_high(bootv, obs, alpha):
    sb = np.sort(bootv)
    return 1 + (len(bootv) - np.searchsorted(sb, obs, side='left')) <= alpha * (len(bootv) + 1)


def evalrej(boot, null, alt):
    out = {}
    for k in PKEYS:
        out[k] = (float(mc_low(boot[k], null[k], ALPHA).mean()),
                  float(mc_low(boot[k], alt[k], ALPHA).mean()),
                  float(naive_reject(null)[k].mean()))
    for k in EKEYS:
        out[k] = (float(mc_high(boot[k], null[k], ALPHA).mean()),
                  float(mc_high(boot[k], alt[k], ALPHA).mean()),
                  float(naive_reject(null)[k].mean()))
    return out


def draw(mu, S, k):
    return mu + rng.standard_normal((k, S.shape[0])) @ np.linalg.cholesky(S).T


def all_stats(X, Linv_r, Sinv_hat, Wsym, perms):
    """order stats via Linv_r (M,n,n); chi2 via Sinv_hat; sym via Wsym."""
    st = order_methods(np.einsum('mij,kmj->kmi', Linv_r, X[:, perms]))
    st["chi2"] = np.einsum('ki,ij,kj->k', X, Sinv_hat, X)
    st["sym"] = simes2(X @ Wsym.T, N, idx)
    return st


Ntr_grid = [40, 80, 160, 0]                     # 0 = known F
R, B, NTE = 300, 299, 200
CS = {k: {"csize": [], "cpow": [], "nsize": []} for k in ALLK}
GAP = {g: {"mean": [], "se": []} for g in ["pm", "ea"]}   # paired vs single
print("Bootstrap-calibrated comparison v2 (n=%d, M=%d, rho=0.5, ncp=%.0f, alpha=%.2f, R=%d, B=%d)"
      % (N, MORD, NCP, ALPHA, R, B))
print("  ('single' = fresh random ordering each realization -> expected-ordering power)")
for Ntr in Ntr_grid:
    agg = {k: {"csize": 0., "cpow": 0., "nsize": 0.} for k in ALLK}
    perreal = {k: [] for k in ALLK}
    for _ in range(R):
        perms = np.stack([rng.permutation(N) for _ in range(MORD)])   # FRESH orderings
        if Ntr == 0:
            Shat = Sigma_true
            Linv_r = linv1(Shat, perms)
            Sinv_hat = np.linalg.inv(Shat); Wsym = invsqrt(Shat)
            Vc = draw(np.zeros(N), Shat, B)
            boot = order_methods(np.einsum('mij,bmj->bmi', Linv_r, Vc[:, perms]))
            boot["chi2"] = np.einsum('ki,ij,kj->k', Vc, Sinv_hat, Vc)
            boot["sym"] = simes2(Vc @ Wsym.T, N, idx)
        else:
            Shat = (lambda Xt: Xt.T @ Xt / Ntr + 1e-3 * np.eye(N))(draw(np.zeros(N), Sigma_true, Ntr))
            Linv_r = linv1(Shat, perms)
            Sinv_hat = np.linalg.inv(Shat); Wsym = invsqrt(Shat)
            Xtb = draw(np.zeros(N), Shat, B * Ntr).reshape(B, Ntr, N)
            Sst = np.einsum('bki,bkj->bij', Xtb, Xtb) / Ntr + 1e-3 * np.eye(N)
            Lst = linvB(Sst, perms)                       # (B,M,n,n)
            Vc = draw(np.zeros(N), Shat, B)
            boot = order_methods(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]))
            boot["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
            boot["sym"] = simes2(np.einsum('bij,bj->bi', invsqrt(Sst), Vc), N, idx)
        Xn = draw(np.zeros(N), Sigma_true, NTE)
        Xa = draw(MU, Sigma_true, NTE)
        nul = all_stats(Xn, Linv_r, Sinv_hat, Wsym, perms)
        alt = all_stats(Xa, Linv_r, Sinv_hat, Wsym, perms)
        ev = evalrej(boot, nul, alt)
        for k in ALLK:
            agg[k]["csize"] += ev[k][0]; agg[k]["cpow"] += ev[k][1]; agg[k]["nsize"] += ev[k][2]
            perreal[k].append(ev[k][1])
    lab = "known" if Ntr == 0 else str(Ntr)
    print("  N_train=%6s" % lab)
    for k in ALLK:
        cs, cp, ns = agg[k]["csize"]/R, agg[k]["cpow"]/R, agg[k]["nsize"]/R
        CS[k]["csize"].append(cs); CS[k]["cpow"].append(cp); CS[k]["nsize"].append(ns)
        print("     %-7s naive-size=%.3f  calibrated-size=%.3f  calibrated-POWER=%.3f"
              % (k, ns, cs, cp))
    for g, kk in [("pm", "pmerge"), ("ea", "eavg")]:
        diff = np.array(perreal[kk]) - np.array(perreal["single"])
        m, s = float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(R))
        GAP[g]["mean"].append(m); GAP[g]["se"].append(s)
        print("     >> gap (%s - single): %.4f +/- %.4f (SE)  [t=%.1f]"
              % (kk, m, s, m / (s + 1e-12)))

np.savez("results_training_size.npz",
         Ntr=np.array([g if g != 0 else 100000 for g in Ntr_grid]),
         **{f"{k}_{m}": np.array(CS[k][m]) for k in ALLK for m in ["csize", "cpow", "nsize"]},
         gap_mean=np.array(GAP["pm"]["mean"]), gap_se=np.array(GAP["pm"]["se"]),
         gape_mean=np.array(GAP["ea"]["mean"]), gape_se=np.array(GAP["ea"]["se"]),
         alpha=ALPHA, ncp=NCP, N=N, M=MORD)
print("saved sim6.npz")
