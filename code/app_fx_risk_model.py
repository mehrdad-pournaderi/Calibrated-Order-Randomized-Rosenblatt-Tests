"""
APPLICATION: validating a Gaussian FX risk model with order-randomized Rosenblatt tests.

Data: FRED H.10 daily noon spot rates, 9 major currencies vs USD, 2015-2025
      (files fx_DEX*.csv fetched from fred.stlouisfed.org/graph/fredgraph.csv
      on 2026-07-15; series DEXUSEU DEXJPUS DEXUSUK DEXCAUS DEXSZUS DEXUSAL
      DEXUSNZ DEXSDUS DEXNOUS).
Model: for each test year y (2016..2025), fit mu_hat, Sigma_hat on year y-1 log-returns
      (~250 days); the per-day null is H0: X_t ~ N(mu_hat, Sigma_hat).
Tests (all bootstrap-calibrated per year by Procedure 1, B replicates, alpha=0.05):
      single random ordering (two-sided Simes), pooled orderings (p-merge, Bonferroni,
      e-value average; M=12 fresh orderings per day), chi2 and symmetric-root references.
Also: naive e-BH day screening at q=0.1 on the raw averaged e-values, and top flagged days.
"""
import glob
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(20260715)
ALPHA, Q_EBH = 0.05, 0.10
M, B = 12, 999
TAUS = np.array([1.0, 2.0, 3.0])
LOG2 = np.log(2.0)

# ---------------- load & align ----------------
SERIES = ["DEXUSEU", "DEXJPUS", "DEXUSUK", "DEXCAUS", "DEXSZUS",
          "DEXUSAL", "DEXUSNZ", "DEXSDUS", "DEXNOUS"]
PER_USD = {"DEXJPUS", "DEXCAUS", "DEXSZUS", "DEXSDUS", "DEXNOUS"}  # quoted as FC per USD
data = {}
for s in SERIES:
    d, v = [], []
    with open(f"fx_{s}.csv") as f:
        next(f)
        for line in f:
            a, b = line.strip().split(",")
            if b not in (".", ""):
                d.append(a); v.append(float(b))
    data[s] = dict(zip(d, v))
dates = sorted(set.intersection(*[set(data[s]) for s in SERIES]))
X = np.array([[data[s][t] for s in SERIES] for t in dates])
logp = np.log(X)
for j, s in enumerate(SERIES):                      # express all as USD per foreign unit
    if s in PER_USD:
        logp[:, j] = -logp[:, j]
ret = np.diff(logp, axis=0) * 100.0                 # daily log-returns in percent
rdates = dates[1:]
years = np.array([int(t[:4]) for t in rdates])
n = ret.shape[1]
idx = np.arange(1, n + 1)
print(f"panel: {ret.shape[0]} days x {n} currencies, {rdates[0]} .. {rdates[-1]}")

# ---------------- test machinery ----------------
def logcosh(x): return np.logaddexp(x, -x) - LOG2


def invsqrt(S):
    w, V = np.linalg.eigh(S)
    return (V * (w ** -0.5)[..., None, :]) @ np.swapaxes(V, -1, -2)


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (n * np.sort(p2, axis=-1) / idx).min(-1)


def rand_perms(k, m):
    return np.argsort(rng.random((k, m, n)), axis=-1)


def order_stats(Z):
    """Z: (k, M, n) whitened scores -> per-day combined statistics."""
    Ps = simes2(Z)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    logE = logsumexp(terms, axis=(0, -1)) - (np.log(n) + np.log(len(TAUS)))
    return dict(single=Ps[..., 0],
                pmerge=np.minimum(2 * Ps.mean(-1), 1.0),
                bonf=np.minimum(M * Ps.min(-1), 1.0),
                eavg=logsumexp(logE, -1) - np.log(M))          # log e-value


def all_stats(V, Linv, Sinv, W):
    """V: (k, n) centered vectors; Linv: (n,n) chol-inverse of Sigma_hat (identity ordering);
    fresh M orderings per row. Exchangeability does NOT hold here, so permute Sigma too."""
    P = rand_perms(V.shape[0], M)                              # (k,M,n)
    st = {}
    # per-row, per-ordering cholesky of permuted Sigma_hat: batch over unique perms is
    # overkill at k<=260, M=12 -> loop days in chunks
    Zs = np.empty((V.shape[0], M, n))
    Sig = np.linalg.inv(Sinv)
    for i in range(V.shape[0]):
        for m in range(M):
            p = P[i, m]
            L = np.linalg.cholesky(Sig[np.ix_(p, p)])
            Zs[i, m] = np.linalg.solve(L, V[i, p])
    st.update(order_stats(Zs))
    st["chi2"] = np.einsum('ki,ij,kj->k', V, Sinv, V)
    st["sym"] = simes2(V @ W.T)
    return st


def calibrate(Shat, Ntr):
    """Procedure 1: B replicates with re-estimated Sigma*, fresh orderings per replicate."""
    A = np.linalg.cholesky(Shat)
    Xtr = rng.standard_normal((B, Ntr, n)) @ A.T
    Sst = np.einsum('bki,bkj->bij', Xtr - Xtr.mean(1, keepdims=True),
                    Xtr - Xtr.mean(1, keepdims=True)) / (Ntr - 1) + 1e-8 * np.eye(n)
    Vc = rng.standard_normal((B, n)) @ A.T
    P = rand_perms(B, M)
    Zc = np.empty((B, M, n))
    for b in range(B):
        for m in range(M):
            p = P[b, m]
            L = np.linalg.cholesky(Sst[b][np.ix_(p, p)])
            Zc[b, m] = np.linalg.solve(L, Vc[b, p])
    boot = order_stats(Zc)
    boot["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
    Wb = invsqrt(Sst)
    boot["sym"] = simes2(np.einsum('bij,bj->bi', Wb, Vc))
    crit = {k: np.quantile(boot[k], ALPHA) for k in ("single", "pmerge", "bonf", "sym")}
    crit["eavg"] = np.quantile(boot["eavg"], 1 - ALPHA)
    crit["chi2"] = np.quantile(boot["chi2"], 1 - ALPHA)
    return crit


# ---------------- run per year ----------------
KEYS = ["single", "pmerge", "bonf", "eavg", "chi2", "sym"]
rows, gaps, logE_all, dates_all, rej_flags = [], [], [], [], []
for y in range(2016, 2026):
    tr = ret[years == y - 1]; te = ret[years == y]
    dte = [t for t, yy in zip(rdates, years) if yy == y]
    if len(tr) < 100 or len(te) < 100:
        continue
    mu, Shat = tr.mean(0), np.cov(tr.T, ddof=1) + 1e-8 * np.eye(n)
    Sinv, W = np.linalg.inv(Shat), invsqrt(Shat)
    Linv = np.linalg.inv(np.linalg.cholesky(Shat))
    crit = calibrate(Shat, len(tr))
    st = all_stats(te - mu, Linv, Sinv, W)
    rej = {k: (st[k] <= crit[k]) for k in ("single", "pmerge", "bonf", "sym")}
    rej["eavg"] = st["eavg"] >= crit["eavg"]
    rej["chi2"] = st["chi2"] >= crit["chi2"]
    rows.append((y, len(te)) + tuple(float(rej[k].mean()) for k in KEYS))
    gaps.append(rej["eavg"].astype(float) - rej["single"].astype(float))
    logE_all.append(st["eavg"]); dates_all.extend(dte); rej_flags.append(rej["eavg"])
    print("  %d (T=%d): " % (y, len(te)) +
          "  ".join("%s=%.3f" % (k, rej[k].mean()) for k in KEYS))

gap = np.concatenate(gaps)
print("\npaired day-level gap (e-avg - single): %.4f +/- %.4f (T=%d days, t=%.1f)"
      % (gap.mean(), gap.std(ddof=1) / np.sqrt(len(gap)), len(gap),
         gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap)))))

# ---------------- naive e-BH across all days ----------------
logE = np.concatenate(logE_all); E = np.exp(logE); N = len(E)
srt = np.argsort(-E); k = np.arange(1, N + 1)
ok = E[srt] >= N / (Q_EBH * k)
kstar = k[ok].max() if ok.any() else 0
sel = srt[:kstar]
print("\ne-BH at q=%.2f on raw averaged e-values: %d of %d days flagged" % (Q_EBH, kstar, N))
top = np.argsort(-E)[:15]
print("top days by pooled e-value:")
for i in top:
    print("   %s  E=%9.1f" % (dates_all[i], E[i]))

np.savez("results_fx_application.npz",
         rows=np.array(rows), keys=np.array(KEYS),
         gap_mean=gap.mean(), gap_se=gap.std(ddof=1) / np.sqrt(len(gap)),
         logE=logE, dates=np.array(dates_all),
         rej_eavg=np.concatenate(rej_flags),
         ebh_flag=np.isin(np.arange(N), sel), alpha=ALPHA, q=Q_EBH, M=M, B=B)
print("saved fx_results.npz")
