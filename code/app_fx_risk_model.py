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
      The e-BH layer is a statement about the ISSUED FORECAST null H0: X_t ~ N(mu_hat,
      Sigma_hat), under which the whitened scores are exactly standard normal and the raw
      e-values are valid; it is not a claim under the estimated-population null.

v2 (review credit: GPT Astra):
  - rank-based Monte-Carlo p-value decisions, p = (1+#{T* at least as extreme})/(B+1),
    replacing interpolated quantile thresholds (anti-conservative at finite B);
  - the bootstrap now re-estimates the MEAN inside each replicate and centers the calibration
    vector with it, so mean-estimation uncertainty is carried by the calibration null exactly
    as in the real pipeline (previously only the covariance was re-estimated);
  - the day-level paired-gap standard error uses a Newey-West (Bartlett) estimator to account
    for serial dependence.
v3 (review credit: GPT Astra, round 2):
  - TWO calibration targets are now run and stored separately, because they are different nulls
    and need different simulations:
      (a) ESTIMATED-POPULATION null (rows, gap_*): the re-estimating bootstrap above, i.e.
          Procedure 1 with mean and covariance re-estimated inside every replicate;
      (b) ISSUED-FORECAST null H0,t: X_t | F_{t-1} ~ N(mu_hat_t, Sigma_hat_t) (rows_fc, gap_fc_*):
          the parameters are FIXED at the issued values, so the calibration vectors are drawn
          from N(mu_hat, Sigma_hat) and evaluated with the same fixed whitening as the observed
          days (exact Monte-Carlo null; no re-estimation).
  - merger summaries used for calibration are uncapped (mean and min of per-order p-values).
"""
import glob
import numpy as np
from scipy import stats
from scipy.special import logsumexp

rng = np.random.default_rng(20260715)
rng_fc = np.random.default_rng(20260924)   # separate stream for the forecast-null calibration (v3),
                                            # so the estimated-population run reproduces the earlier one
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


def rand_perms(k, m, gen=None):
    return np.argsort((gen or rng).random((k, m, n)), axis=-1)


def order_stats(Z):
    """Z: (k, M, n) whitened scores -> per-day combined statistics."""
    Ps = simes2(Z)
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    logE = logsumexp(terms, axis=(0, -1)) - (np.log(n) + np.log(len(TAUS)))
    return dict(single=Ps[..., 0],
                pmerge=2 * Ps.mean(-1),      # uncapped merger summaries (v3): the capped
                bonf=M * Ps.min(-1),         # p-values min{1,.} are for reporting only
                eavg=logsumexp(logE, -1) - np.log(M))          # log e-value


def all_stats(V, Linv, Sinv, W, gen=None):
    """V: (k, n) centered vectors; Linv: (n,n) chol-inverse of Sigma_hat (identity ordering);
    fresh M orderings per row. Exchangeability does NOT hold here, so permute Sigma too."""
    P = rand_perms(V.shape[0], M, gen)                         # (k,M,n)
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
    """Procedure 1: B replicates with re-estimated mean AND Sigma*, fresh orderings per
    replicate. Returns the sorted bootstrap statistics; decisions are rank-based."""
    A = np.linalg.cholesky(Shat)
    Xtr = rng.standard_normal((B, Ntr, n)) @ A.T
    mu_st = Xtr.mean(1)                                  # re-estimated mean, per replicate
    Sst = np.einsum('bki,bkj->bij', Xtr - mu_st[:, None, :],
                    Xtr - mu_st[:, None, :]) / (Ntr - 1) + 1e-8 * np.eye(n)
    # the real pipeline centers the test day by the ESTIMATED mean; carry that error here:
    Vc = rng.standard_normal((B, n)) @ A.T - mu_st
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
    return {k: np.sort(v) for k, v in boot.items()}


def calibrate_forecast(Shat, Linv, Sinv, W):
    """Issued-forecast null: parameters fixed at the issued (mu_hat, Sigma_hat); the calibration
    vectors are exact draws from the null, centred exactly, and whitened by the SAME fixed
    Sigma_hat as the observed days. Returns sorted statistics."""
    Vc = rng_fc.standard_normal((B, n)) @ np.linalg.cholesky(Shat).T
    boot = all_stats(Vc, Linv, Sinv, W, gen=rng_fc)
    return {k: np.sort(v) for k, v in boot.items()}


def mc_low(sorted_boot, obs):
    return 1 + np.searchsorted(sorted_boot, obs, side='right') <= ALPHA * (B + 1)


def mc_high(sorted_boot, obs):
    return 1 + (B - np.searchsorted(sorted_boot, obs, side='left')) <= ALPHA * (B + 1)


# ---------------- run per year ----------------
KEYS = ["single", "pmerge", "bonf", "eavg", "chi2", "sym"]
rows, gaps, logE_all, dates_all, rej_flags = [], [], [], [], []
rows_fc, gaps_fc = [], []                            # issued-forecast calibration
for y in range(2016, 2026):
    tr = ret[years == y - 1]; te = ret[years == y]
    dte = [t for t, yy in zip(rdates, years) if yy == y]
    if len(tr) < 100 or len(te) < 100:
        continue
    mu, Shat = tr.mean(0), np.cov(tr.T, ddof=1) + 1e-8 * np.eye(n)
    Sinv, W = np.linalg.inv(Shat), invsqrt(Shat)
    Linv = np.linalg.inv(np.linalg.cholesky(Shat))
    boot = calibrate(Shat, len(tr))
    st = all_stats(te - mu, Linv, Sinv, W)
    rej = {k: mc_low(boot[k], st[k]) for k in ("single", "pmerge", "bonf", "sym")}
    rej["eavg"] = mc_high(boot["eavg"], st["eavg"])
    rej["chi2"] = mc_high(boot["chi2"], st["chi2"])
    rows.append((y, len(te)) + tuple(float(rej[k].mean()) for k in KEYS))
    gaps.append(rej["eavg"].astype(float) - rej["single"].astype(float))
    logE_all.append(st["eavg"]); dates_all.extend(dte); rej_flags.append(rej["eavg"])
    print("  %d (T=%d): " % (y, len(te)) +
          "  ".join("%s=%.3f" % (k, rej[k].mean()) for k in KEYS))
    # ---- issued-forecast null: fixed parameters, exact Monte-Carlo calibration
    bfc = calibrate_forecast(Shat, Linv, Sinv, W)
    rfc = {k: mc_low(bfc[k], st[k]) for k in ("single", "pmerge", "bonf", "sym")}
    rfc["eavg"] = mc_high(bfc["eavg"], st["eavg"])
    rfc["chi2"] = mc_high(bfc["chi2"], st["chi2"])
    rows_fc.append((y, len(te)) + tuple(float(rfc[k].mean()) for k in KEYS))
    gaps_fc.append(rfc["eavg"].astype(float) - rfc["single"].astype(float))
    print("        forecast-null calibration: " +
          "  ".join("%s=%.3f" % (k, rfc[k].mean()) for k in KEYS))

gap = np.concatenate(gaps)


def nw_se(x, lag=None):
    """Newey-West (Bartlett) standard error of the mean under serial dependence."""
    x = np.asarray(x, float) - np.mean(x)
    T = len(x)
    if lag is None:
        lag = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))   # standard rule of thumb
    s = np.dot(x, x) / T
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * np.dot(x[l:], x[:-l]) / T
    return np.sqrt(max(s, 0.0) / T)


se_iid = gap.std(ddof=1) / np.sqrt(len(gap))
se_nw = nw_se(gap)
print("\npaired day-level gap (e-avg - single): %.4f +/- %.4f NW (iid se %.4f; T=%d days, t=%.1f)"
      % (gap.mean(), se_nw, se_iid, len(gap), gap.mean() / se_nw))
gap_fc = np.concatenate(gaps_fc); se_fc = nw_se(gap_fc)
print("  under the issued-forecast calibration: %.4f +/- %.4f NW" % (gap_fc.mean(), se_fc))

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
         rows_fc=np.array(rows_fc), gap_fc_mean=gap_fc.mean(), gap_fc_se=se_fc,
         gap_mean=gap.mean(), gap_se=se_nw, gap_se_iid=se_iid,
         logE=logE, dates=np.array(dates_all),
         rej_eavg=np.concatenate(rej_flags),
         ebh_flag=np.isin(np.arange(N), sel), alpha=ALPHA, q=Q_EBH, M=M, B=B)
print("saved fx_results.npz")
