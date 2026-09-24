"""Two bridges from expected log-evidence to fixed-level power (Remark 3 of the paper).

(i)  Repeated observations.  With T independent observations whose e-values are multiplied,
     log of the product has mean T*mu, so the number of observations needed to cross
     t = log(1/alpha) is asymptotically t/mu: two procedures need sample sizes in inverse
     ratio to their expected log-evidence.  Unconditional -- it needs only mu finite.

(ii) A single observation.  Cantelli gives
         P(log E >= t) >= (mu - t)_+^2 / (s^2 + (mu - t)^2),
     increasing in mu and decreasing in s^2.  The bound is informative only when mu > t, that
     is when the expected evidence already exceeds the level being tested at.  That condition
     fails in most of the paper's designs, which is why the fixed-level gains there are a
     fraction of the evidence gains rather than proportional to them.

Design: the Figure 3 setting (n = 10, sparse alternative, M = 12, known F), with the dependence
and the noncentrality swept.  Each replication draws its OWN ordering for the single-ordering
arm: reusing or summing several orderings of one realization is not a sum of independent copies
and overstates the pooling gain roughly twofold.
"""
import numpy as np
from scipy.special import logsumexp

SEED = 31415
n, M = 10, 12
TAUS = np.array([1.0, 2.0, 3.0])
ALPHA = 0.05
ND, CHUNK = 200_000, 20_000
LOG2 = np.log(2.0)


def logcosh(x):
    return np.logaddexp(x, -x) - LOG2


def per_order_logE(V, LINV, PERMS):
    """log of the mixture e-value, per realization and per ordering."""
    out = np.empty((len(V), len(PERMS)))
    for a in range(0, len(V), CHUNK):          # chunked: the full 3-tau stack does not fit in memory
        Z = np.einsum('mij,kmj->kmi', LINV, V[a:a + CHUNK][:, PERMS])
        tt = np.stack([-0.5 * s * s + logcosh(s * Z) for s in TAUS])
        out[a:a + CHUNK] = logsumexp(tt, axis=(0, -1)) - (np.log(n) + np.log(len(TAUS)))
    return out


def arms(rng, rho, ncp):
    """log-evidence under the alternative, for one random ordering and for the pooled average."""
    S = np.full((n, n), rho)
    np.fill_diagonal(S, 1.0)
    Sinv = np.linalg.inv(S)
    L = np.linalg.cholesky(S)
    v = np.zeros(n)
    v[0] = 1.0                                  # sparse: one coordinate carries the departure
    mu_vec = v * np.sqrt(ncp / (v @ Sinv @ v))
    PERMS = np.stack([rng.permutation(n) for _ in range(M)])
    LINV = np.stack([np.linalg.inv(np.linalg.cholesky(S[np.ix_(p, p)])) for p in PERMS])
    lE = per_order_logE(mu_vec + rng.standard_normal((ND, n)) @ L.T, LINV, PERMS)
    pick = rng.integers(0, M, ND)               # a fresh ordering in every realization
    return {"single": lE[np.arange(ND), pick], "pooled": logsumexp(lE, -1) - np.log(M)}


def T_for_power(x, t, target=0.80, Tmax=14):
    """Interpolated number of independent copies whose summed log-evidences pass t."""
    pw = [(T, (x[:len(x) // T * T].reshape(-1, T).sum(1) >= t).mean()) for T in range(1, Tmax + 1)]
    for (T0, p0), (T1, p1) in zip(pw, pw[1:]):
        if p0 < target <= p1:
            return T0 + (target - p0) / (p1 - p0)
    return float('nan')


if __name__ == "__main__":
    t = np.log(1 / ALPHA)
    print("t = log(1/alpha) = %.3f nats;  n=%d, M=%d, sparse alternative, %d draws\n"
          % (t, n, M, ND))

    print("(i) repeated observations: is the required T inversely proportional to E[log E]?  ncp=12")
    print("rho    E[logE] single -> pooled    predicted T ratio    observed T ratio")
    rng = np.random.default_rng(SEED)
    for rho in (0.2, 0.5, 0.8):
        a = arms(rng, rho, 12.0)
        ms, mp = a["single"].mean(), a["pooled"].mean()
        Ts, Tp = T_for_power(a["single"], t), T_for_power(a["pooled"], t)
        print("%.1f       %6.3f -> %6.3f           %8.3f            %8.3f"
              % (rho, ms, mp, mp / ms, Ts / Tp))

    print("\n(ii) one observation: does Cantelli bite, and does it order the arms?  rho=0.8")
    print("ncp  proc      E[logE]    Var     Cantelli LB   true power")
    rng = np.random.default_rng(SEED + 1)
    for ncp in (12.0, 20.0, 30.0):
        for nm, x in arms(rng, 0.8, ncp).items():
            mu, s2 = x.mean(), x.var()
            lb = max(mu - t, 0.0) ** 2 / (s2 + max(mu - t, 0.0) ** 2)
            print("%3.0f  %-8s %7.3f  %7.3f  %11.3f  %10.3f"
                  % (ncp, nm, mu, s2, lb, (x >= t).mean()))
        print()
    print("At ncp=12 the bound is vacuous because E[log E] < t; at ncp=20 and 30 it bites, and")
    print("there pooling is measured to lower the variance as well as raise the mean, so both")
    print("terms of the bound move the same way.")
