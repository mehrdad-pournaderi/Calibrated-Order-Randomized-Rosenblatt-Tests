"""
DECISION STABILITY versus the number of orderings M (Section 7.2 and the corollary to the
finite-M proposition of Section 4).

How often would a different random set of M orderings change the decision?  In each realization
two independent banks of MMAX orderings are drawn; each bank is calibrated on its own B exact null
draws (Procedure 1 at known F, conditional on the bank), and both are applied to the same NTE test
vectors.  Stored, per M: the probability that the two banks reach different decisions, for the
calibrated e-value average, p-merge and Bonferroni-over-orders and for the e-value average at the
nominal threshold 1/alpha, under the alternative and under the null.  Monte-Carlo standard errors
are across realizations.

Designs (known F, ncp = 12): `one` is the M-sweep design (n=10, rho=0.6, one shifted coordinate);
`all` shifts all coordinates of an n=10, rho=0.5 vector (the dense design of Figure 3).
Usage: python exp_decision_stability.py         ->  results_decision_stability.npz
       python exp_decision_stability.py shared  ->  results_decision_stability_shared.npz
The plain run calibrates each bank on its own null draws, so the two decisions differ through the
orderings AND the calibration sample (the whole randomized procedure; two independent calibrations of
one continuous statistic alone disagree on about 0.55% of null observations at B = 1999).  With
`shared`, both banks are calibrated on the same null draws (each with its own orderings), so the
decisions differ only through the orderings; the paper quotes this version.
"""
import sys
import numpy as np
from scipy import stats
from scipy.special import logsumexp

ALPHA = 0.05; LOG2 = np.log(2.0); NCP = 12.0
TAUS = np.array([1.0, 2.0, 3.0])
MGRID = np.array([1, 2, 4, 8, 12, 16, 32, 64, 128]); MMAX = int(MGRID[-1])
B, NTE, R = 1999, 200, 200
DESIGNS = [("one", 10, 0.6, 1, 2201), ("all", 10, 0.5, 10, 2202)]
KEYS = ("eavg", "pmerge", "bonf", "eavg_nominal")
SHARED = "shared" in sys.argv[1:]


def per_order(X, perms, Li, n):
    Z = np.einsum('ij,rmj->rmi', Li, X[:, perms])
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    P = (n * np.sort(p2, axis=2) / np.arange(1, n + 1)).min(axis=2)
    lc = np.logaddexp(TAUS[:, None, None, None] * Z, -TAUS[:, None, None, None] * Z) - LOG2
    lE = logsumexp(-0.5 * TAUS[:, None, None, None] ** 2 + lc, axis=(0, 3)) - np.log(n * len(TAUS))
    return P, lE


def stats_M(P, lE, M):
    return {"pmerge": P[:, :M].mean(1), "bonf": P[:, :M].min(1),
            "eavg": logsumexp(lE[:, :M], 1) - np.log(M)}


def reject(null_s, s, low):
    sb = np.sort(null_s); K = len(null_s)
    cnt = np.searchsorted(sb, s, side="right") if low else K - np.searchsorted(sb, s, side="left")
    return 1 + cnt <= ALPHA * (K + 1)


def run(n, rho, k, seed):
    rng = np.random.default_rng(seed)
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); L = np.linalg.cholesky(S); Li = np.linalg.inv(L)
    v = np.zeros(n); v[:k] = 1.0; mu = v * np.sqrt(NCP / (v @ np.linalg.inv(S) @ v))
    dis = {h: np.zeros((R, len(KEYS), len(MGRID))) for h in ("alt", "null")}
    pw = np.zeros((R, len(KEYS), len(MGRID)))
    for r in range(R):
        Xt = {"alt": mu + rng.standard_normal((NTE, n)) @ L.T, "null": rng.standard_normal((NTE, n)) @ L.T}
        dec = {}
        Xc = rng.standard_normal((B, n)) @ L.T if SHARED else None   # shared calibration draws
        for bank in (0, 1):
            perms = np.argsort(rng.random((MMAX, n)), axis=1)
            Pn, lEn = per_order(Xc if SHARED else rng.standard_normal((B, n)) @ L.T, perms, Li, n)
            obs = {h: per_order(Xt[h], perms, Li, n) for h in Xt}
            for j, M in enumerate(MGRID):
                sn = stats_M(Pn, lEn, int(M))
                for h, (P, lE) in obs.items():
                    s = stats_M(P, lE, int(M))
                    dec[bank, h, j] = np.stack([reject(sn["eavg"], s["eavg"], False),
                                                reject(sn["pmerge"], s["pmerge"], True),
                                                reject(sn["bonf"], s["bonf"], True),
                                                s["eavg"] >= np.log(1 / ALPHA)])
        for h in ("alt", "null"):
            for j in range(len(MGRID)):
                dis[h][r, :, j] = (dec[0, h, j] != dec[1, h, j]).mean(1)
                if h == "alt":
                    pw[r, :, j] = 0.5 * (dec[0, h, j].mean(1) + dec[1, h, j].mean(1))
    return dis, pw


if __name__ == "__main__":
    out = {"MGRID": MGRID, "keys": np.array(KEYS), "R": R, "B": B, "NTE": NTE, "alpha": ALPHA}
    for name, n, rho, k, seed in DESIGNS:
        dis, pw = run(n, rho, k, seed)
        for h in ("alt", "null"):
            out[f"dis_{h}_{name}"] = dis[h].mean(0); out[f"dis_{h}_se_{name}"] = dis[h].std(0, ddof=1) / np.sqrt(R)
        out[f"power_{name}"] = pw.mean(0)
        print("%s (n=%d, rho=%.1f, k=%d): probability that two independent banks disagree" % (name, n, rho, k))
        print("   M               " + " ".join("%6d" % m for m in MGRID))
        for i, key in enumerate(KEYS):
            print("   %-13s alt " % key + " ".join("%6.3f" % v for v in out[f"dis_alt_{name}"][i]))
            print("   %-13s null" % "" + " ".join("%6.3f" % v for v in out[f"dis_null_{name}"][i]))
    fn = "results_decision_stability%s.npz" % ("_shared" if SHARED else "")
    np.savez(fn, **out)
    print("saved " + fn)
