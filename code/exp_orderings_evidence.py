"""
HOW MANY ORDERINGS?  Expected log-evidence of the e-value average as a function of M
(the finite-M proposition of Section 4 and the M paragraph of Section 7.2).

For each design, X is drawn from the alternative (known Sigma) and the two-sided mixture e-value
is computed under MMAX = 2048 random orderings.  For each M in MGRID, E[log Ebar_M | X] is estimated
by averaging log(mean of each disjoint block of M e-values), which is unbiased for it.  Stored:

  L[M]      mean over X of that estimate (expected log-evidence of the e-value average with M orderings)
  share[M]  (L[M] - L[1]) / (L[MMAX] - L[1]): fraction of the order-averaging gain captured at M
  V         mean over X of the squared coefficient of variation of the per-ordering e-values,
            V(X) = Var_sigma(E^sigma | X) / Ebar(X)^2, so V/(2M) is the second-order prediction of
            the log-evidence still missing at M

Equicorrelated Sigma is invariant under permutation, so every ordering has the same Cholesky
factor and only X is permuted.  Usage: python exp_orderings_evidence.py  ->  results_orderings_evidence.npz
"""
import numpy as np
from scipy.special import logsumexp

TAUS = np.array([1.0, 2.0, 3.0]); NCP = 12.0
MMAX = 2048
MGRID = np.array([1, 2, 4, 8, 12, 16, 32, 64, 128, 256, 512, 1024, 2048])
R, CHUNK = 4000, 100
DESIGNS = [  # name, n, rho, k (number of same-sign shifted coordinates), seed
    ("one", 10, 0.5, 1, 4101),       # sparse design of Figure 3
    ("all", 10, 0.5, 10, 4102),      # dense design of Figure 3
    ("two", 20, 0.5, 2, 4103),       # design of Figure 2
    ("sweep", 10, 0.6, 1, 4104),     # design of the calibrated M sweep (exp_number_of_orderings.py)
]


def log_evalues(Z):
    """Z (..., n): two-sided mixture e-value of each row, on the log scale."""
    n = Z.shape[-1]
    lc = np.logaddexp(TAUS[:, None, None, None] * Z, -TAUS[:, None, None, None] * Z) - np.log(2.0)
    terms = -0.5 * TAUS[:, None, None, None] ** 2 + lc
    return logsumexp(terms, axis=(0, -1)) - np.log(n * len(TAUS))


def run(n, rho, k, seed):
    rng = np.random.default_rng(seed)
    S = np.full((n, n), rho); np.fill_diagonal(S, 1.0)
    L = np.linalg.cholesky(S); Li = np.linalg.inv(L); Sinv = np.linalg.inv(S)
    v = np.zeros(n); v[:k] = 1.0; mu = v * np.sqrt(NCP / (v @ Sinv @ v))
    est = np.zeros((R, len(MGRID))); V = np.zeros(R)
    for c0 in range(0, R, CHUNK):
        X = mu + rng.standard_normal((CHUNK, n)) @ L.T
        P = np.argsort(rng.random((CHUNK, MMAX, n)), axis=-1)          # fresh orderings per X
        Z = np.einsum('ij,kmj->kmi', Li, np.take_along_axis(X[:, None, :], P, axis=-1))
        lE = log_evalues(Z)                                            # (CHUNK, MMAX)
        for j, M in enumerate(MGRID):
            nb = MMAX // M
            blocks = lE[:, :nb * M].reshape(CHUNK, nb, M)
            est[c0:c0 + CHUNK, j] = (logsumexp(blocks, axis=-1) - np.log(M)).mean(-1)
        E = np.exp(lE - lE.max(-1, keepdims=True))
        V[c0:c0 + CHUNK] = E.var(-1) / E.mean(-1) ** 2
    Lm = est.mean(0); se = est.std(0, ddof=1) / np.sqrt(R)
    share = (Lm - Lm[0]) / (Lm[-1] - Lm[0])
    return Lm, se, share, V.mean(), np.median(V)


if __name__ == "__main__":
    out = {"MGRID": MGRID, "MMAX": MMAX, "R": R, "ncp": NCP, "names": np.array([d[0] for d in DESIGNS])}
    for name, n, rho, k, seed in DESIGNS:
        Lm, se, share, Vmean, Vmed = run(n, rho, k, seed)
        out.update({f"L_{name}": Lm, f"Lse_{name}": se, f"share_{name}": share,
                    f"Vmean_{name}": Vmean, f"Vmed_{name}": Vmed, f"n_{name}": n, f"rho_{name}": rho, f"k_{name}": k})
        gain = Lm[-1] - Lm[0]
        print("%-5s n=%2d rho=%.1f k=%2d  E log E (M=1) %.3f  gain to M=%d %.3f nats  mean V %.3f  median V %.3f"
              % (name, n, rho, k, Lm[0], MMAX, gain, Vmean, Vmed))
        for j, M in enumerate(MGRID):
            print("      M=%4d  share %.3f   deficit %.4f   V/(2M) %.4f" % (M, share[j], Lm[-1] - Lm[j], Vmean / (2 * M)))
    np.savez("results_orderings_evidence.npz", **out)
    print("saved results_orderings_evidence.npz")
