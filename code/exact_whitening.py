"""
Exact predictive whitening under an estimated Gaussian null: the optional `exact` mode of
exp_calibration_pipeline.py, exp_multiplicity_screening.py and exp_estimation_grid.py.

The plug-in Rosenblatt components z = L^{-1} x, with L the Cholesky factor of the training
covariance Sigma_hat = W'W/N (N training rows, mean known to be zero, no ridge), are standard
normal only as N -> infinity.  Their exact predictive counterparts are

    u   = z / sqrt(N)                       (= chol(W'W)^{-1} x),
    t_k = sqrt(N - k + 1) * u_k / sqrt(1 + u_1^2 + ... + u_{k-1}^2)   ~  t_{N-k+1},

independent over k = 1, ..., n when x is a fresh draw from the null, so that
z~_k = Phi^{-1}(F_{t, N-k+1}(t_k)) are i.i.d. N(0, 1) exactly, for every N >= n and every Sigma.
If the mean is known but nonzero, mu, subtract it first, from x and from every training row
(W'W then becomes sum_i (W_i - mu)(W_i - mu)'); without that centring the formula is wrong
whenever mu != 0.
With the mean estimated as well, A = sum_i (W_i - Wbar)(W_i - Wbar)' and u = chol(A)^{-1}(x - Wbar)
(i.e. z from Sigma_hat = A/N and the centred test point), and for N > n

    t_k = sqrt(N - k) * u_k / sqrt(1 + 1/N + u_1^2 + ... + u_{k-1}^2)   ~  t_{N-k},

again independent over k (exact_scores(..., mean_known=False)).  t_k is the studentized one-step
prediction error of coordinate k regressed on the earlier coordinates.  These are the conditional
probability integral transforms of O'Reilly and Quesenberry (1973, Ann. Statist. 1, 74-83) for the
multivariate normal family, with mean and covariance known up to scale or completely unknown, in
Rincon-Gallardo, Quesenberry and O'Reilly (1979, Ann. Statist. 7, 1052-1057); under the group
structure here they coincide with predictive probability integral transforms (O'Reilly and Villegas
1987, Canad. J. Statist. 15, 151-158).  The transformation is established methodology; what this
repository adds is its use as the whitening inside the order-randomized pooled tests.  The
experiments here use the known-mean form.
The transform is exact per ordering; the joint law across orderings still depends on Sigma, so the
pooled combiners are still calibrated by the bootstrap.

Usage in the experiment scripts: compute plug-in components with the ridge-free training
covariance (perm_chol_inv), then pass them through exact_scores(Z, N).  At known F (N = 0)
exact_scores returns Z unchanged.  Run this file to check the null law by simulation.
"""
import numpy as np
from scipy.special import ndtri, stdtr


def perm_chol_inv(S, perms):
    """Inverse Cholesky factors of S (shape (..., n, n)) under each ordering in perms (M, n):
    returns (..., M, n, n)."""
    return np.linalg.inv(np.linalg.cholesky(S[..., perms[:, :, None], perms[:, None, :]]))


def exact_scores(Z, N, mean_known=True):
    """Exact predictive components from plug-in components Z (last axis = coordinates in the
    ordering), computed with Sigma_hat = W'W/N and no ridge.  N = 0 means known F (identity)."""
    if not N:
        return Z
    n = Z.shape[-1]; k = np.arange(1, n + 1)
    df, c = (N - k + 1.0, 1.0) if mean_known else (N - k + 0.0, 1.0 + 1.0 / N)
    u = Z / np.sqrt(N)
    prev = np.cumsum(u * u, axis=-1) - u * u                # u_1^2 + ... + u_{k-1}^2
    t = np.sqrt(df) * u / np.sqrt(c + prev)
    tail = np.maximum(stdtr(df, -np.abs(t)), 1e-300)        # lower tail, no cancellation
    return -np.sign(t) * ndtri(tail)


if __name__ == "__main__":       # null-law check, known mean and estimated mean (true mean != 0 in both)
    from scipy import stats
    rng = np.random.default_rng(0)
    n, rho, reps, batch = 20, 0.5, 40000, 4000
    Sig = np.full((n, n), rho); np.fill_diagonal(Sig, 1.0); L = np.linalg.cholesky(Sig)
    mu = rng.normal(0.0, 3.0, n)
    perm = rng.permutation(n)[None, :]

    def simes_rejects(Z, a=0.05):                                 # two-sided Simes, one ordering
        p = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
        return (n * np.sort(p, -1) / np.arange(1, n + 1)).min(-1) <= a

    print("n=%d, equicorrelation %.1f, %d training/test pairs per line" % (n, rho, reps))
    for N in (25, 40, 80):
        for known in (True, False):
            zp, zx = [], []
            for _ in range(reps // batch):
                W = mu + rng.standard_normal((batch, N, n)) @ L.T; x = mu + rng.standard_normal((batch, n)) @ L.T
                if known:                                         # centre on the known mean
                    W = W - mu; x = x - mu
                else:                                             # centre on the training mean
                    Wb = W.mean(1); W = W - Wb[:, None]; x = x - Wb
                Li = perm_chol_inv(np.einsum('bki,bkj->bij', W, W) / N, perm)[:, 0]
                Z = np.einsum('bij,bj->bi', Li, x[:, perm[0]])
                zp.append(Z); zx.append(exact_scores(Z, N, mean_known=known))
            zp, zx = np.concatenate(zp), np.concatenate(zx)
            ks = min(stats.kstest(zx[:, j], "norm").pvalue for j in range(n))
            C = np.corrcoef(zx.T) - np.eye(n)
            print("N=%3d %-14s Simes size at 0.05: plug-in %.4f, exact %.4f | exact: sd %.3f-%.3f, "
                  "max |corr| %.1f SE over %d pairs, min KS p over %d coords %.3f"
                  % (N, "known mean" if known else "estimated mean", simes_rejects(zp).mean(),
                     simes_rejects(zx).mean(), zx.std(0).min(), zx.std(0).max(),
                     np.abs(C).max() * np.sqrt(reps), n * (n - 1) // 2, n, ks))
