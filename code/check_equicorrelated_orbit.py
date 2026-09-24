"""The exactly solvable orbit: equicorrelated Sigma, sparse departure (Section 4 of the paper).

With Sigma = (1-rho)I + rho 11' every permutation satisfies P Sigma P' = Sigma, so the Cholesky
factor is the same for all orderings and only the slot j into which the departing coordinate
falls matters.  The whitened mean profile is then closed-form (Lemma: slot profile), the power
of the within-ordering Bonferroni test is a finite product of normal probabilities (exact, no
simulation), and the orbit is totally ordered: later slots majorize earlier ones.

This script verifies the closed form against direct Cholesky computation and prints every
number quoted in the subsection: the rho-free oracle, the oracle-minus-average gain and the
max-min spread by rho, the location of the spread's peak by n, and the degeneracy of the dense
alternative.
"""
import numpy as np
from scipy import stats

ALPHA = 0.05


def profile(n, rho, delta, j):
    """Whitened mean profile when the departing coordinate falls in slot j (1-indexed)."""
    a = np.array([np.nan] + [rho / (1 + (k - 2) * rho) for k in range(2, n + 1)])
    v = np.array([1.0] + [1 - (k - 1) * rho**2 / (1 + (k - 2) * rho) for k in range(2, n + 1)])
    m = np.zeros(n)
    m[j - 1] = delta / np.sqrt(v[j - 1])
    for k in range(j + 1, n + 1):
        m[k - 1] = -delta * a[k - 1] / np.sqrt(v[k - 1])
    return m


def profile_direct(n, rho, delta, j, rng):
    """Ground truth: L^{-1} P mu for a random permutation that puts the spike in slot j."""
    S = np.full((n, n), rho)
    np.fill_diagonal(S, 1.0)
    mu = np.zeros(n)
    mu[0] = delta
    rest = list(rng.permutation(range(1, n)))    # predecessors in random order: must not matter
    perm = np.array(rest[:j - 1] + [0] + rest[j - 1:])
    return np.linalg.solve(np.linalg.cholesky(S[np.ix_(perm, perm)]), mu[perm])


def beta(n, rho, delta, j, alpha=ALPHA):
    """Exact power at slot j of the within-ordering Bonferroni (max-|z|) test.

    c = Phi^{-1}(1 - alpha/2n); the exact size is 1 - (1-alpha/n)^n, slightly below alpha.
    Bonferroni rejection implies Simes rejection, so beta is also a lower bound on the
    per-ordering Simes power.
    """
    c = stats.norm.isf(alpha / (2 * n))
    m = profile(n, rho, delta, j)
    return 1 - np.prod(stats.norm.cdf(c - m) - stats.norm.cdf(-c - m))


def delta_for_ncp(n, rho, ncp):
    S = np.full((n, n), rho)
    np.fill_diagonal(S, 1.0)
    return np.sqrt(ncp / np.linalg.inv(S)[0, 0])


def gain_and_spread(n, rho, ncp):
    d = delta_for_ncp(n, rho, ncp) if rho else np.sqrt(ncp)
    b = np.array([beta(n, rho, d, j) for j in range(1, n + 1)])
    return b.max() - b.mean(), b.max() - b.min(), b


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("(a) closed form vs direct Cholesky (random predecessor arrangements)")
    err = max(np.abs(profile(n, rho, 1.7, j) - profile_direct(n, rho, 1.7, j, rng)).max()
              for n in (3, 5, 10, 20, 30)
              for rho in (0.0, 0.3, 0.7, 0.9, 0.95)
              for j in range(1, n + 1))
    nerr = max(np.abs(profile(n, -0.9 / (n - 1), 1.7, j)
                      - profile_direct(n, -0.9 / (n - 1), 1.7, j, rng)).max()
               for n in (5, 10, 20) for j in range(1, n + 1))
    print("    max abs error: %.2e (positive rho), %.2e (negative rho)" % (err, nerr))

    print("\n(b) majorization along the orbit (sorted squared profiles, partial sums)")
    viol = 0
    for n in (5, 10, 20, 30):
        for rho in np.arange(0.05, 1.0, 0.05):
            for j in range(1, n):
                A = np.sort(profile(n, rho, 1.7, j) ** 2)[::-1]
                B = np.sort(profile(n, rho, 1.7, j + 1) ** 2)[::-1]
                viol += not np.all(np.cumsum(B) >= np.cumsum(A) - 1e-11)
    print("    violations: %d" % viol)

    print("\n(c) exact size of the max test: 1-(1-alpha/n)^n = %.5f at n=10" % (1 - (1 - ALPHA / 10) ** 10))

    print("\n(d) n=10, ncp=12: per-slot power, oracle, gain, spread")
    print("    rho    beta(1)  beta(n)   E[beta]   gain=max-mean   spread=max-min   monotone")
    for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
        g, s, b = gain_and_spread(10, rho, 12.0)
        print("    %.2f   %.4f   %.4f   %.4f     %.4f          %.4f         %s"
              % (rho, b[0], b[-1], b.mean(), g, s, bool(np.all(np.diff(b) >= -1e-12))))

    print("\n(e) the spread's peak in rho, by dimension (the turnover prediction)")
    grid = np.arange(0.30, 0.996, 0.005)
    for n in (5, 10, 20, 50):
        sp = np.array([gain_and_spread(n, r, 12.0)[1] for r in grid])
        gn = np.array([gain_and_spread(n, r, 12.0)[0] for r in grid])
        print("    n=%2d: spread peaks at rho*=%.3f (peak %.3f);  max gain %.3f"
              % (n, grid[sp.argmax()], sp.max(), gn.max()))

    print("\n(f) dense alternative mu = c*1: P mu = mu, the orbit is a single point")
    n = 10
    S = np.full((n, n), 0.5)
    np.fill_diagonal(S, 1.0)
    mu = np.ones(n) * np.sqrt(12.0 / np.ones(n) @ np.linalg.inv(S) @ np.ones(n))
    L = np.linalg.cholesky(S)
    base = np.linalg.solve(L, mu)
    dev = max(np.abs(np.linalg.solve(L, mu[rng.permutation(n)]) - base).max() for _ in range(200))
    print("    max deviation of the whitened profile across 200 random orderings: %.1e" % dev)
