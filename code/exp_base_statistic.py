"""
How far from uniform does the Simes-on-Rosenblatt test land under a
misspecified null F? Monte-Carlo + analytic check.

Setup: assumed null F = N(0, I_n)  ->  Rosenblatt U_i = 1 - Phi(X_i)
(coordinatewise, since assumed coords are independent). Truth G differs.
Simes per-vector p-value: P = min_i n*U_(i)/i ; reject H0 if P < alpha.
"""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20260527)
ALPHA = 0.05


def simes(U):
    """Simes p-value P = min_i n U_(i)/i for each row of U (reps x n)."""
    n = U.shape[1]
    Us = np.sort(U, axis=1)
    ratios = n * Us / np.arange(1, n + 1)
    return ratios.min(axis=1)


def reject_rate(X, two_sided=False):
    """Rosenblatt under assumed N(0,I): U = sf(X); return P(reject).
    Two-sided version matches the paper's base statistic (Sec. 3.1):
    Simes on per-coordinate two-sided p-values 2*Phi(-|z|) = 2*min(U, 1-U)."""
    U = stats.norm.sf(X)                 # 1 - Phi(X)
    if not two_sided:
        return np.mean(simes(U) < ALPHA)
    p2 = 2.0 * np.minimum(U, 1.0 - U)    # = 2*Phi(-|X|), exact p-values under H0
    return np.mean(simes(p2) < ALPHA)


# ----------------------------------------------------------------------
# 0. Null calibration: P should be Uniform[0,1], size == alpha
# ----------------------------------------------------------------------
print("=== Null calibration (truth == assumed N(0,I)) ===")
for n in (10, 20, 50):
    X = rng.standard_normal((200_000, n))
    U = stats.norm.sf(X)
    P = simes(U)
    size = np.mean(P < ALPHA)
    ks = stats.kstest(P, "uniform").pvalue
    # two-sided size
    size2 = reject_rate(X, two_sided=True)
    print(f" n={n:3d}  one-sided size={size:.4f}  KS-unif p={ks:.3f}  two-sided size={size2:.4f}")

# ----------------------------------------------------------------------
# A. Power vs single-coordinate mean shift delta; compare analytic bound
#    Power >= P(U_1 <= alpha/n) = Phi(delta - z),  z = Phi^{-1}(1-alpha/n)
# ----------------------------------------------------------------------
print("\n=== A. Single-coordinate mean shift (one-sided Simes) ===")
deltas_A = np.linspace(0, 5, 26)
REPS = 40_000
powA = {}
boundA = {}
for n in (10, 20, 50):
    z = stats.norm.isf(ALPHA / n)
    pw, bd = [], []
    for d in deltas_A:
        X = rng.standard_normal((REPS, n))
        X[:, 0] += d
        pw.append(reject_rate(X))
        bd.append(stats.norm.cdf(d - z))
    powA[n] = np.array(pw)
    boundA[n] = np.array(bd)
    # report worst-case gap between sim power and analytic lower bound
    gap = (powA[n] - boundA[n])
    print(f" n={n:3d}  min(sim-bound)={gap.min():+.4f}  max gap={gap.max():+.4f}"
          f"  power@delta=3: {powA[n][np.argmin(abs(deltas_A-3))]:.3f}")

# ----------------------------------------------------------------------
# B. Directionality: one-sided vs two-sided, delta from -4 to +4, n=20
# ----------------------------------------------------------------------
print("\n=== B. Directionality (n=20) ===")
n = 20
deltas_B = np.linspace(-4, 4, 33)
pow_one, pow_two = [], []
for d in deltas_B:
    X = rng.standard_normal((REPS, n)); X[:, 0] += d
    pow_one.append(reject_rate(X, two_sided=False))
    X2 = rng.standard_normal((REPS, n)); X2[:, 0] += d
    pow_two.append(reject_rate(X2, two_sided=True))
pow_one, pow_two = np.array(pow_one), np.array(pow_two)
for d in (-3, -1, 1, 3):
    j = np.argmin(abs(deltas_B - d))
    print(f" delta={d:+d}: one-sided={pow_one[j]:.3f}  two-sided={pow_two[j]:.3f}")

# ----------------------------------------------------------------------
# C. Sparse vs dense at EQUAL KL budget kappa, n=50 (one-sided)
#    sparse: shift 1 coord by sqrt(2 kappa); dense: shift all 50 by sqrt(2 kappa/50)
#    (mean-shift KL per coord = shift^2/2, additive over independent coords)
# ----------------------------------------------------------------------
print("\n=== C. Sparse vs dense at equal KL (n=50) ===")
n = 50
kappas = np.linspace(0, 8, 25)
pow_sparse, pow_dense = [], []
for kap in kappas:
    # sparse
    Xs = rng.standard_normal((REPS, n))
    Xs[:, 0] += np.sqrt(2 * kap)
    pow_sparse.append(reject_rate(Xs))
    # dense
    Xd = rng.standard_normal((REPS, n))
    Xd += np.sqrt(2 * kap / n)
    pow_dense.append(reject_rate(Xd))
pow_sparse, pow_dense = np.array(pow_sparse), np.array(pow_dense)
for kap in (2, 4, 6):
    j = np.argmin(abs(kappas - kap))
    print(f" KL={kap}: sparse={pow_sparse[j]:.3f}  dense={pow_dense[j]:.3f}")

# ----------------------------------------------------------------------
# D. Dispersion misspecification: true sd = sigma, assumed sd = 1, n=20
#    one coord misspecified.  KL = sigma^2/2 - 1/2 - ln(sigma)
# ----------------------------------------------------------------------
print("\n=== D. Dispersion misspecification (n=20, 1 coord) ===")
n = 20
sigmas = np.linspace(0.3, 3.0, 28)
pow_d_one, pow_d_two = [], []
for s in sigmas:
    X = rng.standard_normal((REPS, n)); X[:, 0] *= s
    pow_d_one.append(reject_rate(X, two_sided=False))
    X2 = rng.standard_normal((REPS, n)); X2[:, 0] *= s
    pow_d_two.append(reject_rate(X2, two_sided=True))
pow_d_one, pow_d_two = np.array(pow_d_one), np.array(pow_d_two)
for s in (0.5, 1.5, 2.5):
    j = np.argmin(abs(sigmas - s))
    print(f" sigma={s}: one-sided={pow_d_one[j]:.3f}  two-sided={pow_d_two[j]:.3f}")

# ----------------------------------------------------------------------
# E. Mixed-sign dense shift: +c/sqrt(n) on half the coords, -c/sqrt(n) on
#    the other half (total energy c^2). Either one-sided orientation sees
#    at most half the signal; two-sided sees all of it.
# ----------------------------------------------------------------------
print("\n=== E. Mixed-sign shift +/- c/sqrt(n) on half the coords (n=20) ===")
n = 20
cs = np.linspace(0, 10, 26)
mu_mix = np.zeros(n); mu_mix[: n // 2] = 1.0; mu_mix[n // 2:] = -1.0
pow_mix_one, pow_mix_two = [], []
for c in cs:
    X = rng.standard_normal((REPS, n)) + (c / np.sqrt(n)) * mu_mix
    pow_mix_one.append(reject_rate(X, two_sided=False))
    X2 = rng.standard_normal((REPS, n)) + (c / np.sqrt(n)) * mu_mix
    pow_mix_two.append(reject_rate(X2, two_sided=True))
pow_mix_one, pow_mix_two = np.array(pow_mix_one), np.array(pow_mix_two)
for c in (4, 6, 8):
    j = np.argmin(abs(cs - c))
    print(f" c={c}: one-sided={pow_mix_one[j]:.3f}  two-sided={pow_mix_two[j]:.3f}")

# ----------------------------------------------------------------------
# Save arrays for plotting
# ----------------------------------------------------------------------
np.savez("results_base_statistic.npz",
         deltas_A=deltas_A, powA10=powA[10], powA20=powA[20], powA50=powA[50],
         bd10=boundA[10], bd20=boundA[20], bd50=boundA[50],
         deltas_B=deltas_B, pow_one=pow_one, pow_two=pow_two,
         kappas=kappas, pow_sparse=pow_sparse, pow_dense=pow_dense,
         sigmas=sigmas, pow_d_one=pow_d_one, pow_d_two=pow_d_two,
         cs=cs, pow_mix_one=pow_mix_one, pow_mix_two=pow_mix_two,
         alpha=ALPHA)
print("\nsaved sim_results.npz")
