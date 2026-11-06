"""
Is the AdaDetect score invariant to the order of the calibration + test rows?

AdaDetect's validity rests on the score function being invariant to permutations within
calibration + test.  This check fits the score, permutes the calibration + test rows, fits again,
restores the original order, and reports the largest change in any score -- for the implementation
used in the paper (canonical row order before fitting) and for the earlier one, which trained on
the rows in the order given.  The data follow the design of Section 7.3 at N_tr = 1000: n = 20,
equicorrelation 0.5, a fit part of 800 reference points and 200 calibration + 200 test points.

Expected output: changes of order 0.1 for the earlier random forest, of order 1e-12 (floating-point
rounding) for the earlier logistic score, and exactly 0 for both with the canonical order.
"""
import numpy as np
from adadetect_score import adadetect_scores

n, rho, reps = 20, 0.5, 5
rng = np.random.default_rng(2026)
S = np.full((n, n), rho); np.fill_diagonal(S, 1.0); L = np.linalg.cholesky(S)
fit = rng.standard_normal((800, n)) @ L.T
mixed = rng.standard_normal((400, n)) @ L.T

fails = 0
for kind, name in (("rf", "random forest"), ("lr", "penalized logistic")):
    for canonical in (False, True):
        base = adadetect_scores(fit, mixed, kind, canonical)
        worst = 0.0
        for _ in range(reps):
            p = rng.permutation(len(mixed))
            s = np.empty(len(mixed)); s[p] = adadetect_scores(fit, mixed[p], kind, canonical)
            worst = max(worst, np.abs(s - base).max())
        tag = "canonical order (paper)" if canonical else "rows as given (earlier)"
        print("%-18s %-24s max |score change| over %d permutations: %.3g" % (name, tag, reps, worst))
        if canonical and worst != 0.0:
            fails += 1
print("RESULT: %s" % ("PASS -- exactly invariant" if fails == 0 else "FAIL"))
