"""
AdaDetect score functions for exp_conformal_comparison.py and exp_conformal_novelty_density.py.

AdaDetect (Marandon, Lei, Mary & Roquain 2024) trains a probabilistic classifier to separate the
fit part of the reference sample (label 0) from the calibration part pooled with the test sample
(label 1), and scores every calibration and test point by its predicted probability of label 1.
The conformal p-values stay valid because that score function is invariant to permutations
within calibration + test.

Here the invariance holds literally, not only in distribution: the classifier is trained on the
calibration + test rows in a canonical (lexicographic) order, so the fitted score depends on
those rows only through their unordered set.  This matters for the random forest: with a fixed
seed it draws its bootstrap samples by row position, so permuting the rows would otherwise change
the fitted forest (check_adadetect_invariance.py shows the size of that effect).  Prediction is
row by row and single-threaded, so it adds no order dependence either.  The canonical order was
introduced in November 2026; the logistic-regression score was already invariant up to
floating-point rounding, and its decisions are unchanged.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def canonical_order(A):
    """Row order of A sorted lexicographically over all columns (first column first), so that
    A[canonical_order(A)] depends on the rows of A only through their multiset: rows that are
    equal in every column are interchangeable, and any other tie is broken by a later column."""
    return np.lexsort(A.T[::-1])


def quad_features(Z):
    """(z, {z_i z_j}_{i <= j}): the quadratic feature map of the logistic score."""
    iu = np.triu_indices(Z.shape[1])
    return np.concatenate([Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]], axis=1)


def adadetect_scores(fit, mixed, kind, canonical=True):
    """Predicted P(label 1) for every row of `mixed` (= calibration + test), from a classifier
    separating `fit` (label 0) from `mixed` (label 1).  kind = "rf": random forest, 100 trees,
    minimum leaf size 5; kind = "lr": l2-penalised logistic regression (C = 0.05) on standardised
    quadratic features.  canonical=False reproduces the pre-November-2026 implementation, which
    trained on `mixed` in the order given; it is kept only for check_adadetect_invariance.py."""
    o = canonical_order(mixed) if canonical else np.arange(len(mixed))
    Xd = np.vstack([fit, mixed[o]])
    yd = np.r_[np.zeros(len(fit)), np.ones(len(mixed))]
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=100, min_samples_leaf=5,
                                     random_state=0, n_jobs=-1)
        clf.fit(Xd, yd)
        clf.set_params(n_jobs=1)          # deterministic summation over trees
        return clf.predict_proba(mixed)[:, 1]
    Xf = quad_features(Xd)
    sd = Xf.std(0); sd[sd == 0] = 1.0
    Xf = (Xf - Xf.mean(0)) / sd
    clf = LogisticRegression(C=0.05, max_iter=3000)
    clf.fit(Xf, yd)
    sc = np.empty(len(mixed))
    sc[o] = clf.predict_proba(Xf[len(fit):])[:, 1]      # back to the original row order
    return sc
