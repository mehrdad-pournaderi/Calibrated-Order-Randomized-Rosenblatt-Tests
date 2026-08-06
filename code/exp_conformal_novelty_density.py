"""
Where does the crossover with AdaDetect sit?  A sweep in the density of novelties.

In exp_conformal_comparison.py the pooled parametric test leads throughout at pi_0 = 0.95.
That is not the whole story, and this script isolates why.  AdaDetect learns its nonconformity
score by classifying the reference sample against the test sample, so the only signal it can
learn from is the novelties that happen to be present there.  Its power should therefore fall
away as the novelties become rare, while the pooled parametric test -- which learns nothing
from the test sample -- should degrade only as fast as the Benjamini-Hochberg threshold
tightens.  The nearest-neighbour score is the control: it is conformal but does not learn, so
if the mechanism is right it should track the parametric test rather than AdaDetect.

The reference sample is held FIXED at N_tr = 1000 and pi_0 is swept over 0.80, 0.90 and 0.95,
giving 40, 20 and 10 novelties among the 200 hypotheses.  Everything else is the design of
Section 7: n=20, half of the non-nulls failing on one coordinate and half on all of them,
common energy ncp=20, rho=0.5, q=0.10, M=12, B=2500.

Every nuisance choice is fixed in advance and never selected on the results: k = 1 for the
nearest-neighbour score, calibration size l = N_tr/2 for the one-class scores, and l = m for
AdaDetect, the value recommended in its authors' Remark 2.2 (a balanced split would halve the
sample available to train its classifier).

Usage: python exp_conformal_novelty_density.py pi0 [R] [seed]      (pi0 as a percentage, e.g. 95)
       then: python merge_conformal_chunks.py pi0 80 90 95
"""
import sys
import numpy as np
from scipy import stats
from scipy.special import logsumexp
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

N, M, NCP, RHO = 20, 12, 20.0, 0.5
TAUS = np.array([1.0, 2.0, 3.0])
NUNITS, Q = 200, 0.10
R_DEFAULT, B, BCHUNK = 25, 2500, 500
NTR = 1000
LGRID = tuple(sorted({NUNITS, NTR // 2}))
KNN_GRID = (1, 5, 20, 50)
idx = np.arange(1, N + 1); ridge = 1e-3 * np.eye(N); LOG2 = np.log(2.0)

S = np.full((N, N), RHO); np.fill_diagonal(S, 1.0)
Sinv = np.linalg.inv(S); Lt = np.linalg.cholesky(S)

OURS = ["pmerge", "bonf", "eavg", "chi2"]
CONF = ["mahal", "kde", "ada-rf", "ada-lr"] + ["knn:%d" % k for k in KNN_GRID]


def logcosh(x): return np.logaddexp(x, -x) - LOG2


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (N * np.sort(p2, axis=-1) / idx).min(-1)


def order_stats(Z):
    Ps = simes2(Z); Mn = Z.shape[-2]
    terms = np.stack([-0.5 * t * t + logcosh(t * Z) for t in TAUS])
    lE = logsumexp(terms, axis=(0, -1)) - (np.log(N) + np.log(len(TAUS)))
    return dict(pmerge=np.minimum(2 * Ps.mean(-1), 1.0),
                bonf=np.minimum(Mn * Ps.min(-1), 1.0),
                eavg=np.exp(logsumexp(lE, -1) - np.log(Mn)))


def bh(p, q):
    m = len(p); o = np.argsort(p)
    ok = np.where(p[o] <= q * np.arange(1, m + 1) / m)[0]
    r = np.zeros(m, bool)
    if len(ok): r[o[:ok.max() + 1]] = True
    return r


def wishart_cov(Lc, dof, size):
    A = np.zeros((size, N, N)); tl = np.tril_indices(N, -1)
    A[:, tl[0], tl[1]] = rng.standard_normal((size, len(tl[0])))
    A[:, np.arange(N), np.arange(N)] = np.sqrt(rng.chisquare(dof - np.arange(N), size=(size, N)))
    LA = Lc @ A
    return LA @ np.swapaxes(LA, -1, -2) / dof


def conformal_p(s_cal, s_test):
    sc = np.sort(s_cal); l = len(sc)
    return (1.0 + l - np.searchsorted(sc, s_test, side="left")) / (l + 1.0)


def knn_scores(fit, Y, kgrid=KNN_GRID):
    d = np.sqrt(((Y[:, None, :] - fit[None, :, :]) ** 2).sum(-1))
    part = np.partition(d, [k - 1 for k in kgrid], axis=1)
    return {k: part[:, k - 1] for k in kgrid}


def kde_score(fit, Y):
    kde = stats.gaussian_kde(fit.T)
    return -np.log(np.maximum(kde(Y.T), 1e-300))


def quad(Z):
    iu = np.triu_indices(N)
    return np.concatenate([Z, (Z[:, :, None] * Z[:, None, :])[:, iu[0], iu[1]]], 1)


def ada(fit, mixed, kind):
    Xd = np.vstack([fit, mixed]); yd = np.r_[np.zeros(len(fit)), np.ones(len(mixed))]
    if kind == "rf":
        c = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
        c.fit(Xd, yd); return c.predict_proba(mixed)[:, 1]
    Xf = quad(Xd); sd = Xf.std(0); sd[sd == 0] = 1.0; Xf = (Xf - Xf.mean(0)) / sd
    c = LogisticRegression(C=0.05, max_iter=3000); c.fit(Xf, yd)
    return c.predict_proba(Xf[len(fit):])[:, 1]


def run(pi0, R):
    n_null = int(round(NUNITS * pi0)); n_alt = NUNITS - n_null
    truth = np.zeros(NUNITS, bool); truth[n_null:] = True
    v_sp = np.zeros(N); v_sp[0] = 1.0
    MU = np.zeros((NUNITS, N))
    MU[n_null:n_null + n_alt // 2] = v_sp * np.sqrt(NCP / (v_sp @ Sinv @ v_sp))
    MU[n_null + n_alt // 2:] = np.ones(N) * np.sqrt(NCP / (np.ones(N) @ Sinv @ np.ones(N)))

    acc_ours = {k: [] for k in OURS}
    acc_conf = {l: {k: [] for k in CONF} for l in LGRID}

    for _ in range(R):
        perms = np.stack([rng.permutation(N) for _ in range(M)])
        pidx = (perms[:, :, None], perms[:, None, :])
        ref = rng.standard_normal((NTR, N)) @ Lt.T
        X = MU + rng.standard_normal((NUNITS, N)) @ Lt.T

        Shat = ref.T @ ref / NTR + ridge; Lc = np.linalg.cholesky(Shat)
        Li = np.linalg.inv(np.linalg.cholesky(Shat[pidx])); Sih = np.linalg.inv(Shat)
        pool = {k: [] for k in OURS}
        for s in range(0, B, BCHUNK):
            b = min(BCHUNK, B - s)
            Sst = wishart_cov(Lc, NTR, b) + ridge
            Lst = np.linalg.inv(np.linalg.cholesky(Sst[:, perms[:, :, None], perms[:, None, :]]))
            Vc = rng.standard_normal((b, N)) @ Lc.T
            bs = order_stats(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]))
            bs["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
            for k in OURS: pool[k].append(bs[k])
        boot = {k: np.sort(np.concatenate(v)) for k, v in pool.items()}
        st = order_stats(np.einsum('mij,kmj->kmi', Li, X[:, perms]))
        st["chi2"] = np.einsum('ki,ij,kj->k', X, Sih, X)
        for k in OURS:
            cnt = (B - np.searchsorted(boot[k], st[k], side="left")) if k in ("eavg", "chi2") \
                else np.searchsorted(boot[k], st[k], side="right")
            r = bh((1 + cnt) / (B + 1), Q)
            acc_ours[k].append((np.sum(r & ~truth) / max(r.sum(), 1), np.sum(r & truth) / n_alt))

        for l in LGRID:
            fit, cal = ref[:NTR - l], ref[NTR - l:]
            Sfi = np.linalg.inv(fit.T @ fit / len(fit) + ridge)
            mah = lambda Y: np.einsum('ki,ij,kj->k', Y, Sfi, Y)
            rj = {"mahal": bh(conformal_p(mah(cal), mah(X)), Q),
                  "kde": bh(conformal_p(kde_score(fit, cal), kde_score(fit, X)), Q)}
            kc, kx = knn_scores(fit, cal), knn_scores(fit, X)
            for k in KNN_GRID:
                rj["knn:%d" % k] = bh(conformal_p(kc[k], kx[k]), Q)
            mixed = np.vstack([cal, X])
            for kind, nm in (("rf", "ada-rf"), ("lr", "ada-lr")):
                sc = ada(fit, mixed, kind)
                rj[nm] = bh(conformal_p(sc[:l], sc[l:]), Q)
            for k in CONF:
                r = rj[k]
                acc_conf[l][k].append((np.sum(r & ~truth) / max(r.sum(), 1),
                                       np.sum(r & truth) / n_alt))

    raw = {k: np.asarray(acc_ours[k]) for k in OURS}
    raw.update({"%s@%d" % (k, l): np.asarray(acc_conf[l][k]) for l in LGRID for k in CONF})
    return raw, n_alt


pi0_pct = int(sys.argv[1]) if len(sys.argv) > 1 else 95
R = int(sys.argv[2]) if len(sys.argv) > 2 else R_DEFAULT
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(4711 + 1000 * seed)

raw, n_alt = run(pi0_pct / 100.0, R)
print("Novelty-density sweep: N_tr=%d FIXED, n=%d, %d hypotheses, ncp=%.0f, q=%.2f, M=%d, B=%d"
      % (NTR, N, NUNITS, NCP, Q, M, B))
print("pi_0=%.2f (%d novelties)  chunk seed=%d  R=%d" % (pi0_pct / 100.0, n_alt, seed, R))
for k in OURS:
    a = raw[k]; print("   %-8s FDR=%.3f  power=%.3f" % (k, a[:, 0].mean(), a[:, 1].mean()))
for k in CONF:
    best = max(LGRID, key=lambda l: raw["%s@%d" % (k, l)][:, 1].mean())
    a = raw["%s@%d" % (k, best)]
    print("   %-8s FDR=%.3f  power=%.3f   (l=%d)" % (k, a[:, 0].mean(), a[:, 1].mean(), best))

fn = "chunk_conformal_pi0%d_s%d.npz" % (pi0_pct, seed)
np.savez(fn, ntr=NTR, pi0=pi0_pct / 100.0, n_alt=n_alt, lgrid=np.array(LGRID), R=R, seed=seed,
         q=Q, n=N, nunits=NUNITS, ncp=NCP, M=M, B=B, **raw)
print("saved " + fn)
