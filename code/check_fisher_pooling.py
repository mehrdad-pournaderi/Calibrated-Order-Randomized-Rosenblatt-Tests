import numpy as np
from scipy import stats
from scipy.special import logsumexp
rng = np.random.default_rng(77)
ALPHA = 0.05; LOG2 = np.log(2.0); TAUS = np.array([1., 2., 3.])
n, M, NCP, RHO, K = 20, 12, 12.0, 0.5, 2
R, B, NTE = 100, 199, 160
S = np.full((n, n), RHO); np.fill_diagonal(S, 1.0)
Sinv = np.linalg.inv(S)
v = np.zeros(n); v[:K] = 1.0; MU = v * np.sqrt(NCP / (v @ Sinv @ v))
idx = np.arange(1, n + 1); ridge = 1e-3 * np.eye(n)


def simes2(Z):
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    return (n * np.sort(p2, axis=-1) / idx).min(-1)


def om(Z):
    Ps = simes2(Z); Mn = Z.shape[-2]
    p2 = np.minimum(2 * stats.norm.sf(np.abs(Z)), 1.0)
    pf = stats.chi2.sf(-2 * np.log(np.maximum(p2, 1e-300)).sum(-1), 2 * n)
    return dict(pmerge=np.minimum(2 * Ps.mean(-1), 1), fisher=pf[..., 0],
                pmerge_f=np.minimum(2 * pf.mean(-1), 1),
                bonf_f=np.minimum(Mn * pf.min(-1), 1))


def draw(mu, Sm, k): return mu + rng.standard_normal((k, n)) @ np.linalg.cholesky(Sm).T


KEYS = ["pmerge", "fisher", "pmerge_f", "bonf_f"]
for Ntr in [80, 160, 0]:
    cs = {k: 0. for k in KEYS}; cp = {k: 0. for k in KEYS}
    for _ in range(R):
        perms = np.stack([rng.permutation(n) for _ in range(M)])
        Shat = S if Ntr == 0 else (lambda X: X.T @ X / Ntr + ridge)(draw(np.zeros(n), S, Ntr))
        Li = np.linalg.inv(np.linalg.cholesky(Shat[perms[:, :, None], perms[:, None, :]]))
        if Ntr == 0:
            Vc = draw(np.zeros(n), Shat, B)
            boot = om(np.einsum('mij,bmj->bmi', Li, Vc[:, perms]))
        else:
            Xtb = draw(np.zeros(n), Shat, B * Ntr).reshape(B, Ntr, n)
            Sst = np.einsum('bki,bkj->bij', Xtb, Xtb) / Ntr + ridge
            Lst = np.linalg.inv(np.linalg.cholesky(Sst[:, perms[:, :, None], perms[:, None, :]]))
            Vc = draw(np.zeros(n), Shat, B)
            boot = om(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]))
        nul = om(np.einsum('mij,kmj->kmi', Li, draw(np.zeros(n), S, NTE)[:, perms]))
        alt = om(np.einsum('mij,kmj->kmi', Li, draw(MU, S, NTE)[:, perms]))
        for k in KEYS:
            c = np.quantile(boot[k], ALPHA)
            cs[k] += (nul[k] <= c).mean(); cp[k] += (alt[k] <= c).mean()
    print("Ntr=%s: " % ("known" if Ntr == 0 else Ntr) +
          "  ".join("%s[size=%.3f pow=%.3f]" % (k, cs[k] / R, cp[k] / R) for k in KEYS))
print("done")
