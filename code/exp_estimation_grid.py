"""
POWER UNDER ESTIMATION, BY DEPARTURE SHAPE (Appendix B, Table B.1).

How large must the training sample be before the e-value average over orderings matches the
stronger order-invariant baseline?  For three single-hypothesis departures at equal energy
(ncp = 12, rho = 0.5, M = 12, alpha = 0.05) this sweeps N_tr on a dense grid, bootstrap-calibrates
every test by Procedure 1 (re-estimating double bootstrap, rank-based Monte-Carlo decisions,
B = 199) and stores the calibrated power of each test together with the paired differences
(per-realization power, R = 300 realizations, fresh orderings in every realization).

    two  n = 20, two same-sign shifted coordinates   (the design of Figure 2)
    one  n = 10, one shifted coordinate              (the sparse design of Figure 3)
    all  n = 10, all coordinates shifted equally     (the dense design of Figure 3)

Usage:  python exp_estimation_grid.py {two|one|all}   ->  results_estimation_grid_<design>.npz
        python exp_estimation_grid.py {two|one|all} exact
            adds the exact predictive components of exact_whitening.py on the same draws
            (ridge-free training covariance): keys pow_<m>_x, gap_<m>_x_<baseline> and the paired
            exact-minus-plug-in differences gap_<m>_x_<m>, in results_estimation_grid_<design>_exact.npz

The whitening, base statistics, combiners and calibration are taken verbatim from
exp_departure_shape.py (its definitions are executed below, without running its experiment), so
the two scripts cannot drift apart.  Each grid point has its own seed (seed0 + its index).
"""
import os
import sys
import numpy as np

_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp_departure_shape.py")).read()
_ns = {}
exec(_src[:_src.index("mode = sys.argv")], _ns)            # definitions only
order_methods, simes2, linv1, linvB, invsqrt, draw, mc_low, mc_high, equicorr = (
    _ns[k] for k in ["order_methods", "simes2", "linv1", "linvB", "invsqrt", "draw",
                     "mc_low", "mc_high", "equicorr"])

ALPHA, B, NTE, NCP, M, RHO, R = 0.05, 199, 160, 12.0, 12, 0.5, 300
ARGS = [a for a in sys.argv[1:] if a != "exact"]
EXACT = len(ARGS) < len(sys.argv) - 1
KEYS = ["eavg", "pmerge", "bonf", "sym", "chi2"]
POOLED = ["eavg", "pmerge", "bonf"]
if EXACT:
    from exact_whitening import exact_scores
    KEYS += [k + "_x" for k in POOLED]
DESIGNS = {   # grid in absolute N_tr; 0 = known F
    "two": dict(n=20, k=2, seed0=9100, grid=[80, 100, 120, 140, 160, 200, 240, 320, 400, 600, 0]),
    "one": dict(n=10, k=1, seed0=9200, grid=[40, 50, 60, 70, 80, 100, 120, 160, 200, 300, 0]),
    "all": dict(n=10, k=10, seed0=9300, grid=[40, 50, 60, 70, 80, 100, 120, 160, 200, 300, 0]),
}


def run(n, k, Ntr, seed):
    rng = np.random.default_rng(seed)
    Sigma = equicorr(n, RHO); Sinv = np.linalg.inv(Sigma); ridge = 1e-3 * np.eye(n)
    v = np.zeros(n); v[:k] = 1.0; mu = v * np.sqrt(NCP / (v @ Sinv @ v))
    pr = {kk: [] for kk in KEYS}
    for _ in range(R):
        perms = np.stack([rng.permutation(n) for _ in range(M)])
        if Ntr == 0:
            Shat = Sigma; Linv_r = linv1(Shat, perms); Sinv_hat = np.linalg.inv(Shat); Wsym = invsqrt(Shat)
            Vc = draw(np.zeros(n), Shat, B, rng)
            boot = order_methods(np.einsum('mij,bmj->bmi', Linv_r, Vc[:, perms]), n)
            boot["chi2"] = np.einsum('ki,ij,kj->k', Vc, Sinv_hat, Vc); boot["sym"] = simes2(Vc @ Wsym.T, n)
            if EXACT: boot.update({k + "_x": boot[k] for k in POOLED})   # known F: identical
            Linv0 = Linv_r
        else:
            G = (lambda X: X.T @ X / Ntr)(draw(np.zeros(n), Sigma, Ntr, rng)); Shat = G + ridge
            Linv_r = linv1(Shat, perms); Sinv_hat = np.linalg.inv(Shat); Wsym = invsqrt(Shat)
            Xtb = draw(np.zeros(n), Shat, B * Ntr, rng).reshape(B, Ntr, n)
            Gst = np.einsum('bki,bkj->bij', Xtb, Xtb) / Ntr; Sst = Gst + ridge
            Lst = linvB(Sst, perms); Vc = draw(np.zeros(n), Shat, B, rng)
            boot = order_methods(np.einsum('bmij,bmj->bmi', Lst, Vc[:, perms]), n)
            boot["chi2"] = np.einsum('bi,bij,bj->b', Vc, np.linalg.inv(Sst), Vc)
            boot["sym"] = simes2(np.einsum('bij,bj->bi', invsqrt(Sst), Vc), n)
            if EXACT:
                bx = order_methods(exact_scores(np.einsum('bmij,bmj->bmi', linvB(Gst, perms), Vc[:, perms]), Ntr), n)
                boot.update({k + "_x": bx[k] for k in POOLED}); Linv0 = linv1(G, perms)
        Xa = draw(mu, Sigma, NTE, rng)
        st = order_methods(np.einsum('mij,kmj->kmi', Linv_r, Xa[:, perms]), n)
        st["chi2"] = np.einsum('ki,ij,kj->k', Xa, Sinv_hat, Xa); st["sym"] = simes2(Xa @ Wsym.T, n)
        if EXACT:
            sx = order_methods(exact_scores(np.einsum('mij,kmj->kmi', Linv0, Xa[:, perms]), Ntr), n)
            st.update({k + "_x": sx[k] for k in POOLED})
        for kk in [k for k in KEYS if k.startswith(("pmerge", "bonf", "sym"))]:
            pr[kk].append(mc_low(boot[kk], st[kk], ALPHA).mean())
        for kk in [k for k in KEYS if k.startswith(("eavg", "chi2"))]:
            pr[kk].append(mc_high(boot[kk], st[kk], ALPHA).mean())
    return {kk: np.array(v) for kk, v in pr.items()}


if __name__ == "__main__":
    name = ARGS[0]; D = DESIGNS[name]; n = D["n"]
    PAIRS = [(a, b) for a in POOLED for b in ["sym", "chi2"]]
    if EXACT:
        PAIRS += [(a + "_x", b) for a in POOLED for b in ["sym", "chi2"]] + [(a + "_x", a) for a in POOLED]
    out = {f"pow_{kk}": [] for kk in KEYS}
    for a, b in PAIRS:
        out[f"gap_{a}_{b}"] = []; out[f"gapse_{a}_{b}"] = []
    for i, g in enumerate(D["grid"]):
        pr = run(n, D["k"], g, D["seed0"] + i)
        for kk in KEYS:
            out[f"pow_{kk}"].append(pr[kk].mean())
        for a, b in PAIRS:
            d = pr[a] - pr[b]
            out[f"gap_{a}_{b}"].append(d.mean()); out[f"gapse_{a}_{b}"].append(d.std(ddof=1) / np.sqrt(R))
        lab = "known" if g == 0 else "%dn" % (g // n)
        print("%-4s %-6s eavg %.3f pmerge %.3f bonf %.3f | sym %.3f chi2 %.3f | eavg-sym %+.3f (%.3f)"
              "  eavg-chi2 %+.3f (%.3f)" % (name, lab, *(out[f"pow_{kk}"][-1] for kk in KEYS[:5]),
                                            out["gap_eavg_sym"][-1], out["gapse_eavg_sym"][-1],
                                            out["gap_eavg_chi2"][-1], out["gapse_eavg_chi2"][-1]), flush=True)
        if EXACT:
            print("     exact: eavg %.3f pmerge %.3f bonf %.3f | exact-plug-in eavg %+.3f (%.3f)"
                  % (*(out[f"pow_{kk}_x"][-1] for kk in POOLED), out["gap_eavg_x_eavg"][-1],
                     out["gapse_eavg_x_eavg"][-1]), flush=True)
    np.savez("results_estimation_grid_%s%s.npz" % (name, "_exact" if EXACT else ""),
             Ntr=np.array([g if g else 100000 for g in D["grid"]]),
             **{k: np.array(v) for k, v in out.items()},
             n=n, k=D["k"], rho=RHO, ncp=NCP, M=M, R=R, B=B, alpha=ALPHA)
    print("saved results_estimation_grid_%s%s.npz" % (name, "_exact" if EXACT else ""))
