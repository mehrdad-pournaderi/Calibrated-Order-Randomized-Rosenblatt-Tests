"""
Pool the chunks written by exp_conformal_comparison.py and exp_conformal_novelty_density.py
into one archive per setting.  Each chunk holds the raw per-replication (FDP, TDP) of every
method, so pooling is a concatenation and the Monte-Carlo standard errors are exact.

NO TUNING IS SELECTED HERE.  Every nuisance choice is fixed in advance:

  * k = 1 for the nearest-neighbour score;
  * l = N_tr/2, a balanced split, for the one-class scores (Mahalanobis, nearest neighbour,
    kernel density), which want as large a calibration set as the reference sample allows;
  * for AdaDetect, l = m (the number of test points), which is the value recommended in
    Remark 2.2 of Marandon et al.  AdaDetect trains a classifier on the fit part of the
    reference sample, so a balanced split halves what it has to learn from and would
    understate it.  Their recommendation is subject to their own resolution condition,
    l > m/(q*m1) with m1 a lower bound on the number of rejections; where l = m fails that
    condition -- which happens once novelties are rare, m1 = 10 requiring l > 200 -- the
    balanced split is used instead, since l = m would then cap AdaDetect by p-value
    granularity rather than by its score and understate it far more severely.

  The rule is stated in advance and applied identically at every setting; it is a function of
  the design (m, q, m1), not of any realized power.

Choosing instead the best of several tuning values -- even on averages taken over
replications -- would report the maximum of several correlated estimates, which is biased
upward and is not the power of any procedure that could be run in practice.

The remaining neighbour counts are still computed (they cost nothing, sharing one distance
matrix) and are written to the archive under `knn_sens_*`, so that the sensitivity of the
nearest-neighbour score to k can be reported separately without ever feeding back into the
reported comparison.

Usage: python merge_conformal_chunks.py ntr 400 1000 2000
       python merge_conformal_chunks.py pi0 80 90 95
"""
import glob
import sys
import numpy as np

OURS = ["pmerge", "bonf", "eavg", "chi2"]
ONECLASS = ["mahal", "knn", "kde"]          # calibration size = N_tr/2
ADAPTIVE = ["ada-rf", "ada-lr"]             # calibration size = m  (Marandon et al., Rmk 2.2)
CONF = ONECLASS + ADAPTIVE
ALL = OURS + CONF
KNN_FIXED = 1                               # pre-specified; see the module docstring

kind = sys.argv[1] if len(sys.argv) > 1 else "ntr"
values = [int(a) for a in sys.argv[2:]] or ([400, 1000, 2000] if kind == "ntr" else [80, 90, 95])

for val in values:
    files = sorted(glob.glob("chunk_conformal_%s%d_s*.npz" % (kind, val)))
    if not files:
        print("no chunks for %s=%d" % (kind, val))
        continue
    chunks = [np.load(f, allow_pickle=True) for f in files]
    meta = chunks[0]
    lgrid = [int(x) for x in meta["lgrid"]]
    m = int(meta["nunits"])
    ntr = int(meta["ntr"])
    # AdaDetect: their recommended l = m, unless it fails their resolution condition
    n_alt = int(meta["n_alt"]); q = float(meta["q"])
    ell_ada = m if m > m / (q * n_alt) else ntr // 2
    ell = {k: ntr // 2 for k in ONECLASS}
    ell.update({k: ell_ada for k in ADAPTIVE})
    for k, l in ell.items():
        if l not in lgrid:
            raise SystemExit("calibration size %d for %s not present in chunks (%s)"
                             % (l, k, lgrid))
    name = {k: k for k in CONF}
    name["knn"] = "knn:%d" % KNN_FIXED

    def pooled(key):
        return np.concatenate([c[key] for c in chunks])

    out, se = {}, {}
    for k in ALL:
        a = pooled(k) if k in OURS else pooled("%s@%d" % (name[k], ell[k]))
        out[k], se[k] = a.mean(0), a.std(0, ddof=1) / np.sqrt(len(a))
    R = len(pooled("bonf"))

    kgrid = sorted({int(n.split("@")[0].split(":")[1]) for n in meta.files if n.startswith("knn:")})
    sens = {kk: pooled("knn:%d@%d" % (kk, ell["knn"])).mean(0) for kk in kgrid}

    print("%s=%-5d pooled over %d chunk(s), R=%d, k=%d, l=%d (one-class) / %d (AdaDetect)"
          "   (+- Monte-Carlo standard error)"
          % (kind, val, len(files), R, KNN_FIXED, ell["knn"], ell["ada-rf"]))
    for k in ALL:
        print("   %-8s FDR=%.3f+-%.3f  power=%.3f+-%.3f"
              % (k, out[k][0], se[k][0], out[k][1], se[k][1]))
    print("   [sensitivity, not used in the tables] knn power by k: "
          + "  ".join("k=%d:%.3f" % (kk, sens[kk][1]) for kk in kgrid))

    fn = "results_conformal_%s_%d.npz" % (kind, val)
    np.savez(fn, R=R, methods=np.array(ALL), nchunks=len(files), knn_k=KNN_FIXED,
             ell_oneclass=ell["knn"], ell_adadetect=ell["ada-rf"],
             **{"%s_%s" % (k, s): np.array(out[k][j])
                for k in ALL for j, s in ((0, "fdr"), (1, "pow"))},
             **{"%s_se_%s" % (k, s): np.array(se[k][j])
                for k in ALL for j, s in ((0, "fdr"), (1, "pow"))},
             knn_sens_k=np.array(kgrid),
             knn_sens_pow=np.array([sens[kk][1] for kk in kgrid]),
             knn_sens_fdr=np.array([sens[kk][0] for kk in kgrid]),
             **{f: meta[f] for f in ("ntr", "pi0", "n_alt", "q", "n", "nunits", "ncp", "M", "B")
                if f in meta.files})
    print("   saved %s\n" % fn)
