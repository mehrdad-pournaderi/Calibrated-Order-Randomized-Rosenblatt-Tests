"""
Pool the chunks written by exp_multiplicity_screening.py into the archive the figure script
reads.  Each chunk stores the raw per-replication false and true discovery proportions, so
pooling is a concatenation and the Monte-Carlo standard errors are exact.  With a sparse
screen (pi_0 = 0.95, ten non-nulls) the realized FDP takes few distinct values, which is why
the replications are pooled rather than run in one pass.

Usage: python merge_multiplicity_chunks.py
"""
import glob
import numpy as np

KEYS = ["single", "pmerge", "bonf", "sym", "eavg", "chi2"]

files = sorted(glob.glob("chunk_multiplicity_s*.npz"))
if not files:
    raise SystemExit("no chunks found")
chunks = [np.load(f) for f in files]
meta = chunks[0]

out = {}
for m in ("naive", "cal"):
    for k in KEYS:
        for s in ("fdr", "pow"):
            a = np.concatenate([c["%s_%s_%s" % (m, k, s)] for c in chunks], axis=1)  # (n_Ntr, R)
            out["%s_%s_%s" % (m, k, s)] = a.mean(1)
            out["%s_%s_%s_se" % (m, k, s)] = a.std(1, ddof=1) / np.sqrt(a.shape[1])
R = sum(int(c["R"]) for c in chunks)

print("pooled %d chunk(s), R=%d, pi_0=%.2f" % (len(files), R, float(meta["pi0"])))
for i, lab in enumerate(["80", "160", "known"]):
    print("  N_tr=%s" % lab)
    for k in KEYS:
        print("    %-7s naive[FDR=%.3f pow=%.3f]  calibrated[FDR=%.3f+-%.3f pow=%.3f+-%.3f]"
              % (k, out["naive_%s_fdr" % k][i], out["naive_%s_pow" % k][i],
                 out["cal_%s_fdr" % k][i], out["cal_%s_fdr_se" % k][i],
                 out["cal_%s_pow" % k][i], out["cal_%s_pow_se" % k][i]))

np.savez("results_multiplicity_screening.npz", Ntr=meta["Ntr"], R=R, nchunks=len(files),
         q=meta["q"], n=meta["n"], M=meta["M"], ncp=meta["ncp"], rho=meta["rho"],
         nunits=meta["nunits"], pi0=meta["pi0"], B=meta["B"], **out)
print("saved results_multiplicity_screening.npz")
