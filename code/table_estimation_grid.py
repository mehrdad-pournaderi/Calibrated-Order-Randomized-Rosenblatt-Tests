"""
Rows of Table B.1 (Appendix B), built directly from results_estimation_grid_{two,one,all}_exact.npz:
calibrated power of the e-value average minus that of the stronger order-invariant baseline
(symmetric-root test for `two` and `one`, chi^2 for `all`), with the e-value average on the exact
predictive components (the paper's procedure) and, beside it, on the plug-in components of the same
draws.  The plug-in half of each archive equals results_estimation_grid_<design>.npz exactly.
Run from results/; check_quoted_numbers.py uses rows() to confirm that the table in paper.tex
matches the archives, and se_range() for the standard errors quoted in the caption.
"""
import numpy as np

COLS = [("two", "sym"), ("one", "sym"), ("all", "chi2")]
LABELS = [4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 0]          # multiples of n; 0 = known F


def _arch():
    return {d: np.load("results_estimation_grid_%s_exact.npz" % d) for d, _ in COLS}


def rows():
    arch = _arch()
    out = []
    for m in LABELS:
        cells = []
        for d, base in COLS:
            a = arch[d]; n = int(a["n"])
            target = 100000 if m == 0 else m * n
            idx = np.flatnonzero(a["Ntr"] == target)
            for meth in ("eavg_x", "eavg"):
                cells.append("$%+.3f$" % a["gap_%s_%s" % (meth, base)][int(idx[0])] if len(idx) else "---")
        out.append(("known $F$" if m == 0 else "$%dn$" % m) + " & " + " & ".join(cells) + " \\\\")
    return out


def se_range():
    arch = _arch()
    se = np.concatenate([arch[d]["gapse_%s_%s" % (meth, base)] for d, base in COLS for meth in ("eavg_x", "eavg")])
    return float(se.min()), float(se.max())


if __name__ == "__main__":
    print("\n".join(rows()))
    print("standard errors of the paired differences: %.4f to %.4f" % se_range())
