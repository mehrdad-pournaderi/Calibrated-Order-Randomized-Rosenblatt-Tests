"""
Rows of Table B.1 (Appendix B), built directly from results_estimation_grid_{two,one,all}.npz:
calibrated power of the e-value average minus that of the stronger order-invariant baseline
(symmetric-root test for `two` and `one`, chi^2 for `all`), with the Monte-Carlo standard error of
the paired difference.  Run from results/; check_quoted_numbers.py uses rows() to confirm that the
table in paper.tex matches the archives.
"""
import numpy as np

COLS = [("two", "sym"), ("one", "sym"), ("all", "chi2")]
LABELS = [4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 0]          # multiples of n; 0 = known F


def cell(a, i, base):
    g, s = a["gap_eavg_%s" % base][i], a["gapse_eavg_%s" % base][i]
    return "$%+.3f$\\,(%.3f)" % (g, s)


def rows():
    arch = {d: np.load("results_estimation_grid_%s.npz" % d) for d, _ in COLS}
    out = []
    for m in LABELS:
        cells = []
        for d, base in COLS:
            a = arch[d]; n = int(a["n"])
            target = 100000 if m == 0 else m * n
            idx = np.flatnonzero(a["Ntr"] == target)
            cells.append(cell(a, int(idx[0]), base) if len(idx) else "---")
        out.append(("known $F$" if m == 0 else "$%dn$" % m) + " & " + " & ".join(cells) + " \\\\")
    return out


if __name__ == "__main__":
    print("\n".join(rows()))
