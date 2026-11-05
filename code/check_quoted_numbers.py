"""
Programmatic verification of every number the manuscript quotes from the single-test and FX
archives (the experiments rerun on 19 Sep 2026 under rank-based calibration).

Run from a directory containing the .npz archives (results/) and, optionally, pass the path to
paper.tex as argv[1] to also confirm that each expected string is present in the source.

Each check states the archive value, the value as the paper quotes it, and PASS/FAIL.

Nov 2026: under an estimated null the paper's Rosenblatt-based tests use the exact predictive
components (exact_whitening.py).  Their numbers are read from the *_exact.npz archives (keys
<method>_x); the plug-in numbers the paper still quotes for comparison, and every known-F number, are
read from the plug-in keys, which equal the original archives exactly.
"""
import sys
import numpy as np

fails = 0
tex = open(sys.argv[1]).read() if len(sys.argv) > 1 else None


def check(label, value, quoted, nd=3, tol=None):
    """value: archive number; quoted: number as printed in the paper (already rounded)."""
    global fails
    r = round(float(value), nd)
    ok = abs(r - quoted) < (tol if tol is not None else 0.5 * 10 ** -nd + 1e-12)
    if not ok:
        fails += 1
    print("  %-62s archive=%.4f  paper=%-7g %s" % (label, value, quoted, "PASS" if ok else "FAIL"))
    return ok


def intex(s):
    global fails
    if tex is None:
        return
    ok = s in tex
    if not ok:
        fails += 1
    print("  %-62s %s" % ("tex contains %r" % s[:40], "PASS" if ok else "FAIL"))


def claim(label, ok):
    global fails
    fails += (not ok)
    print("  %-62s %s" % (label, "PASS" if ok else "FAIL"))


# ---------------------------------------------------------------- Fig 2 / §6.1
c = np.load("results_calibration_pipeline.npz")
print("Figure 2 / §6.1 (results_calibration_pipeline.npz)")
check("naive size single, Ntr=4n", c["single_ns"][0], 0.20, 2)
check("naive size chi2, Ntr=4n", c["chi2_ns"][0], 0.28, 2)
check("naive size fisher, Ntr=4n", c["fisher_ns"][0], 0.28, 2)
check("naive size sym, Ntr=4n", c["sym_ns"][0], 0.19, 2)
cx = np.load("results_calibration_pipeline_exact.npz")      # same draws; plug-in keys equal c
print("Figure 2 exact components (results_calibration_pipeline*_exact.npz)")
for i, q in enumerate([0.448, 0.471, 0.495]):
    check("Fig2C p-merge (exact) power col %d" % i, cx["pmerge_x_cp"][i], q)
for i, q in enumerate([0.410, 0.438, 0.475]):
    check("Fig2C sym power col %d" % i, cx["sym_cp"][i], q)
for i, q in enumerate([0.034, 0.046, 0.032]):
    check("Fig2 eavg - e1 (exact) col %d" % i, cx["eavg_x_cp"][i] - cx["e1_x_cp"][i], q)
intex("\\addD{$0.448$, $0.471$}, $0.495$")
intex("$0.410$, $0.438$, $0.475$")
intex("\\addD{$+0.034$, $+0.046$},\n$+0.032$")
cx10 = None   # filled below with all ten training sizes


# ---------------------------------------------------------------- §4 sparse-vs-dense at equal KL (results_base_statistic.npz)
bs = np.load("results_base_statistic.npz")
print("§4 sparse vs dense at KL=4, n=50 (results_base_statistic.npz)")
j = int(np.argmin(np.abs(bs["kappas"] - 4)))
check("two-sided Simes, sparse (36%)", bs["pow_sparse2"][j], 0.36, 2)
check("two-sided Simes, dense (10%)", bs["pow_dense2"][j], 0.10, 2)
check("one-sided Simes, sparse (43%)", bs["pow_sparse"][j], 0.43, 2)
check("one-sided Simes, dense (17%)", bs["pow_dense"][j], 0.17, 2)
intex("detects the departure $36\\%$ of the time")
intex("$10\\%$ when it is spread evenly over all fifty (one-sided: $43\\%$ and $17\\%$)")

# ---------------------------------------------------------------- calibrated-size sweep cells
print("§6.1 sweep-cell calibrated sizes")
cells = []
XKS = ("single", "pmerge", "bonf", "eavg", "e1", "fisher")
for f in ("results_shape_sparse_dense_exact.npz", "results_shape_rho_exact.npz", "results_shape_n_exact.npz"):
    d = np.load(f)
    for k in ("single", "pmerge", "bonf", "eavg", "e1", "chi2", "sym", "fisher"):
        cells += list(d["csz_%s_x" % k] if k in XKS else d["csz_" + k])
cells = np.array(cells)
ok = len(cells) == 64
fails += (not ok)
print("  %-62s %d %s" % ("number of cells", len(cells), "PASS" if ok else "FAIL"))
check("mean calibrated size", cells.mean(), 0.050)
check("min calibrated size", cells.min(), 0.042)
check("max calibrated size", cells.max(), 0.060)
intex("across all $64$ configuration")
intex("(range $0.042$--\\delD{$0.059$}\\addD{$0.060$})")

# ---------------------------------------------------------------- Fig 3 / §6.2
a = np.load("results_shape_sparse_dense.npz")
print("Figure 3 / §6.2 (results_shape_sparse_dense.npz, known F)")
S, D = 0, 1
check("sparse: Fisher", a["cpw_known_fisher"][S], 0.578)
check("sparse: chi2", a["cpw_known_chi2"][S], 0.629)
check("sparse: sym", a["cpw_known_sym"][S], 0.723)
check("dense: sym", a["cpw_known_sym"][D], 0.376)
check("sparse: p-merge", a["cpw_known_pmerge"][S], 0.730)
check("sparse: eavg", a["cpw_known_eavg"][S], 0.728)
check("dense: bonf", a["cpw_known_bonf"][D], 0.691)
check("dense: eavg", a["cpw_known_eavg"][D], 0.662)
check("dense: p-merge", a["cpw_known_pmerge"][D], 0.577)
check("sparse: bonf", a["cpw_known_bonf"][S], 0.713)
check("eavg - e1, sparse, known", a["g1e_k"][S], 0.025)
check("eavg - e1, dense, known", a["g1e_k"][D], 0.095)
# ranking claims
top2 = lambda i: set(sorted(["single", "pmerge", "bonf", "eavg", "e1", "chi2", "sym", "fisher"],
                            key=lambda k: -a["cpw_known_" + k][i])[:2])
ok = top2(S) == {"pmerge", "eavg"} and top2(D) == {"bonf", "eavg"}
fails += (not ok)
print("  %-62s %s" % ("top two: sparse {pmerge,eavg}, dense {bonf,eavg}", "PASS" if ok else "FAIL"))
ok = a["cpw_known_chi2"][D] > a["cpw_known_fisher"][D] > a["cpw_known_sym"][D]
fails += (not ok)
print("  %-62s %s" % ("dense: chi2 > Fisher > sym among references", "PASS" if ok else "FAIL"))
# Fisher-base remark (§4): Bonferroni at known F, dense: 0.642 vs 0.691
check("Fisher remark: bonf_f dense known", a["cpw_known_bonf_f"][D], 0.642)
check("Fisher remark: bonf dense known", a["cpw_known_bonf"][D], 0.691)
# Intro: sym collapses by more than 0.30; gain up to +0.17 for dense
ok = (a["cpw_known_sym"][S] - a["cpw_known_sym"][D]) > 0.30
fails += (not ok)
print("  %-62s %.3f %s" % ("intro: sym collapse > 0.30", a["cpw_known_sym"][S] - a["cpw_known_sym"][D], "PASS" if ok else "FAIL"))
check("intro: best pooled gain vs single, dense (bonf)", a["cpw_known_bonf"][D] - a["cpw_known_single"][D], 0.17, 2)
intex("Fisher $0.578$, $\\chi^2$\n$0.629$")
intex("competitor ($0.723$)")
intex("collapses to $0.376$")
intex("$p$-merge $0.730$ and the e-value average $0.728$")
intex("Bonferroni-over-orders $0.691$ and the e-value average $0.662$")
intex("drops to $0.577$")
intex("Bonferroni to $0.713$")
intex("known $F$ ($0.691$ vs.\\ $0.642$)")

# ---------------------------------------------------------------- rho / n sweeps
r = np.load("results_shape_rho.npz")
n = np.load("results_shape_n.npz")
print("§6.2 gains vs single with dependence (results_shape_rho.npz, known F)")
for i, q in enumerate([0.017, 0.042, 0.063]):
    check("eavg gain, rho=%s" % r["labels"][i][4:], r["gme_k"][i], q)
for i, q in enumerate([0.018, 0.041, 0.059]):
    check("p-merge gain, rho=%s" % r["labels"][i][4:], r["gm_k"][i], q)
tk = np.concatenate([d[g + "_k"] / d[g.replace("gm", "ge").replace("gme", "gee") + "_k"]
                     for d in (a, r, n) for g in ("gm", "gme")])
ok = tk.min() >= 5.0
fails += (not ok)
print("  %-62s min t=%.2f %s" % ("paired t >= 5 in every known-F cell (pmerge, eavg vs single)", tk.min(), "PASS" if ok else "FAIL"))
print("§6.2 aggregation-isolated gap eavg - e1")
for i, q in enumerate([0.012, 0.033, 0.049]):
    check("eavg - e1, rho=%s, known" % r["labels"][i][4:], r["g1e_k"][i], q)
check("eavg - e1, n sweep known, min", n["g1e_k"].min(), 0.021)
check("eavg - e1, n sweep known, max", n["g1e_k"].max(), 0.040)
t1k = np.concatenate([d["g1e_k"] / d["g1ee_k"] for d in (a, r, n)])
t1s = np.concatenate([d["g1e_s"] / d["g1ee_s"] for d in (a, r, n)])
check("min paired t, eavg - e1, known F (paper: >= 4.7)", t1k.min() >= 4.7, 1, 0)
ax, rx, nx = (np.load(f) for f in ("results_shape_sparse_dense_exact.npz", "results_shape_rho_exact.npz",
                                    "results_shape_n_exact.npz"))
t1x = np.concatenate([d["gxe1_eavg_s"] / d["gxe1_eavg_s_se"] for d in (ax, rx, nx)])
claim_ok = t1x.min() >= 4.6
fails += (not claim_ok)
print("  %-62s min t=%.2f %s" % ("eavg - e1 under estimation (exact), paired t >= 4.6", t1x.min(), "PASS" if claim_ok else "FAIL"))
intex("$+0.017$ at $\\rho=0.2$ to $+0.042$ and $+0.063$")
intex("$+0.018$, $+0.041$, $+0.059$")
intex("$+0.012$, $+0.033$ and $+0.049$")
intex("$+0.021$ to $+0.040$")
intex("paired $t\\ge4.7$ in every known-$F$ cell, \\delD{$t\\ge3.3$}\\addD{$t\\ge4.6$} under estimation")
# Fisher-base remark across all configurations
# known F from the plug-in keys, N_tr = 4n from the exact components
SZ = (("cpw_", "_x"), ("cpw_known_", ""))
mxk1 = max(np.abs(d["cpw_known_pmerge_f"] - d["cpw_known_fisher"]).max() for d in (ax, rx, nx))
mxk2 = max((d["cpw_known_bonf_f"] - d["cpw_known_fisher"]).max() for d in (ax, rx, nx))
mxs1 = max(np.abs(d["cpw_pmerge_f_x"] - d["cpw_fisher_x"]).max() for d in (ax, rx, nx))
mxs2 = max((d["cpw_bonf_f_x"] - d["cpw_fisher_x"]).max() for d in (ax, rx, nx))
sp = []   # Simes-based combiner minus Fisher-based combiner, sparse configurations only
for d, cfg in ((ax, [0]), (rx, [0, 1, 2]), (nx, [0, 1, 2])):
    for pre, suf in SZ:
        for i in cfg:
            sp += [d[pre + "pmerge" + suf][i] - d[pre + "pmerge_f" + suf][i], d[pre + "bonf" + suf][i] - d[pre + "bonf_f" + suf][i]]
check("Fisher remark: sparse, Simes minus Fisher combiner, min (0.04)", min(sp), 0.04, 2)
check("Fisher remark: sparse, Simes minus Fisher combiner, max (0.29)", max(sp), 0.29, 2)
check("Fisher remark: dense known, p-merge(Fisher)", a["cpw_known_pmerge_f"][D], 0.623)
check("Fisher remark: dense known, p-merge(Simes)", a["cpw_known_pmerge"][D], 0.577)
check("Fisher remark: dense 4n, p-merge(Fisher), exact", ax["cpw_pmerge_f_x"][D], 0.533)
check("Fisher remark: dense 4n, p-merge(Simes), exact", ax["cpw_pmerge_x"][D], 0.540)
check("Fisher remark: dense 4n, bonf(Fisher), exact", ax["cpw_bonf_f_x"][D], 0.510)
check("Fisher remark: dense 4n, bonf(Simes), exact", ax["cpw_bonf_x"][D], 0.605)
intex("($0.540$ vs.\\ $0.533$ and $0.605$ vs.\\ $0.510$)")
sp_ok = all(d[pre + k + suf][i] > d[pre + k + "_f" + suf][i] for d, cfg in ((ax, [0]), (rx, [0, 1, 2]), (nx, [0, 1, 2])) for pre, suf in SZ for i in cfg for k in ("pmerge", "bonf"))
fails += (not sp_ok)
print("  %-62s %s" % ("§3.2: Simes base wins every sparse config, both training sizes", "PASS" if sp_ok else "FAIL"))
ok = a["cpw_known_pmerge_f"][D] > a["cpw_known_pmerge"][D] and a["cpw_known_bonf"][D] > a["cpw_known_bonf_f"][D] \
     and ax["cpw_pmerge_x"][D] > ax["cpw_pmerge_f_x"][D] and ax["cpw_bonf_x"][D] > ax["cpw_bonf_f_x"][D]
fails += (not ok)
print("  %-62s %s" % ("Fisher base leads only for p-merge at known F (dense)", "PASS" if ok else "FAIL"))
claim_ok = mxk1 <= 0.005
fails += (not claim_ok)
print("  %-62s %.4f %s" % ("Fisher remark: |p-merge(Fisher) - Fisher| <= 0.005, known F", mxk1, "PASS" if claim_ok else "FAIL"))
check("Fisher remark: Bonferroni(Fisher) - Fisher max, known F (+0.02)", mxk2, 0.02, 2)
check("Fisher remark: |p-merge(Fisher) - Fisher| max, 4n exact (0.017)", mxs1, 0.017)
check("Fisher remark: Bonferroni(Fisher) - Fisher max, 4n exact (+0.05)", mxs2, 0.05, 2)
# intro: e-value average vs single ordering at 4n, sparse; symmetric-root edge at 4n, sparse
gs = np.array([ax["gxs_eavg_s"][0]] + list(rx["gxs_eavg_s"]) + list(nx["gxs_eavg_s"]))
check("intro: e-avg - single at 4n (exact), sparse, min (+0.04)", gs.min(), 0.04, 2)
check("intro: e-avg - single at 4n (exact), sparse, max (+0.06)", gs.max(), 0.06, 2)
edge = max(-ax["geavg_x_sym_s"][0], *(-rx["geavg_x_sym_s"]), *(-nx["geavg_x_sym_s"]))
claim_ok = 0 < edge <= 0.02
fails += (not claim_ok)
print("  %-62s %.3f %s" % ("intro: symmetric-root edge at 4n, sparse, at most 0.02", edge, "PASS" if claim_ok else "FAIL"))
intex("($+0.04$ to $+0.06$ for the e-value average at $\\Ntr=4n$)")

# ---------------------------------------------------------------- §8 FX
f = np.load("results_fx_application_exact.npz", allow_pickle=True)   # plug-in keys equal the original archive
rows, keys = f["rows_x"], list(f["keys"])          # estimated-population null, exact predictive scores
col = lambda k: 2 + keys.index(k)
yr = lambda y: rows[rows[:, 0] == y][0]
print("§8 FX application (results_fx_application_exact.npz)")
for y, q in [(2017, 0.06), (2021, 0.03), (2023, 0.06), (2024, 0.05)]:
    check("eavg rejection rate %d (calm)" % y, yr(y)[col("eavg")], q, 2)
for y, q in [(2020, 0.30), (2022, 0.49), (2025, 0.16), (2016, 0.14)]:
    check("eavg rejection rate %d (stress)" % y, yr(y)[col("eavg")], q, 2)
check("day-level gap eavg - single (exact)", float(f["gap_x_mean"]), 0.006, 3)
check("day-level gap NW standard error (exact)", float(f["gap_x_se"]), 0.004, 3)
intex("$+0.006\\pm0.004$ under the estimated-population calibration")
ok = int(f["ebh_flag"].sum()) == 78
fails += (not ok)
print("  %-62s %d %s" % ("e-BH flags 78 days (issued-forecast e-values)", int(f["ebh_flag"].sum()), "PASS" if ok else "FAIL"))
ok = int(f["ebh_flag_x"].sum()) == 62 and bool(np.all(f["ebh_flag"][f["ebh_flag_x"]]))
fails += (not ok)
print("  %-62s %d %s" % ("e-BH on exact e-values flags 62 days, all among the 78", int(f["ebh_flag_x"].sum()), "PASS" if ok else "FAIL"))
Ex = np.exp(f["logE_x"]); Ep = np.exp(f["logE"])
ok = f["dates"][np.argmax(Ex)] == "2016-06-24" and 4.5e13 <= Ex.max() < 5.5e13
fails += (not ok)
print("  %-62s %.2e %s" % ("exact e-value of the referendum day ~5e13, the largest", Ex.max(), "PASS" if ok else "FAIL"))
ok = set(np.argsort(-Ex)[:8]) == set(np.argsort(-Ep)[:8])
fails += (not ok)
print("  %-62s %s" % ("same eight days lead under exact and issued-forecast e-values", "PASS" if ok else "FAIL"))
rk = lambda v: np.argsort(np.argsort(v))
rho_s = np.corrcoef(rk(f["logE_x"]), rk(f["logE"]))[0, 1]
ok = rho_s > 0.9999
fails += (not ok)
print("  %-62s %.6f %s" % ("rank correlation of exact and issued-forecast e-values > 0.9999", rho_s, "PASS" if ok else "FAIL"))
intex("e-BH flags $62$ days, all\namong the $78$")
ok = yr(2020)[col("sym")] > max(yr(2020)[col("chi2")], yr(2020)[col("eavg")]) and \
     yr(2022)[col("chi2")] > max(yr(2022)[col("sym")], yr(2022)[col("eavg")])
fails += (not ok)
print("  %-62s %s" % ("sym leads 2020, chi2 leads 2022", "PASS" if ok else "FAIL"))
d20 = yr(2020)[col("sym")] - yr(2020)[col("eavg")]
d22 = yr(2022)[col("chi2")] - yr(2022)[col("eavg")]
ok = 0.005 <= min(d20, d22) and max(d20, d22) <= 0.05
fails += (not ok)
print("  %-62s %.3f, %.3f %s" % ("pooled tracks leader within 0.01-0.05", d20, d22, "PASS" if ok else "FAIL"))
E = np.exp(f["logE"]); dates = f["dates"]
top = np.argsort(-E)[:2]
ok = list(dates[top]) == ["2016-06-24", "2022-12-20"] and 1e19 < E[top[0]] < 1e21 and 1e17 < E[top[1]] < 1e18
fails += (not ok)
print("  %-62s %s %s" % ("largest e-values: Brexit ~1e20, BoJ 2022-12-20 ~5e17", list(dates[top]), "PASS" if ok else "FAIL"))
intex("$0.01$--\\delD{$0.04$}\\addD{$0.05$} of whichever is winning")
covid = {str(dt): float(e) for dt, e in zip(dates, E) if str(dt) in ("2020-03-18", "2020-03-20", "2020-03-23")}
ok = 4.5e12 <= min(covid.values()) and max(covid.values()) < 2e14   # 5.0e12 prints as 5 x 10^12
fails += (not ok)
print("  %-62s %s %s" % ("COVID days 2020-03-18/20/23 in 5e12..1e14", {k: "%.1e" % v for k, v in covid.items()}, "PASS" if ok else "FAIL"))
intex("2020-03-18/20/23 ($5\\times10^{12}$--$10^{14}$)")

# ---------------------------------------------------------------- §8 issued-forecast calibration (24 Sep)
print("§8 issued-forecast calibration (rows_fc)")
rfc = f["rows_fc"]
yfc = lambda y: rfc[rfc[:, 0] == y][0]
for y, q in [(2017, 0.06), (2021, 0.03), (2023, 0.08), (2024, 0.05)]:
    check("forecast-null eavg rate %d (calm)" % y, yfc(y)[col("eavg")], q, 2)
for y, q in [(2020, 0.32), (2022, 0.55), (2025, 0.17), (2016, 0.14)]:
    check("forecast-null eavg rate %d (stress)" % y, yfc(y)[col("eavg")], q, 2)
fc_ok = f["rows_fc"][:, col("eavg")].min() >= 0 and bool(np.all(f["rows_fc"][:, col("eavg")] >= f["rows_x"][:, col("eavg")] - 1e-12))
fails += (not fc_ok)
print("  %-62s %s" % ("§8: forecast-null eavg rate >= estimated-population rate every year", "PASS" if fc_ok else "FAIL"))
check("forecast-null day-level gap", float(f["gap_fc_mean"]), 0.004, 3)
check("forecast-null gap NW se", float(f["gap_fc_se"]), 0.004, 3)
intex("$0.06$, $0.03$, $0.08$, $0.05$ under the issued-forecast calibration")
intex("$32\\%$, $55\\%$, $17\\%$ and $14\\%$")

# ---------------------------------------------------------------- §7.2 threshold sweep (results_threshold_dependence.npz)
t = np.load("results_threshold_dependence.npz", allow_pickle=True)
print("§7.2 threshold sweep (results_threshold_dependence.npz; alphas %s)" % list(t["alphas"]))
check("fig2 case: p-merge at 0.05", t["fig2_pmerge"][0], 0.510)
check("fig2 case: bonf at 0.05", t["fig2_bonf"][0], 0.487)
check("fig2 case: p-merge at 0.0005", t["fig2_pmerge"][-1], 0.045)
check("fig2 case: bonf at 0.0005", t["fig2_bonf"][-1], 0.066)
ok = (1 - t["fig2_pmerge"][-1] / t["fig2_pmerge"][0]) > 0.9 and abs((1 - t["fig2_bonf"][-1] / t["fig2_bonf"][0]) - 6 / 7) < 0.02
fails += (not ok)
print("  %-62s %.3f, %.3f %s" % ("fig2 case: losses > 9/10 and ~6/7", 1 - t["fig2_pmerge"][-1] / t["fig2_pmerge"][0], 1 - t["fig2_bonf"][-1] / t["fig2_bonf"][0], "PASS" if ok else "FAIL"))
check("dense case: p-merge 0.05", t["dense_pmerge"][0], 0.757)
check("dense case: p-merge 0.0005", t["dense_pmerge"][-1], 0.094)
check("dense case: bonf 0.05", t["dense_bonf"][0], 0.892)
check("dense case: bonf 0.0005", t["dense_bonf"][-1], 0.393)
check("dense case: gap at 0.05", t["dense_bonf"][0] - t["dense_pmerge"][0], 0.135)
check("dense case: gap at 0.0005", t["dense_bonf"][-1] - t["dense_pmerge"][-1], 0.299)
intex("($0.510$ against $0.487$)")
intex("($0.045$ against $0.066$)")
rank_ok = all(int((np.stack([t[c + "_pmerge"], t[c + "_bonf"], t[c + "_eavg"]])[:, j] > t[c + "_eavg"][j]).sum()) <= 1 for c in ("fig2", "sparse", "dense") for j in range(5))
fails += (not rank_ok)
print("  %-62s %s" % ("§7.2: eavg first or second at every level in every case", "PASS" if rank_ok else "FAIL"))

# ---------------------------------------------------------------- §7.3 conformal comparison (Table 1)
print("§7.3 conformal comparison (results_conformal_ntr_*.npz)")
cf = {nn: np.load("results_conformal_ntr_%d.npz" % nn, allow_pickle=True) for nn in (400, 1000, 2000)}
cfx = {nn: np.load("results_conformal_ntr_%d_exact.npz" % nn) for nn in (400, 1000, 2000)}
# pooled rows on the exact predictive scores; chi2 and the conformal rows do not depend on the whitening
table = {"bonf_x": (0.642, 0.648, 0.666), "eavg_x": (0.629, 0.620, 0.637), "pmerge_x": (0.497, 0.454, 0.468),
         "chi2": (0.336, 0.338, 0.369), "knn": (0.073, 0.405, 0.520), "mahal": (0.038, 0.183, 0.336),
         "kde": (0.037, 0.141, 0.265), "ada-rf": (0.005, 0.116, 0.235), "ada-lr": (0.003, 0.026, 0.063)}
src = lambda nn, k: cfx[nn] if k.endswith("_x") else cf[nn]
for k, vals in table.items():
    for nn, q in zip((400, 1000, 2000), vals):
        check("Table 1 %s at N_tr=%d" % (k, nn), float(src(nn, k)[k + "_pow"]), q)
claim("chi2 row identical in the plug-in and exact archives",
      all(float(cfx[nn]["chi2_pow"]) == float(cf[nn]["chi2_pow"]) for nn in cf))
claim("Bonferroni-over-orders is the best row at every N_tr",
      all(float(cfx[nn]["bonf_x_pow"]) == max(float(src(nn, k)[k + "_pow"]) for k in table) for nn in cf))
for nn, z in ((1000, 11.0), (2000, 8.7)):
    g = float(cfx[nn]["bonf_x_pow"]) - float(cf[nn]["knn_pow"])
    se = np.hypot(float(cfx[nn]["bonf_x_se_pow"]), float(cf[nn]["knn_se_pow"]))
    check("gap bonf - knn in SEs at N_tr=%d" % nn, g / se, z, 1)
mx = max(float(src(nn, k)[k + "_fdr"]) for nn in cf for k in table)
check("largest realized FDR in Table 1", mx, 0.108)
mse_f = max(float(src(nn, k)[k + "_se_fdr"]) for nn in cf for k in table)
mse_p = max(float(src(nn, k)[k + "_se_pow"]) for nn in cf for k in table)
claim("Table 1 standard errors at most 0.012 (FDR) and 0.018 (power)", mse_f <= 0.012 and mse_p <= 0.018)
intex("\\addD{\\textbf{0.642}} & \\addD{\\textbf{0.648}} & \\addD{\\textbf{0.666}}")
intex("gaps of \\delD{$10.6$ and $8.4$}\\addD{$11.0$ and $8.7$}")
ok = all(int(cf[nn]["ell_oneclass"]) == nn // 2 and int(cf[nn]["ell_adadetect"]) == nn // 2 for nn in cf)
fails += (not ok)
print("  %-62s %s" % ("balanced split l = N_tr/2 for every conformal method", "PASS" if ok else "FAIL"))
intex("$0.038$ against $0.336$ at $\\Ntr=400$, $0.183$ against $0.338$ at $1000$, $0.336$ against\n$0.369$ at $2000$")

# ---------------------------------------------------------------- added training sizes (Nov 2026)
# Figures 2 and 4 with N_tr = 5n-7n and 12n-32n added (the *_extra.npz and *_extra2.npz archives),
# the twenty-setting combiner comparison of §7.2, the screen's p-merge FDR, and Appendix B (Table B.1).
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import table_estimation_grid


def merged(base, *extras):
    d = dict(np.load(base))
    for extra in extras:
        e = np.load(extra)
        m = len(d["Ntr"]); o = np.argsort(np.concatenate([d["Ntr"], e["Ntr"]]))
        for k in list(d):
            if k in e.files and np.ndim(d[k]) == 1 and len(d[k]) == m:
                d[k] = np.concatenate([d[k], e[k]])[o]
    return d




print("Figures 2 and 4 with ten training sizes, exact components (*_exact.npz)")
c6 = merged("results_calibration_pipeline_exact.npz", "results_calibration_pipeline_extra_exact.npz",
            "results_calibration_pipeline_extra2_exact.npz")
s6 = merged("results_multiplicity_screening_exact.npz", "results_multiplicity_screening_extra_exact.npz",
            "results_multiplicity_screening_extra2_exact.npz")
c6p = merged("results_calibration_pipeline.npz", "results_calibration_pipeline_extra.npz",
             "results_calibration_pipeline_extra2.npz")
s6p = merged("results_multiplicity_screening.npz", "results_multiplicity_screening_extra.npz",
             "results_multiplicity_screening_extra2.npz")
GRID = [80, 100, 120, 140, 160, 240, 320, 480, 640, 100000]
claim("Fig 2 and Fig 4 grids are 4n-8n, 12n, 16n, 24n, 32n, known",
      list(c6["Ntr"]) == GRID and list(s6["Ntr"]) == GRID)
claim("plug-in halves of the exact archives equal the original archives",
      all(np.array_equal(c6[k], c6p[k]) for k in c6p if k in c6) and all(np.array_equal(s6[k], s6p[k]) for k in s6p if k in s6))
X = lambda k: k + "_x" if k in ("single", "pmerge", "bonf", "fisher", "eavg", "e1") else k
# Figure 2A: naive sizes on the exact components
for k, lo, hi in (("single", 0.046, 0.054), ("fisher", 0.046, 0.054), ("bonf", 0.014, 0.020), ("pmerge", 0.008, 0.012)):
    v = c6[X(k) + "_ns"]
    claim("Fig 2A: naive size %s (exact) in [%.3f, %.3f] at every N_tr" % (k, lo, hi),
          bool(np.all((np.round(v, 3) >= lo) & (np.round(v, 3) <= hi))))
claim("Fig 2A: naive size e-avg (exact) 0.001 at every N_tr", bool(np.all(np.round(c6["eavg_x_ns"], 3) == 0.001)))
intex("between $0.046$ and $0.054$ at every training size")
intex("\\addD{($0.014$ to\n$0.020$)}")
intex("\\addD{($0.008$ to $0.012$)}")
cs = np.stack([c6[X(k) + "_cs"] for k in ("single", "pmerge", "bonf", "fisher", "sym", "eavg", "e1", "chi2")])
claim("Fig 2B: every calibrated size in [0.042, 0.061] at every N_tr", bool(np.all((cs >= 0.042) & (cs <= 0.061))))
for k in ("chi2", "sym", "single", "fisher"):
    claim("Fig 2C: p-merge > %s at all ten N_tr" % k, bool(np.all(c6["pmerge_x_cp"] > c6[X(k) + "_cp"])))
claim("Fig 2: Bonferroni last of the three pooled at all ten N_tr",
      bool(np.all(c6["bonf_x_cp"] < np.minimum(c6["pmerge_x_cp"], c6["eavg_x_cp"]))))
claim("Fig 2: e-avg ahead of both baselines at all ten N_tr",
      bool(np.all(c6["eavg_x_cp"] > np.maximum(c6["sym_cp"], c6["chi2_cp"]))))
# Figure 4 and §7.1
mid = list(range(1, 9))                               # 5n ... 32n
check("§7.1: p-merge calibrated FDR at 4n (exact)", s6["cal_pmerge_x_fdr"][0], 0.152)
z0 = (s6["cal_pmerge_x_fdr"][0] - 0.095) / s6["cal_pmerge_x_fdr_se"][0]
check("§7.1: p-merge excess at 4n in standard errors (five)", z0, 5, 0)
check("Fig 4B / §7.1: p-merge calibrated FDR, min over 5n-32n", s6["cal_pmerge_x_fdr"][mid].min(), 0.102)
check("Fig 4B / §7.1: p-merge calibrated FDR, max over 5n-32n", s6["cal_pmerge_x_fdr"][mid].max(), 0.126)
z = (s6["cal_pmerge_x_fdr"] - 0.095) / s6["cal_pmerge_x_fdr_se"]
claim("§7.1: p-merge excess within two SEs from 5n on except at 6n and 7n",
      bool(np.all(z[[1, 4, 5, 6, 7, 8]] < 2) and np.all(z[[2, 3]] >= 2)))
claim("Fig 4B: p-merge within two SEs from 8n on", bool(np.all(z[4:] < 2)))
claim("§7.1: no p-merge excess at known F", bool(s6["cal_pmerge_x_fdr"][-1] <= 0.095))
intex("$p$-merge realizes \\delD{$0.137$}\\addD{$0.152$} at $\\Ntr=4n$, \\delD{three}\\addD{five} standard errors")
intex("$0.102$ to $0.126$")
# naive screen on exact components
nfs = s6["naive_single_x_fdr"]
check("§7.1: naive BH single ordering (exact), FDR min", nfs.min(), 0.09, 2)
check("§7.1: naive BH single ordering (exact), FDR max", nfs.max(), 0.11, 2)
claim("§7.1: naive BH Bonferroni (exact) FDR at most 0.05", bool(np.all(s6["naive_bonf_x_fdr"] <= 0.05)))
claim("§7.1: e-BH on exact e-avg, realized FDR 0.000", bool(np.all(np.round(s6["naive_eavg_x_fdr"], 3) == 0)))
check("§7.1: e-BH on exact e-avg, largest power (at most 0.09)", s6["naive_eavg_x_pow"].max(), 0.09, 2)
check("§7.1: naive plug-in BH FDR chi2 at 4n", s6p["naive_chi2_fdr"][0], 0.80, 2)
check("§7.1: naive plug-in BH FDR single at 4n", s6p["naive_single_fdr"][0], 0.66, 2)
check("§7.1: naive plug-in BH FDR sym at 4n", s6p["naive_sym_fdr"][0], 0.61, 2)
intex("BH realizes $0.09$ to $0.11$ on a single ordering and at most\n$0.05$ on Bonferroni-over-orders")
pooled = np.stack([s6["cal_%s_x_pow" % k] for k in ("pmerge", "bonf", "eavg")])
base = np.maximum(s6["cal_sym_pow"], s6["cal_chi2_pow"])
claim("Fig 4C: every pooled combiner beats both baselines at every N_tr", bool(np.all(pooled > base)))
others = np.stack([s6["cal_%s_pow" % k] for k in ("single_x", "sym", "chi2")]).max(0)
claim("§7.1: Bonferroni and e-avg lead single, sym, chi2 at every N_tr",
      bool(np.all(np.minimum(s6["cal_bonf_x_pow"], s6["cal_eavg_x_pow"]) > others)))
for k, q in (("bonf_x", 0.579), ("eavg_x", 0.576), ("single_x", 0.397), ("sym", 0.320), ("chi2", 0.240)):
    check("§7.1: screen power at 8n, %s" % k, s6["cal_%s_pow" % k][4], q)
for k, q in (("bonf", 0.642), ("eavg", 0.619), ("single", 0.474), ("sym", 0.353), ("chi2", 0.349)):
    check("§7.1: screen power at known F, %s" % k, s6["cal_%s_pow" % k][-1], q)
check("Appendix B: screen, e-avg - sym at 4n (exact)", s6["cal_eavg_x_pow"][0] - s6["cal_sym_pow"][0], 0.252)
intex("\\addD{$0.579$ and $0.576$ against $0.397$}, $0.320$ and $0.240$")

print("§7.2 which combiner, twenty settings (Figures 2 and 4, ten training sizes each, exact components)")
P = {k: np.concatenate([c6[k + "_x_cp"], s6["cal_%s_x_pow" % k]]) for k in ("pmerge", "bonf", "eavg")}
lead = np.max(np.stack(list(P.values())), axis=0)
check("max shortfall e-avg vs leader, twenty settings", (lead - P["eavg"]).max(), 0.023)
check("max shortfall Bonferroni vs leader, twenty settings", (lead - P["bonf"]).max(), 0.067)
check("max shortfall p-merge vs leader, twenty settings", (lead - P["pmerge"]).max(), 0.212)
check("p-merge trails Bonferroni in the screen at known F", s6["cal_bonf_x_pow"][-1] - s6["cal_pmerge_x_pow"][-1], 0.212)
rank = (np.stack([P["pmerge"], P["bonf"]]) > P["eavg"]).sum(0) + 1
claim("e-avg second in 13 and first in 7 of the twenty, never last",
      int((rank == 2).sum()) == 13 and int((rank == 1).sum()) == 7 and int((rank == 3).sum()) == 0)
sl = np.argmax(np.stack([s6["cal_%s_x_pow" % k] for k in ("pmerge", "bonf", "eavg")]), axis=0)
claim("Bonferroni leads the screen at 6n and from 8n on", list(np.flatnonzero(sl == 1)) == [2] + list(range(4, 10)))
intex("thirteen of the twenty\nsettings and first in the remaining seven")
intex("\\delD{$0.040$, against $0.076$}\\addD{$0.023$, against $0.067$}")

print("Appendix B, Table B.1 (results_estimation_grid_*_exact.npz)")
G = {d: np.load("results_estimation_grid_%s_exact.npz" % d) for d in ("two", "one", "all")}
for d in G:
    gp = np.load("results_estimation_grid_%s.npz" % d)
    claim("Table B.1 %s: plug-in half equals the original archive" % d, all(np.array_equal(G[d][k], gp[k]) for k in gp.files))
for r in table_estimation_grid.rows():
    intex(r)
lo, hi = table_estimation_grid.se_range()
claim("Table B.1 standard errors 0.002 to 0.004 (%.4f-%.4f)" % (lo, hi), round(lo, 3) >= 0.002 and round(hi, 3) <= 0.004)
claim("Table B.1: comparator is the stronger baseline at every N_tr",
      bool(np.all(G["two"]["pow_sym"] >= G["two"]["pow_chi2"]) and np.all(G["one"]["pow_sym"] >= G["one"]["pow_chi2"])
           and np.all(G["all"]["pow_chi2"] >= G["all"]["pow_sym"])))
claim("Table B.1: sym and chi2 tied at known F for two coordinates",
      abs(G["two"]["pow_sym"][-1] - G["two"]["pow_chi2"][-1]) < 0.0005)
def gap(d, mult, meth="eavg_x"):
    a = G[d]; b = "chi2" if d == "all" else "sym"; i = int(np.flatnonzero(a["Ntr"] == mult * int(a["n"]))[0])
    return a["gap_%s_%s" % (meth, b)][i], a["gapse_%s_%s" % (meth, b)][i]
MULTS = (4, 5, 6, 7, 8, 10, 12, 16, 20, 30)
claim("exact, two coordinates: ahead at every N_tr", all(gap("two", m)[0] > 2 * gap("two", m)[1] for m in MULTS))
claim("exact, all coordinates: ahead at every N_tr", all(gap("all", m)[0] > 2 * gap("all", m)[1] for m in MULTS))
check("exact, two coordinates: smallest lead", min(gap("two", m)[0] for m in MULTS), 0.021)
check("exact, two coordinates: largest lead (known F)", G["two"]["gap_eavg_x_sym"][-1], 0.037)
check("exact, all coordinates: largest lead", max(gap("all", m)[0] for m in MULTS), 0.105)
check("exact, all coordinates: smallest lead (known F)", G["all"]["gap_eavg_x_chi2"][-1], 0.027)
check("exact, one coordinate: largest deficit up to 8n", -min(gap("one", m)[0] for m in (4, 5, 6, 7, 8)), 0.006)
claim("exact, one coordinate: level or ahead from 10n on",
      all(gap("one", m)[0] > -2 * gap("one", m)[1] for m in (10, 12, 16, 20, 30)))
check("plug-in, all coordinates: deficit at 4n", -gap("all", 4, "eavg")[0], 0.068)
claim("plug-in: level at 5n (two), 10n (one), 12n (all), behind just before",
      abs(gap("two", 5, "eavg")[0]) < 2 * gap("two", 5, "eavg")[1] and gap("two", 4, "eavg")[0] < -2 * gap("two", 4, "eavg")[1]
      and abs(gap("one", 10, "eavg")[0]) < 2 * gap("one", 10, "eavg")[1] and gap("one", 8, "eavg")[0] < -2 * gap("one", 8, "eavg")[1]
      and abs(gap("all", 12, "eavg")[0]) < 2 * gap("all", 12, "eavg")[1] and gap("all", 10, "eavg")[0] < -2 * gap("all", 10, "eavg")[1])
intex("($+0.021$ to $+0.037$)")
intex("($+0.027$ to $+0.105$")
intex("edge of at most\n$0.006$ up to $\\Ntr=8n$")

# ---------------------------------------------------------------- number of orderings (Nov 2026)
print("§4 / §7.2 how many orderings (results_orderings_evidence.npz, results_number_of_orderings.npz)")
ev = np.load("results_orderings_evidence.npz")
NAMES = ("one", "all", "two", "sweep")
j12 = int(np.flatnonzero(ev["MGRID"] == 12)[0])
Vbar = [float(ev["Vmean_" + d]) for d in NAMES]
gain = [float(ev["L_" + d][-1] - ev["L_" + d][0]) for d in NAMES]
miss = [float(ev["L_" + d][-1] - ev["L_" + d][j12]) for d in NAMES]
share = [float(ev["share_" + d][j12]) for d in NAMES]
check("mean spread V, smallest design", min(Vbar), 0.24, 2)
check("mean spread V, largest design", max(Vbar), 1.16, 2)
check("log-evidence uncollected at M=12, smallest", min(miss), 0.009)
check("log-evidence uncollected at M=12, largest", max(miss), 0.054)
check("share collected at M=12, smallest (%)", 100 * min(share), 86, 0)
check("share collected at M=12, largest (%)", 100 * max(share), 92, 0)
check("total order-averaging gain, smallest", min(gain), 0.09, 2)
check("total order-averaging gain, largest", max(gain), 0.39, 2)
big = ev["MGRID"] >= 32
ok = all(np.all(np.abs((ev["L_" + d][-1] - ev["L_" + d])[big & (ev["MGRID"] < ev["MMAX"])]
                       - float(ev["Vmean_" + d]) / (2 * ev["MGRID"][big & (ev["MGRID"] < ev["MMAX"])])) < 0.002)
         for d in NAMES)
claim("uncollected log-evidence within 0.002 of V/(2M) for 32 <= M < 2048", ok)
claim("expected log-evidence nondecreasing in M in every design",
      all(np.all(np.diff(ev["L_" + d]) > -2 * ev["Lse_" + d][1:]) for d in NAMES))
no = np.load("results_number_of_orderings.npz"); Mg = list(no["Mgrid"])
k1, k12 = Mg.index(1), Mg.index(12); beyond = [Mg.index(m) for m in Mg if m > 12]
for k, q in (("pmerge", -0.049), ("eavg", -0.038), ("bonf", -0.034)):
    check("M sweep %s: M=1 minus M=12" % k, no["dep_%s_gap12" % k][k1], q)
worst = max(abs(no["dep_%s_gap12" % k][b]) for k in ("pmerge", "eavg", "bonf") for b in beyond)
check("M sweep: largest change from M=12 to any M up to 128", worst, 0.005)
se = max(no["dep_%s_gap12_se" % k][b] for k in ("pmerge", "eavg", "bonf") for b in beyond)
check("M sweep: paired standard error of those changes", se, 0.001)
claim("M sweep has 400 realizations at rho=0.6", int(no["R_dep"]) == 400)
claim("rho=0: every combiner flat in M", all(np.ptp(no["ind_" + k]) < 1e-12 for k in ("pmerge", "bonf", "eavg")))
check("null spread V (n=10, rho=0.5)", float(ev["Vmean_null"]), 0.044)
check("predicted uncollected at M=12, smallest (Vbar/24)", min(Vbar) / 24, 0.010)
check("predicted uncollected at M=12, largest (Vbar/24)", max(Vbar) / 24, 0.048)
intex("$0.049$ for $p$-merge")
intex("lies between $0.24$ and $1.16$")
intex("$86\\%$ to $92\\%$ of the incremental")

# ---------------------------------------------------------------- decision stability (Nov 2026)
print("§7.2 decision stability (results_decision_stability.npz)")
ds = np.load("results_decision_stability.npz"); Mg = list(ds["MGRID"]); e = list(ds["keys"]).index("eavg")
for name, h, m, q in (("one", "alt", 12, 3.3), ("all", "alt", 12, 6.2), ("one", "null", 12, 1.4),
                      ("all", "null", 12, 1.3), ("one", "alt", 1, 12.6), ("all", "alt", 1, 18.4)):
    check("disagreement %%, e-avg, %s, %s, M=%d" % (name, h, m), 100 * ds["dis_%s_%s" % (h, name)][e][Mg.index(m)], q, 1)
ratios = [ds["dis_%s_%s" % (h, nm)][e][Mg.index(12)] / ds["dis_%s_%s" % (h, nm)][e][Mg.index(128)]
          for nm in ("one", "all") for h in ("alt", "null")]
claim("M=128 roughly halves the M=12 disagreement (ratios %.1f-%.1f)" % (min(ratios), max(ratios)),
      all(1.7 <= r <= 2.5 for r in ratios))
intex("$3.3\\%$ of observations")

print("\nRESULT: %d failure(s)" % fails)
sys.exit(1 if fails else 0)
