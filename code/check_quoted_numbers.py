"""
Programmatic verification of every number the manuscript quotes from the single-test and FX
archives (the experiments rerun on 19 Sep 2026 under rank-based calibration).

Run from a directory containing the .npz archives (results/) and, optionally, pass the path to
paper.tex as argv[1] to also confirm that each expected string is present in the source.

Each check states the archive value, the value as the paper quotes it, and PASS/FAIL. Numbers from
the multiplicity, conformal and threshold-dependence experiments are not covered here: those
archives were not touched by the rerun and their claims were verified when they were produced.
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


# ---------------------------------------------------------------- Fig 2 / §6.1
c = np.load("results_calibration_pipeline.npz")
print("Figure 2 / §6.1 (results_calibration_pipeline.npz)")
check("naive size single, Ntr=4n", c["single_ns"][0], 0.20, 2)
check("naive size chi2, Ntr=4n", c["chi2_ns"][0], 0.28, 2)
check("naive size fisher, Ntr=4n", c["fisher_ns"][0], 0.28, 2)
check("naive size sym, Ntr=4n", c["sym_ns"][0], 0.19, 2)
check("naive size eavg, Ntr=4n", c["eavg_ns"][0], 0.029, 3)
check("naive size eavg, known F", c["eavg_ns"][2], 0.001, 3)
for i, q in enumerate([0.437, 0.467, 0.495]):
    check("Fig2C p-merge power col %d" % i, c["pmerge_cp"][i], q)
for i, q in enumerate([0.410, 0.438, 0.475]):
    check("Fig2C sym power col %d" % i, c["sym_cp"][i], q)
for i, q in enumerate([0.024, 0.038, 0.032]):
    check("Fig2 eavg - e1 col %d" % i, c["eavg_cp"][i] - c["e1_cp"][i], q)
# p-merge beats every order-invariant reference, single and Fisher at every Ntr
for k in ("chi2", "sym", "single", "fisher"):
    ok = bool(np.all(c["pmerge_cp"] > c[k + "_cp"]))
    fails += (not ok)
    print("  %-62s %s" % ("p-merge > %s at every Ntr" % k, "PASS" if ok else "FAIL"))
# §7.2 'which combiner': largest shortfall vs leading pooled combiner over the three Fig-2 columns
lead = np.max(np.stack([c["pmerge_cp"], c["bonf_cp"], c["eavg_cp"]]), axis=0)
check("max shortfall eavg vs leader (Fig 2 cols)", (lead - c["eavg_cp"]).max(), 0.040)
check("max shortfall bonf vs leader (Fig 2 cols)", (lead - c["bonf_cp"]).max(), 0.076)
ok = bool(np.all(c["bonf_cp"] < np.minimum(c["pmerge_cp"], c["eavg_cp"])))
fails += (not ok)
print("  %-62s %s" % ("Bonferroni weakest of the three pooled in Fig 2", "PASS" if ok else "FAIL"))
intex("$0.437$, $0.467$, $0.495$")
intex("$0.410$, $0.438$, $0.475$")
intex("$0.040$, against $0.076$")

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
for f in ("results_shape_sparse_dense.npz", "results_shape_rho.npz", "results_shape_n.npz"):
    d = np.load(f)
    for k in d.files:
        if k.startswith("csz_") and k[4:] not in ("pmerge_f", "bonf_f"):
            cells += list(d[k])
cells = np.array(cells)
ok = len(cells) == 64
fails += (not ok)
print("  %-62s %d %s" % ("number of cells", len(cells), "PASS" if ok else "FAIL"))
check("mean calibrated size", cells.mean(), 0.050)
check("min calibrated size", cells.min(), 0.042)
check("max calibrated size", cells.max(), 0.059)
intex("across all $64$ configuration")

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
check("min paired t, eavg - e1, estimated", t1s.min(), 3.3, 1)
intex("$+0.017$ at $\\rho=0.2$ to $+0.042$ and $+0.063$")
intex("$+0.018$, $+0.041$, $+0.059$")
intex("$+0.012$, $+0.033$ and $+0.049$")
intex("$+0.021$ to $+0.040$")
intex("paired $t\\ge4.7$ in every known-$F$ cell, $t\\ge3.3$ under estimation")
# Fisher-base remark across all configurations
mx1, mx2 = [], []
for d in (a, r, n):
    for pre in ("cpw_", "cpw_known_"):
        mx1 += list(np.abs(d[pre + "pmerge_f"] - d[pre + "fisher"]))
        mx2 += list(d[pre + "bonf_f"] - d[pre + "fisher"])
sp = []   # Simes-based combiner minus Fisher-based combiner, sparse configurations only
for d, cfg in ((a, [0]), (r, [0, 1, 2]), (n, [0, 1, 2])):
    for pre in ("cpw_", "cpw_known_"):
        for i in cfg:
            sp += [d[pre + "pmerge"][i] - d[pre + "pmerge_f"][i], d[pre + "bonf"][i] - d[pre + "bonf_f"][i]]
check("Fisher remark: sparse, Simes minus Fisher combiner, min (0.04)", min(sp), 0.04, 2)
check("Fisher remark: sparse, Simes minus Fisher combiner, max (0.26)", max(sp), 0.26, 2)
check("Fisher remark: dense known, p-merge(Fisher)", a["cpw_known_pmerge_f"][D], 0.623)
check("Fisher remark: dense known, p-merge(Simes)", a["cpw_known_pmerge"][D], 0.577)
check("Fisher remark: dense 4n, p-merge(Fisher)", a["cpw_pmerge_f"][D], 0.502)
check("Fisher remark: dense 4n, p-merge(Simes)", a["cpw_pmerge"][D], 0.415)
check("Fisher remark: dense 4n, bonf(Fisher)", a["cpw_bonf_f"][D], 0.498)
check("Fisher remark: dense 4n, bonf(Simes)", a["cpw_bonf"][D], 0.421)
sp_ok = all(d[pre + k][i] > d[pre + k + "_f"][i] for d, cfg in ((a, [0]), (r, [0, 1, 2]), (n, [0, 1, 2])) for pre in ("cpw_", "cpw_known_") for i in cfg for k in ("pmerge", "bonf"))
fails += (not sp_ok)
print("  %-62s %s" % ("§3.2: Simes base wins every sparse config, both training sizes", "PASS" if sp_ok else "FAIL"))
ok = a["cpw_known_pmerge_f"][D] > a["cpw_known_pmerge"][D] and a["cpw_known_bonf"][D] > a["cpw_known_bonf_f"][D]
fails += (not ok)
print("  %-62s %s" % ("Fisher remark: dense known, only p-merge(Fisher) leads its Simes twin", "PASS" if ok else "FAIL"))
ok = bool(np.all(c["bonf_ns"][1:] < 0.05) and np.all(c["pmerge_ns"][1:] < 0.05) and c["bonf_ns"][0] > 0.05 and c["pmerge_ns"][0] > 0.05)
fails += (not ok)
print("  %-62s %s" % ("§6.1: bonf/p-merge naive size < alpha from Ntr=8n on, > alpha at 4n", "PASS" if ok else "FAIL"))
check("Fisher remark: |p-merge(Fisher) - Fisher| max (within 0.005)", max(mx1), 0.005)
check("Fisher remark: Bonferroni(Fisher) - Fisher max (at most +0.02)", max(mx2), 0.02, 2)

# ---------------------------------------------------------------- §8 FX
f = np.load("results_fx_application.npz", allow_pickle=True)
rows, keys = f["rows"], list(f["keys"])
col = lambda k: 2 + keys.index(k)
yr = lambda y: rows[rows[:, 0] == y][0]
print("§8 FX application (results_fx_application.npz)")
for y, q in [(2017, 0.06), (2021, 0.03), (2023, 0.06), (2024, 0.04)]:
    check("eavg rejection rate %d (calm)" % y, yr(y)[col("eavg")], q, 2)
for y, q in [(2020, 0.30), (2022, 0.49), (2025, 0.16), (2016, 0.14)]:
    check("eavg rejection rate %d (stress)" % y, yr(y)[col("eavg")], q, 2)
check("day-level gap eavg - single", float(f["gap_mean"]), 0.004, 3)
check("day-level gap NW standard error", float(f["gap_se"]), 0.004, 3)
ok = int(f["ebh_flag"].sum()) == 78
fails += (not ok)
print("  %-62s %d %s" % ("e-BH flags 78 days", int(f["ebh_flag"].sum()), "PASS" if ok else "FAIL"))
ok = yr(2020)[col("sym")] > max(yr(2020)[col("chi2")], yr(2020)[col("eavg")]) and \
     yr(2022)[col("chi2")] > max(yr(2022)[col("sym")], yr(2022)[col("eavg")])
fails += (not ok)
print("  %-62s %s" % ("sym leads 2020, chi2 leads 2022", "PASS" if ok else "FAIL"))
d20 = yr(2020)[col("sym")] - yr(2020)[col("eavg")]
d22 = yr(2022)[col("chi2")] - yr(2022)[col("eavg")]
ok = 0.005 <= min(d20, d22) and max(d20, d22) <= 0.045
fails += (not ok)
print("  %-62s %.3f, %.3f %s" % ("pooled tracks leader within 0.01-0.04", d20, d22, "PASS" if ok else "FAIL"))
E = np.exp(f["logE"]); dates = f["dates"]
top = np.argsort(-E)[:2]
ok = list(dates[top]) == ["2016-06-24", "2022-12-20"] and 1e19 < E[top[0]] < 1e21 and 1e17 < E[top[1]] < 1e18
fails += (not ok)
print("  %-62s %s %s" % ("largest e-values: Brexit ~1e20, BoJ 2022-12-20 ~5e17", list(dates[top]), "PASS" if ok else "FAIL"))
intex("$0.01$--$0.04$ of whichever is winning")
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
fc_ok = f["rows_fc"][:, col("eavg")].min() >= 0 and bool(np.all(f["rows_fc"][:, col("eavg")] >= f["rows"][:, col("eavg")] - 1e-12))
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

print("\nRESULT: %d failure(s)" % fails)
sys.exit(1 if fails else 0)
