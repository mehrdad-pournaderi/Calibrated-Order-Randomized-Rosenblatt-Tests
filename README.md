# Calibrated Order-Randomized Rosenblatt Tests

Replication code and data for the paper *Calibrated Order-Randomized Rosenblatt Tests*.

The Rosenblatt transformation reduces the hypothesis $H_0:\mathbf X\sim F$ to a test of uniformity,
but requires an arbitrary ordering of the coordinates, which under dependence materially affects
power. This repository implements **order randomization**: applying the transform under many random
orderings and combining the resulting evidence with dependence-robust merging rules, calibrated by a
re-estimating parametric bootstrap.

## Layout

```
code/       experiments (exp_*), supporting checks (check_*), figures (fig*), application (app_*)
data/       daily FRED H.10 exchange rates used by the application
results/    saved Monte-Carlo output, one .npz archive per experiment
figures/    figures as they appear in the paper
```

## Reproducing the results

Requires Python 3.9+ with `numpy`, `scipy` and `matplotlib`.

```bash
pip install -r requirements.txt
cd code
```

**Experiments** — each writes one `.npz` archive:

| Script | What it measures | Appears in |
|---|---|---|
| `exp_base_statistic.py` | one- versus two-sided base statistics | §4, §5, Figure 1 |
| `exp_calibration_pipeline.py` | naive vs calibrated size, then power, in one setting | Figure 2 |
| `exp_departure_shape.py` | sparse versus dense departures at equal energy | Figure 3 |
| `exp_multiplicity_screening.py` | FDR and power when hypotheses fail heterogeneously | Figure 4, §7.1 |
| `merge_multiplicity_chunks.py` | pools the chunks written above into one archive | Figure 4, §7.1 |
| `exp_training_size.py` | behaviour as the reference sample shrinks | §6.1 |
| `exp_number_of_orderings.py` | calibrated power versus the number of orderings $M$ | (background) |
| `exp_threshold_dependence.py` | why the combiner ranking inverts between a single test and a screen | §7.2 |
| `exp_sparsity_path.py` | power along the sparse-to-dense path | (background) |
| `exp_conformal_comparison.py` | comparison with conformal novelty detection and AdaDetect | Table 1, §7.3 |
| `merge_conformal_chunks.py` | pools the chunks written above into one archive per reference-sample size | Table 1, §7.3 |
| `exp_conformal_novelty_density.py` | how the crossover with AdaDetect moves with the null proportion | Table 2, §7.3 |

`exp_departure_shape.py` takes a mode argument: `alt` (sparse vs dense), `rho` (dependence sweep),
`n` (dimension sweep). It writes `results_shape_<mode>.npz`; the `alt` archive is stored in
`results/` as `results_shape_sparse_dense.npz`, the name `fig3_departure_shape.py` reads.

Three experiments run in seeded chunks and are pooled afterwards, because their realized false
discovery proportion is coarse when non-nulls are rare. `exp_multiplicity_screening.py` takes a
replication count and a seed; `exp_conformal_comparison.py` takes a reference-sample size, a
replication count and a seed; `exp_conformal_novelty_density.py` takes a null proportion (`80`,
`90` or `95`), a replication count and a seed. Each writes one `chunk_*.npz` per seed, which
`merge_multiplicity_chunks.py` and `merge_conformal_chunks.py` then pool into the archives in
`results/`. The chunks store raw per-replication proportions, so pooling is exact and the reported
Monte-Carlo standard errors are exact too. The two conformal scripts are the only ones that need
`scikit-learn`.

**Supporting checks** — small scripts behind claims made in the text:

| Script | Question it answers |
|---|---|
| `check_nominal_threshold_M.py` | does the Bonferroni dome survive the two-sided statistic? |
| `check_nominal_threshold_size.py` | how much level do the combiners waste before calibration? |
| `check_data_dependent_ordering.py` | may the ordering be chosen from the data? (no) |
| `check_predecessor_sets.py` | is the position of the departure a sufficient description? (no) |
| `check_fisher_pooling.py` | does pooling improve the Fisher base statistic? (no) |
| `check_evidence_power_bridge.py` | when does expected log-evidence control fixed-level power? |
| `check_equicorrelated_orbit.py` | the exactly solvable orbit: closed-form profiles and exact power (background) |
| `check_quoted_numbers.py` | re-verifies every number the paper quotes from the single-test and FX archives (run from `results/`, optionally with the path to `paper.tex`) |

**Application and figures:**

```bash
python app_fx_risk_model.py       # foreign-exchange validation -> Figure 5 (reads fx_DEX*.csv from the working directory; copy data/*.csv into code/ first)
python fig1_two_sided_blindspot.py
python fig2_calibration_pipeline.py
python fig3_departure_shape.py
python fig4_multiplicity_screening.py
python fig5_fx_application.py
python fig_orbit_exact.py         # exact-orbit figure (background; not in the paper)
```

Figure scripts read the `.npz` archives; copy `results/*.npz` into `code/` (or run from `results/`)
to rebuild the figures without re-running the simulations. Every script fixes an integer seed and
avoids process-dependent randomness, so reruns reproduce the published numbers exactly — the five
figures regenerate byte-for-byte from the archives in `results/`.

## Two implementation details worth knowing

- **The orderings are redrawn in every Monte-Carlo realization.** The single-ordering baseline is
  therefore the *expected* power of an arbitrarily chosen ordering. Freezing one permutation across
  realizations makes the baseline a lottery ticket and can overstate the gain from pooling several
  fold.
- **The e-value path carries its own single-ordering baseline.** `exp_calibration_pipeline.py` and
  `exp_departure_shape.py` report `e1`, the calibrated mixture e-value of one random ordering, next
  to the e-value average; the paired gap `eavg − e1` (keys `g1e_*`) isolates the effect of
  aggregation from the choice of base statistic, whereas `eavg − single` also changes the base
  statistic (Simes p-value to mixture e-value). In the FX application the bootstrap re-estimates the
  mean as well as the covariance inside each replicate, and the day-level gap standard error is
  Newey–West, and two calibrations are stored side by side: the re-estimating bootstrap for the
  estimated-population null (`rows`, `gap_*`) and exact simulation from the issued forecast with
  fixed parameters (`rows_fc`, `gap_fc_*`), each on its own random stream.
- **Calibration acts on uncapped merger summaries.** The single-test scripts and the application
  hand Procedure 1 the uncapped `mean_m P_m` and `M·min_m P_m` (the capped nominal p-values put an
  atom at 1 that a rank rule cannot break; at M=12 this never binds, but at large M under weak
  dependence it does — see the note in `exp_number_of_orderings.py`).
- **In the multiplicity layer, the bootstrap replicate count should satisfy $B > N/q$.** A bootstrap
  $p$-value cannot fall below $1/(B+1)$, while Benjamini–Hochberg's smallest threshold is $q/N$;
  with too small a $B$ the attainable power of every method is capped by calibration granularity
  rather than by the test (the condition is sufficient for full BH resolution, not necessary for
  making some rejections). All calibrated decisions use rank-based Monte-Carlo p-values,
  (1 + #{T*_b at least as extreme as T_obs})/(B+1), never interpolated quantiles, which are
  anti-conservative at finite B (expected rejection ((B-1)α+1)/(B+1), i.e. 0.0545 at B=199 and
  α=0.05). The single-test experiments (`exp_calibration_pipeline.py`, `exp_departure_shape.py`,
  `exp_training_size.py`, `exp_number_of_orderings.py`, `exp_sparsity_path.py`,
  `app_fx_risk_model.py`) were rerun under the rank rule on 19 September 2026; the archives in
  `results/` and the figures are from those runs. The same constraint binds on the conformal calibration set in
  `exp_conformal_comparison.py`, which is why that script uses larger reference samples than the
  rest: a conformal $p$-value is a multiple of $1/(\ell+1)$, so too small an $\ell$ makes every
  conformal method reject nothing at all.

## Data

`data/` holds nine daily exchange-rate series (euro, yen, sterling, Canadian dollar, Swiss franc,
Australian and New Zealand dollars, Swedish and Norwegian kronor) against the U.S. dollar, from the
Federal Reserve's H.10 release via [FRED](https://fred.stlouisfed.org), retrieved 15 July 2026 and
covering 2015–2025. `code/fetch_fx_data.sh` re-downloads them.

The nine series and their quotation conventions:

| Series | Currency | Quotation |
|---|---|---|
| `DEXUSEU` | Euro | USD per unit |
| `DEXJPUS` | Japanese yen | units per USD |
| `DEXUSUK` | British pound | USD per unit |
| `DEXCAUS` | Canadian dollar | units per USD |
| `DEXSZUS` | Swiss franc | units per USD |
| `DEXUSAL` | Australian dollar | USD per unit |
| `DEXUSNZ` | New Zealand dollar | USD per unit |
| `DEXSDUS` | Swedish krona | units per USD |
| `DEXNOUS` | Norwegian krone | units per USD |

All series are converted to USD per foreign unit before differencing, so a positive return always
means dollar depreciation. Days on which any series is unquoted are dropped, leaving 2,747 daily
log-return vectors spanning 5 January 2015 to 31 December 2025. The model is refitted at the start
of each calendar year on the preceding year's returns, and every day of the new year is tested out
of sample. The data are public-domain U.S. government works.

## Paper

The manuscript is not included in this repository. Section and figure numbers referenced above
follow the published version.

## License

The code is released under the MIT License (see `LICENSE`). The exchange-rate data are U.S.
government works and are not subject to copyright.
