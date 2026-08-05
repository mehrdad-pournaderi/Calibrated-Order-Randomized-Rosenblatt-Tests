# Calibrated Order-Randomized Rosenblatt Tests

Replication code, data and manuscript for the paper *Calibrated Order-Randomized Rosenblatt Tests*.

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
paper/      LaTeX sources (elsarticle), bibliography and compiled PDFs
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
| `exp_multiplicity_screening.py` | FDR and power when hypotheses fail heterogeneously | Figure 4, §7 |
| `exp_training_size.py` | behaviour as the reference sample shrinks | §6.1 |
| `exp_number_of_orderings.py` | calibrated power versus the number of orderings $M$ | §6.3 |
| `exp_sparsity_path.py` | power along the sparse-to-dense path | (background) |

`exp_departure_shape.py` takes a mode argument: `alt` (sparse vs dense), `rho` (dependence sweep),
`n` (dimension sweep).

**Supporting checks** — small scripts behind claims made in the text:

| Script | Question it answers |
|---|---|
| `check_nominal_threshold_M.py` | does the Bonferroni dome survive the two-sided statistic? |
| `check_nominal_threshold_size.py` | how much level do the combiners waste before calibration? |
| `check_data_dependent_ordering.py` | may the ordering be chosen from the data? (no) |
| `check_predecessor_sets.py` | is the position of the departure a sufficient description? (no) |
| `check_fisher_pooling.py` | does pooling improve the Fisher base statistic? (no) |

**Application and figures:**

```bash
python app_fx_risk_model.py       # foreign-exchange validation -> Figure 5
python fig1_two_sided_blindspot.py
python fig2_calibration_pipeline.py
python fig3_departure_shape.py
python fig4_multiplicity_screening.py
python fig5_fx_application.py
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
- **In the multiplicity layer, the bootstrap replicate count must satisfy $B > N/q$.** A bootstrap
  $p$-value cannot fall below $1/(B+1)$, while Benjamini–Hochberg needs values as small as $q/N$;
  with too small a $B$ the attainable power of every method is capped by calibration granularity
  rather than by the test.

## Data

`data/` holds nine daily exchange-rate series (euro, yen, sterling, Canadian dollar, Swiss franc,
Australian and New Zealand dollars, Swedish and Norwegian kronor) against the U.S. dollar, from the
Federal Reserve's H.10 release via [FRED](https://fred.stlouisfed.org), retrieved 15 July 2026 and
covering 2015–2025. `code/fetch_fx_data.sh` re-downloads them. Series identifiers and quotation
conventions are listed in `paper/supplementary.tex`. The data are public-domain U.S. government
works.

## Building the paper

```bash
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

`paper_anon.tex` is the double-anonymized version prepared for review; `title_page.tex` carries the
author details, acknowledgments and declarations that the anonymized manuscript omits.

## License

Code and manuscript are released under the MIT License (see `LICENSE`). The exchange-rate data are
U.S. government works and are not subject to copyright.
