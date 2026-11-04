"""Shared style for all paper figures (matched to the FX application figure)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# method palette (consistent across every figure)
C = dict(single="#238b45", pmerge="#d73027", bonf="#7f0000", eavg="#08519c", e1="#6baed6",
         chi2="#888888", sym="#b8860b", fisher="#8e44ad", fisherpool="#5b2c6f",
         one="#d73027", two="#08519c", accent="#4575b4", light="#74add1",
         neutral="#555555", med="#969696")

LBL = dict(single="single random order", pmerge="p-merge", bonf="Bonferroni/orders",
           eavg="e-value avg", e1="e-value (single order)", chi2=r"$\chi^2$ (order-inv.)",
           sym=r"$\Sigma^{-1/2}$+Simes (order-inv.)", fisher="Fisher (single order)",
           fisherpool="Fisher-pool")


def setup():
    plt.rcParams.update({
        "font.size": 10.5, "axes.titlesize": 10.5, "axes.titleweight": "normal",
        "axes.labelsize": 10.5, "legend.fontsize": 8.5,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150, "savefig.dpi": 150,
        "axes.edgecolor": "0.25", "xtick.color": "0.25", "ytick.color": "0.25",
        "text.color": "0.1", "axes.labelcolor": "0.1",
    })
    return plt


# ---- training-size axis for Figures 2 and 4 -------------------------------------------------
import numpy as np


def with_extras(base, extras):
    """Merge the archive of the original training sizes with any added ones, sorted by N_tr."""
    d = dict(np.load(base))
    for extra in extras:
        try:
            e = np.load(extra)
        except FileNotFoundError:
            continue
        m = len(d["Ntr"]); o = np.argsort(np.concatenate([d["Ntr"], e["Ntr"]]))
        for k in list(d):
            if k in e.files and np.ndim(d[k]) == 1 and len(d[k]) == m:
                d[k] = np.concatenate([d[k], e[k]])[o]
    return d


def log_ntr_axis(d, gap=0.8):
    """x = log2(N_tr / n) for the estimated nulls; known F (archived as N_tr = 100000) is placed
    `gap` to the right of the largest training size, after an axis break."""
    n = int(d["n"]); fin = np.asarray(d["Ntr"]) < 100000
    x = np.empty(len(fin)); x[fin] = np.log2(np.asarray(d["Ntr"])[fin] / n)
    xk = x[fin].max() + gap; x[~fin] = xk
    return x, fin, xk


def draw_series(ax, x, fin, y, ls="-", **kw):
    """Line over the estimated-null points, dotted link across the break to the known-F point."""
    y = np.asarray(y)
    ax.plot(x[fin], y[fin], ls=ls, marker="o", **kw)
    kw.pop("label", None)
    ax.plot([x[fin][-1], x[~fin][0]], [y[fin][-1], y[~fin][0]], ls=":", **kw)
    ax.plot(x[~fin], y[~fin], ls="none", marker="o", **kw)


def finish_log_ntr_axis(ax, x, fin, xk):
    lo, hi = int(np.ceil(x[fin].min())), int(np.floor(x[fin].max()))
    ticks = list(range(lo, hi + 1))
    ax.set_xticks(ticks + [xk]); ax.set_xticklabels([str(t) for t in ticks] + ["known $F$"])
    ax.set_xlabel(r"$\log_2(N_{\mathrm{tr}}/n)$")
    xb = (x[fin].max() + xk) / 2                        # axis-break mark
    for dx in (-0.045, 0.045):
        ax.plot([xb + dx - 0.035, xb + dx + 0.035], [-0.018, 0.018], transform=ax.get_xaxis_transform(),
                color="0.3", lw=1.0, clip_on=False)
