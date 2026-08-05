"""Shared style for all paper figures (matched to the FX application figure)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# method palette (consistent across every figure)
C = dict(single="#238b45", pmerge="#d73027", bonf="#7f0000", eavg="#08519c",
         chi2="#888888", sym="#b8860b", fisher="#8e44ad", fisherpool="#5b2c6f",
         one="#d73027", two="#08519c", accent="#4575b4", light="#74add1",
         neutral="#555555", med="#969696")

LBL = dict(single="single random order", pmerge="p-merge", bonf="Bonferroni/orders",
           eavg="e-value avg", chi2=r"$\chi^2$ (order-inv.)",
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
