import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

d = np.load("results_fx_application.npz", allow_pickle=True)
rows = d["rows"]; keys = list(d["keys"])
dates = [datetime.strptime(s, "%Y-%m-%d") for s in d["dates"]]
logE = d["logE"] / np.log(10)                     # log10 e-value
ebh = d["ebh_flag"].astype(bool)

plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 130})
fig, ax = plt.subplots(2, 1, figsize=(13, 8.6), height_ratios=[1.15, 1])
fig.suptitle("Validating a Gaussian FX risk model (9 USD rates, walk-forward yearly refit, "
             "M=12 orderings, bootstrap-calibrated)", fontsize=12.5, fontweight="bold", y=0.985)

A = ax[0]
A.plot(dates, logE, lw=0.6, color="#4575b4")
A.plot(np.array(dates)[ebh], logE[ebh], ".", ms=5, color="#d73027",
       label="flagged by e-BH (q=0.10), %d of %d days" % (ebh.sum(), len(ebh)))
A.axhline(np.log10(20), color="0.4", lw=1, ls="--")
A.text(dates[30], np.log10(20) + 0.4, "E = 1/α = 20", fontsize=8, color="0.35")
events = {"2016-06-24": "Brexit", "2020-03-20": "COVID", "2022-12-20": "BoJ YCC",
          "2022-06-16": "SNB/Fed", "2025-04-04": "tariffs"}
for s, lab in events.items():
    i = list(d["dates"]).index(s)
    A.annotate(lab, (dates[i], logE[i]), textcoords="offset points", xytext=(4, 4),
               fontsize=8.5, color="0.15")
A.set_title("A. Pooled e-value per day: spikes are recognizable FX events")
A.set_ylabel(r"$\log_{10}\bar{E}_M$")
A.legend(fontsize=8.5, loc="upper right")
A.xaxis.set_major_locator(mdates.YearLocator())
A.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

B = ax[1]
yy = rows[:, 0].astype(int); w = 0.15
sel = ["single", "eavg", "chi2", "sym"]
col = {"single": "#238b45", "eavg": "#08519c", "chi2": "#888888", "sym": "#b8860b"}
lab = {"single": "single random order", "eavg": "e-value avg (M=12)",
       "chi2": r"$\chi^2$ (order-inv.)", "sym": r"$\Sigma^{-1/2}$+Simes (order-inv.)"}
x = np.arange(len(yy))
for j, k in enumerate(sel):
    B.bar(x + (j - 1.5) * w, rows[:, 2 + keys.index(k)], w, color=col[k], label=lab[k])
B.axhline(0.05, color="red", lw=1.3, ls="--")
B.text(-0.45, 0.06, "α=0.05", color="red", fontsize=9)
B.set_xticks(x); B.set_xticklabels(yy)
B.set_title("B. Calibrated per-day rejection rate by year: ≈ nominal in calm years, "
            "large in stress years (2016, 2020, 2022, 2025)")
B.set_ylabel("fraction of days rejected"); B.legend(fontsize=8.5, loc="upper left")

fig.tight_layout(rect=[0, 0, 1, 0.965])
out = "fx_application.png"
fig.savefig(out, bbox_inches="tight"); print("saved", out)
