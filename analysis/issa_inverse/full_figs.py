"""Figures for the full-run addendum (reads full_out/)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "legend.frameon": False})
OUT = os.path.join(HERE, "full_out"); CL = ["EML", "EMIL", "EMI", "EMIH", "EH"]
real = pd.read_csv(os.path.join(OUT, "real_issa_validation_by_class.csv"))
r = pd.read_csv(os.path.join(OUT, "full_fits.csv")); sc = pd.read_csv(os.path.join(OUT, "offset_scan.csv"))

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
x = np.arange(5)
lohi = np.array([[float(v) for v in s.split("-")] for s in real["both % CI (participant bootstrap)"]])
ax[0].bar(x - .2, real["both %"], .38, color=C2, label="real ISSA spectra")
ax[0].bar(x + .2, [100 * r.primary_smoothed_both[r.mi_class == c].mean() for c in CL], .38, color=C1,
          label="BioSkin fit (primary)")
ax[0].errorbar(x - .2, real["both %"], yerr=[real["both %"] - lohi[:, 0], lohi[:, 1] - real["both %"]],
               fmt="none", ecolor=INK, lw=1, capsize=2)
ax[0].text(4, 3, "0 / 0", ha="center", fontsize=8, color=INK)
ax[0].set_xticks(x, [f"{c}\nn={n}" for c, n in zip(CL, real.spectra)]); ax[0].set_ylim(0, 122); ax[0].set_yticks(range(0, 101, 20))
ax[0].set_ylabel("pass both stages (%)")
ax[0].legend(fontsize=7.5, loc="upper left", ncol=2); ax[0].set_title("a  Their test on real ISSA and on fits", loc="left", fontsize=10)
for i, c in enumerate(CL):
    v = r.primary_smoothed_rmse[r.mi_class == c].values
    ax[1].boxplot(v, positions=[i], widths=.5, showfliers=False, patch_artist=True,
                  boxprops=dict(facecolor=SEQ[i], color=SEQ[i]), medianprops=dict(color=INK, lw=2),
                  whiskerprops=dict(color=SEQ[i]), capprops=dict(color=SEQ[i]))
ax[1].set_xticks(x, CL); ax[1].set_ylabel("best-fit RMSE (pp)")
ax[1].set_title("b  Reconstruction, all 7,923 (whiskers 1.5 IQR)", loc="left", fontsize=10)
a = sc[sc["class"] != "all"]
for i, c in enumerate(CL):
    s = a[a["class"] == c]
    ax[2].plot(s.melanin, s.thick, "-o", ms=4, color=SEQ[i], label=c)
    ax[2].annotate("0", (s.melanin.iloc[0], s.thick.iloc[0]), fontsize=7, color=INK2, xytext=(3, 3), textcoords="offset points")
ax[2].axhline(100, color=INK, ls="--", lw=1); ax[2].set_xscale("log")
ax[2].set_xticks([0.02, 0.05, 0.1, 0.2, 0.4], ["0.02", "0.05", "0.1", "0.2", "0.4"]); ax[2].minorticks_off()
ax[2].set_xlabel("median fitted melanin"); ax[2].set_ylabel("median fitted thickness (µm)")
ax[2].legend(fontsize=7, ncol=2, title="global offset 0→5 pp", title_fontsize=7)
ax[2].set_title("c  A specular floor trades thickness for melanin", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig6_full_run.png"), dpi=160); plt.close(fig)
print("ok")
