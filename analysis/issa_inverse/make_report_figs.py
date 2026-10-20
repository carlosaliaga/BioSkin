"""Thickness-cap sensitivity refits + report figures, from `out/` written by inverse_pilot.py.

    venv/Scripts/python.exe -I analysis/issa_inverse/make_report_figs.py
"""

import json, os, sys

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inverse_pilot import (Decoder, fit, normalized_to_physical, evaluate, CLASSES, NAMES,
                           WL_ISSA, UPSTREAM)

OUT = os.path.join(HERE, "out")
# reference palette (dataviz skill): sequential blue for ordered MI class, categorical slots in order
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False, "lines.linewidth": 2})

df = pd.read_csv(os.path.join(OUT, "pilot_fits.csv"))
A = np.load(os.path.join(OUT, "pilot_arrays.npz"))
S = json.load(open(os.path.join(OUT, "summary.json")))
orig = np.load(os.path.join(OUT, "orig_phys.npy"))
lo, hi, tgt = A["lo"], A["hi"], A["target"]
cls = df.mi_class.values
w = df.weight.values


def wq(x, q):
    s = np.argsort(x); c = np.cumsum(w[s]) / w.sum()
    return np.interp(q, c, x[s])


# ---------------------------------------------------------------- thickness-cap refits
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dec = Decoder(dev)
with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"), allow_pickle=False) as z:
    ref = dict(z)
s1_thr = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))["stage1_threshold"]
caps = {}
for cap_um in (250, 100):
    h = hi.copy(); h[2] = (cap_um * 1e-4 - 0.001) / 0.034
    u, rm = fit(dec, tgt, lo, h, 32, 1500, seed=S["seed"])
    ub = u[np.arange(len(u)), rm.argmin(1)]
    ph = normalized_to_physical(lo + ub * (h - lo))
    with torch.no_grad():
        pr = dec.pct31(torch.tensor(lo + ub * (h - lo), dtype=torch.float32, device=dev)).cpu().numpy()
    e = evaluate(pr, ref, s1_thr)
    caps[cap_um] = {"rmse": rm.min(1), "phys": ph, "pass": e["final"]}
    df[f"cap{cap_um}_rmse"] = rm.min(1)
    df[f"cap{cap_um}_melanin"] = ph[:, 0]
    df[f"cap{cap_um}_thickness_um"] = ph[:, 2]
    df[f"cap{cap_um}_haemoglobin"] = ph[:, 1]
    df[f"cap{cap_um}_at_cap"] = ub[:, 2] > 1 - 1e-3
df.to_csv(os.path.join(OUT, "pilot_fits_with_caps.csv"), index=False)

# ---------------------------------------------------------------- numbers for the report
g = df.groupby("mi_class")
rows = []
for c in CLASSES:
    d = df[df.mi_class == c]
    rows.append({"class": c, "n": len(d), "MI range": f"{d.MI.min():.0f}-{d.MI.max():.0f}",
                 "RMSE raw med": d.rmse_fit_raw.median(), "p90": d.rmse_fit_raw.quantile(.9),
                 "max": d.rmse_fit_raw.max(), "RMSE smoothed med": d.rmse_fit_smoothed.median(),
                 "RMSE encoder med": d.rmse_encoder.median(), "RMSE cap250 med": d.cap250_rmse.median(),
                 "RMSE cap100 med": d.cap100_rmse.median(),
                 "rel RMSE %": 100 * (d.rmse_fit_raw / tgt[df.mi_class == c].mean(1)).median(),
                 "mel med": d.melanin.median(), "mel cap100 med": d.cap100_melanin.median(),
                 "thick med": d.thickness_um.median(), "hb med": d.haemoglobin.median(),
                 "at thick cap100 %": 100 * d.cap100_at_cap.mean(),
                 "basins med": d.n_basins.median(), "pass obs": int(d.issa_observed_pass.sum()),
                 "pass fit raw": int(d.fit_raw_pass.sum()), "pass fit sm": int(d.fit_smoothed_pass.sum()),
                 "pass cap100": int(caps[100]["pass"][df.mi_class == c].sum())})
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(OUT, "by_class.csv"), index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
print(tab.round(3).to_string(index=False))

extra = {}
for n in NAMES:
    extra[n] = {"issa_weighted_p05_p50_p95": [float(wq(df[n].values, q)) for q in (.05, .5, .95)]}
for j, n in enumerate(NAMES):
    lo_hit = np.isclose(df[n], S["domain_physical"][n][0], rtol=0, atol=1e-3 * (S["domain_physical"][n][1] - S["domain_physical"][n][0]))
    hi_hit = np.isclose(df[n], S["domain_physical"][n][1], rtol=0, atol=1e-3 * (S["domain_physical"][n][1] - S["domain_physical"][n][0]))
    extra[n].update({"at_lower": int(lo_hit.sum()), "at_upper": int(hi_hit.sum()),
                     "orig_sample_p50": float(np.median(orig[:, j]))})
p95m, p95h = extra["melanin"]["issa_weighted_p05_p50_p95"][2], extra["haemoglobin"]["issa_weighted_p05_p50_p95"][2]
extra["orig_frac_melanin_above_issa_p95"] = float((orig[:, 0] > p95m).mean())
extra["orig_frac_melanin_above_pilot_max"] = float((orig[:, 0] > df.melanin.max()).mean())
extra["orig_frac_hb_above_issa_p95"] = float((orig[:, 1] > p95h).mean())
extra["orig_frac_both_inside_issa_p95"] = float(((orig[:, 0] <= p95m) & (orig[:, 1] <= p95h)).mean())
extra["weighted_thick_le_100"] = float((w * (df.thickness_um <= 100)).sum() / w.sum())
extra["weighted_rmse_median"] = float(wq(df.rmse_fit_raw.values, .5))
extra["weighted_rmse_cap100_median"] = float(wq(df.cap100_rmse.values, .5))
extra["corr_log_mel_thick"] = float(np.corrcoef(np.log(df.melanin), df.thickness_um)[0, 1])
extra["pilot_rmse_median"] = float(df.rmse_fit_raw.median())
extra["weighted_cap100_melanin_p50"] = float(wq(df.cap100_melanin.values, .5))
extra["cap_pass_total"] = {k: int(v["pass"].sum()) for k, v in caps.items()}
json.dump(extra, open(os.path.join(OUT, "extra.json"), "w"), indent=1)
print(json.dumps(extra, indent=1))

# ---------------------------------------------------------------- fig 1: reconstruction by class
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
rng = np.random.default_rng(0)
for i, c in enumerate(CLASSES):
    v = df.rmse_fit_raw[df.mi_class == c].values
    ax[0].scatter(i + rng.uniform(-.18, .18, len(v)), v, s=10, color=SEQ[i], alpha=.75, lw=0)
    ax[0].plot([i - .28, i + .28], [np.median(v)] * 2, color=INK, lw=2, solid_capstyle="round")
    ax[0].text(i, -0.22, f"n={len(v)}", ha="center", va="top", fontsize=8, color=INK2)
ax[0].axhline(1.428, color=INK2, ls="--", lw=1)
ax[0].text(4.45, 1.428, "BVCM population\nlevel (test) 1.43", va="center", ha="left", fontsize=7.5, color=INK2)
ax[0].axhline(0.198, color=INK2, ls=":", lw=1)
ax[0].text(4.45, 0.198, "BVCM + per-spectrum\ndeviation 0.20", va="center", ha="left", fontsize=7.5, color=INK2)
ax[0].set_xticks(range(5), CLASSES); ax[0].set_xlim(-.5, 4.4); ax[0].set_ylim(0, 2.3)
ax[0].set_ylabel("best-fit RMSE (reflectance pp)")
ax[0].set_title("a  Reconstruction RMSE by MI class", loc="left", fontsize=10, color=INK)
pred = A["pred_raw"]
for i, c in enumerate(CLASSES):
    m = df.mi_class.values == c
    ax[1].plot(WL_ISSA, (pred[m] - tgt[m]).mean(0), color=SEQ[i], label=c)
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].set_xlabel("wavelength (nm)"); ax[1].set_ylabel("mean residual, fit - ISSA (pp)")
ax[1].set_title("b  Where the misfit sits", loc="left", fontsize=10, color=INK)
ax[1].set_ylim(-3.4, 2.6); ax[1].legend(title="MI class", fontsize=8, title_fontsize=8, ncol=5, loc="lower center")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_reconstruction_by_class.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 2: example fits
fig, ax = plt.subplots(1, 5, figsize=(12, 2.9), sharey=False)
for i, c in enumerate(CLASSES):
    idx = np.flatnonzero(df.mi_class.values == c)
    r = df.rmse_fit_raw.values[idx]
    for k, (lab, j) in enumerate([("median", idx[np.argsort(r)[len(r) // 2]]), ("worst", idx[r.argmax()])]):
        col = C1 if k == 0 else C2
        ax[i].plot(WL_ISSA, tgt[j], "o", ms=3.2, color=col, mfc="none", mew=1)
        ax[i].plot(WL_ISSA, pred[j], color=col, lw=1.6, label=f"{lab} ({df.rmse_fit_raw[j]:.2f} pp)")
    ax[i].set_title(c, loc="left", fontsize=10, color=INK)
    ax[i].legend(fontsize=7, loc="upper left"); ax[i].set_xlabel("nm")
ax[0].set_ylabel("reflectance (%)")
fig.text(0.5, -0.02, "circles = ISSA observed; lines = best BioSkin decoder fit", ha="center", fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_example_fits.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: parameter histograms
labels = ["melanin volume fraction", "blood volume fraction", "epidermal thickness (µm)",
          "melanin type ratio", "blood oxygenation"]
fig, ax = plt.subplots(1, 5, figsize=(13, 3.0))
for j, n in enumerate(NAMES):
    a = ax[j]; dlo, dhi = S["domain_physical"][n]
    log = j in (0, 1)
    bins = np.geomspace(dlo, dhi, 30) if log else np.linspace(dlo, dhi, 30)
    a.hist(orig[:, j], bins=bins, density=True, color=C2, alpha=.45, label="original 4,000 sample")
    a.hist(df[n], bins=bins, weights=w, density=True, histtype="step", color=C1, lw=2,
           label="pilot fits (ISSA-weighted)")
    a.axvspan(dlo, dhi, color=GRID, alpha=.35, zorder=-1, lw=0)
    if log:
        a.set_xscale("log")
    if j == 2:
        a.axvline(100, color=INK, ls="--", lw=1); a.text(102, a.get_ylim()[1] * .92, "facial lit.\n≤100 µm", fontsize=7, color=INK)
    a.set_title(labels[j], fontsize=9, color=INK, loc="left"); a.set_yticks([])
    a.set_xlim(dlo, dhi)
ax[0].legend(fontsize=7, loc="upper left")
fig.text(0.5, -0.03, "x-range of each panel = decoder training domain; melanin and blood on log axes", ha="center",
         fontsize=8, color=INK2)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_parameter_histograms.png"), dpi=160, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: melanin offset / thickness trade-off
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].scatter(df.MI, df.melanin, s=12, color=C1, lw=0, alpha=.8, label="free thickness (10-350 µm)")
ax[0].scatter(df.MI, df.cap100_melanin, s=12, color=C2, lw=0, alpha=.8, label="thickness ≤ 100 µm")
ax[0].set_yscale("log"); ax[0].set_xlabel("ISSA melanin index"); ax[0].set_ylabel("fitted melanin volume fraction")
ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_title("a  Melanin vs MI, two thickness constraints", loc="left", fontsize=10, color=INK)
sc = ax[1].scatter(df.melanin, df.thickness_um, c=df.MI, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", SEQ), s=14, lw=0)
ax[1].set_xscale("log"); ax[1].set_xticks([0.02, 0.05, 0.1, 0.2], ["0.02", "0.05", "0.1", "0.2"]); ax[1].minorticks_off(); ax[1].set_xlabel("fitted melanin volume fraction"); ax[1].set_ylabel("fitted thickness (µm)")
ax[1].axhline(100, color=INK, ls="--", lw=1)
plt.colorbar(sc, ax=ax[1], label="MI")
ax[1].set_title("b  Thickness rises with MI in the free fit", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_melanin_thickness.png"), dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 5: identifiability
fig, ax = plt.subplots(figsize=(6.4, 3.2))
for i, c in enumerate(CLASSES):
    v = [df[f"halfwidth05_unit_{n}"][df.mi_class == c].median() for n in NAMES]
    ax.plot(np.arange(5) + (i - 2) * .12, v, "o", ms=6, color=SEQ[i], label=c)
ax.set_xticks(range(5), ["melanin", "blood", "thickness", "mel. type\nratio", "oxygen-\nation"])
ax.set_ylim(0, 1.05); ax.set_ylabel("± range at 0.5 pp RMSE change\n(fraction of training domain)")
ax.legend(title="MI class", fontsize=7, title_fontsize=7, ncol=5, loc="upper left")
ax.set_title("Local identifiability at the best fit (median)", loc="left", fontsize=10, color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_identifiability.png"), dpi=160); plt.close(fig)
print("figures written")
