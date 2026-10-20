"""Every number and figure in the proposed Section 4.3, in one reproducible run.

Conventions (decided 2026-10-20):
  - model compared to ISSA: R_SCI = F + (1 - F) * R_d(decoder), F = 0.028 (ISSA is d/8 SCI; BioSkin R_d is
    subsurface only); F = 0 reported only as the published comparison
  - double Savitzky-Golay (15, 3) smoothed decoder output primary, raw disclosed
  - inverse fits: full training domain; melanin type ratio 0.725 and oxygenation 0.95 fixed
  - their two-stage evaluate() + reference_state.npz unchanged; summaries clustered by participant
Nothing derived from the non-shareable Monte Carlo spectra is reported here; only the decoder, ISSA and
the training-domain bounds are used.

    venv/Scripts/python.exe -I analysis/issa_inverse/paper_numbers.py
"""
import json, os, sys

import numpy as np, pandas as pd, torch
from scipy.stats import qmc, gaussian_kde, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from full_run import fit_general, load_meta, cluster_boot, FIX, K, STEPS, SEED, CLASSES
from inverse_pilot import (Decoder, training_box, normalized_to_physical, near_optimal_spread, evaluate,
                           load_issa_face, mi_class, UPSTREAM, ROOT, WL_ISSA)

F = 0.028
OUT = os.path.join(HERE, "paper"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"), allow_pickle=False) as z:
    ref = dict(z)
s1 = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))["stage1_threshold"]
d = load_issa_face(); meta, chk = load_meta(); assert np.array_equal(chk, d["spectra"])
issa, mi, cls, pid = d["spectra"], d["mi"], mi_class(d["mi"]), meta.participant.values
dec = Decoder(dev); lo, hi, _ = training_box(dev)
fixed = {j: (v - lo[j]) / (hi[j] - lo[j]) for j, v in FIX.items()}
N = {}


def sci(p, f=F):
    return 100 * f + (1 - f) * p


def pca_cover(S):
    sc = (S - ref["pca_mean"]) @ ref["pca_components"].T - ref["pca_score_mean"]
    m2 = np.einsum("ij,jk,ik->i", sc, np.linalg.inv(ref["pca_score_cov"]), sc)
    return 100 * float((m2 <= chi2.ppf(.95, 2)).mean()), sc + ref["pca_score_mean"]


def classtab(values, stat=np.median):
    return {c: float(stat(values[cls == c])) for c in CLASSES} | {"all": float(stat(values))}


# ---------------------------------------------------------------- A. real ISSA under their test
e = evaluate(issa, ref, s1)
N["real"] = {c: {"n": int((cls == c).sum()), "participants": int(pd.unique(pid[cls == c]).size),
                 "stage1": 100 * float(e["stage1"][cls == c].mean()), "both": 100 * float(e["final"][cls == c].mean()),
                 "both_ci": (100 * cluster_boot(e["final"][cls == c].astype(float), pid[cls == c])).tolist()}
             for c in CLASSES}
N["real_all_both"] = 100 * float(e["final"].mean())
N["n_participants"] = int(pd.unique(pid).size)

# ---------------------------------------------------------------- B. inverse fits
def run(free, smooth, h=hi, f=F, k=K):
    tgt = (issa - 100 * f) / (1 - f)
    r = fit_general(dec, tgt, lo, h, free, fixed if len(free) == 3 else {}, k, STEPS, smooth, SEED)
    pred = sci(r["pred"], f)
    r["rmse_true"] = np.sqrt(((pred - issa) ** 2).mean(1))
    r["ev"] = evaluate(np.clip(pred, 1e-3, 99.9), ref, s1)
    r["phys"] = normalized_to_physical(lo + r["u"] * (h - lo))
    r["sci_pred"] = pred
    return r

cap100 = hi.copy(); cap100[2] = (100e-4 - 0.001) / 0.034
P = run([0, 1, 2], True)
variants = {"primary": P, "raw": run([0, 1, 2], False), "free5": run([0, 1, 2, 3, 4], True),
            "thick_le_100um": run([0, 1, 2], True, cap100), "no_specular": run([0, 1, 2], True, f=0.0)}
N["inverse_overall"] = {k: {"rmse_median": float(np.median(v["rmse_true"])),
                            "pass_both": 100 * float(v["ev"]["final"].mean()),
                            "thickness_median": float(np.median(v["phys"][:, 2]))} for k, v in variants.items()}
ph, u = P["phys"], P["u"]
spread, n_near, basins = near_optimal_spread(P["u_all"], P["rmse_all"])
part_med = pd.Series(P["rmse_true"]).groupby([cls, pid]).median()
N["inverse_by_class"] = {
    "rmse_median": classtab(P["rmse_true"]),
    "rmse_q25": classtab(P["rmse_true"], lambda x: np.percentile(x, 25)),
    "rmse_q75": classtab(P["rmse_true"], lambda x: np.percentile(x, 75)),
    "rmse_participant_median": {c: float(part_med[c].median()) for c in CLASSES},
    "rmse_rel_pct": classtab(100 * P["rmse_true"] / issa.mean(1)),
    "melanin_median": classtab(ph[:, 0]), "melanin_q25": classtab(ph[:, 0], lambda x: np.percentile(x, 25)),
    "melanin_q75": classtab(ph[:, 0], lambda x: np.percentile(x, 75)),
    "blood_median": classtab(ph[:, 1]), "thickness_median": classtab(ph[:, 2]),
    "thickness_q25": classtab(ph[:, 2], lambda x: np.percentile(x, 25)),
    "thickness_q75": classtab(ph[:, 2], lambda x: np.percentile(x, 75)),
    "thick_le_100_pct": classtab(100 * (ph[:, 2] <= 100).astype(float), np.mean),
    "melanin_at_ceiling_pct": classtab(100 * (u[:, 0] > 1 - 1e-3).astype(float), np.mean),
    "fit_pass_both": classtab(100 * P["ev"]["final"].astype(float), np.mean),
    "raw_fit_pass_both": classtab(100 * variants["raw"]["ev"]["final"].astype(float), np.mean),
    "basins_mean": classtab(basins.astype(float), np.mean),
    "mi_abs_err": classtab(np.abs(P["ev"]["mi"] - mi)), "ei_abs_err": classtab(np.abs(P["ev"]["ei"] - d["ei"])),
}
N["inverse_overall_extra"] = {
    "melanin_p05_p50_p95": np.percentile(ph[:, 0], [5, 50, 95]).tolist(),
    "blood_p05_p50_p95": np.percentile(ph[:, 1], [5, 50, 95]).tolist(),
    "thickness_p05_p50_p95": np.percentile(ph[:, 2], [5, 50, 95]).tolist(),
    "thick_le_100_pct": 100 * float((ph[:, 2] <= 100).mean()),
    "spectra_below_F": int((issa < 100 * F).any(1).sum()),
    "free5_ratio_at_bound_pct": 100 * float(((variants["free5"]["u"][:, 3] < 1e-3) | (variants["free5"]["u"][:, 3] > 1 - 1e-3)).mean()),
    "free5_oxy_at_bound_pct": 100 * float(((variants["free5"]["u"][:, 4] < 1e-3) | (variants["free5"]["u"][:, 4] > 1 - 1e-3)).mean()),
    "rmse_median_ci": cluster_boot(P["rmse_true"], pid, np.median, B=500).tolist(),
}

# ---------------------------------------------------------------- C. forward sets under their test
cp = pd.read_csv(os.path.join(ROOT, "analysis/corrected/corrected_skin_props_network_ready.csv"),
                 header=None).to_numpy(np.float32)[:, :5]
orig_phys = normalized_to_physical(cp)
U = qmc.Sobol(5, scramble=True, seed=SEED).random(4096)
kde = gaussian_kde(u[:, :3].T)
Ue = np.clip(kde.resample(4000, seed=SEED).T, 0, 1)
Xe = lo + np.column_stack([Ue, np.full((4000, 1), fixed[3]), np.full((4000, 1), fixed[4])]) * (hi - lo)
sets = {"original": cp, "sobol": (lo + U * (hi - lo)).astype(np.float32), "empirical": Xe.astype(np.float32)}
N["forward"], spectra_for_pca = {}, {}
for name, X in sets.items():
    with torch.no_grad():
        xt = torch.tensor(X, device=dev)
        sm, rw = dec.pct31(xt, True).cpu().numpy(), dec.pct31(xt, False).cpu().numpy()
    out = {}
    for lab, S in [("smoothed_F0", sm), ("smoothed_F", sci(sm)), ("raw_F0", rw), ("raw_F", sci(rw))]:
        ev = evaluate(S, ref, s1)
        out[lab] = {"stage1": 100 * float(ev["stage1"].mean()), "both": 100 * float(ev["final"].mean()),
                    "n_both": int(ev["final"].sum()), "n": len(S)}
    N["forward"][name] = out
    spectra_for_pca[name] = sci(sm)
    if name == "original":
        spectra_for_pca["original_F0"] = sm
p95m, p95b = np.percentile(ph[:, 0], 95), np.percentile(ph[:, 1], 95)
N["original_vs_fits"] = {"frac_melanin_above_fit_p95": 100 * float((orig_phys[:, 0] > p95m).mean()),
                         "frac_blood_above_fit_p95": 100 * float((orig_phys[:, 1] > p95b).mean()),
                         "frac_both_within": 100 * float(((orig_phys[:, 0] <= p95m) & (orig_phys[:, 1] <= p95b)).mean()),
                         "orig_melanin_median": float(np.median(orig_phys[:, 0])),
                         "orig_blood_median": float(np.median(orig_phys[:, 1]))}

# ---------------------------------------------------------------- D. PCA coverage (their basis and ellipse)
cov_issa, sc_issa = pca_cover(issa)
N["pca"] = {"issa": cov_issa}
pca_sc = {}
for k in ["original_F0", "original", "empirical"]:
    N["pca"][k], pca_sc[k] = pca_cover(spectra_for_pca[k])
N["pca"]["inverse_fits"], pca_sc["fits"] = pca_cover(P["sci_pred"])
json.dump(N, open(os.path.join(OUT, "paper_numbers.json"), "w"), indent=1)
print(json.dumps(N, indent=1))

# ---------------------------------------------------------------- figures (PDF for Overleaf + PNG preview)
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
C1, C2, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "pdf.fonttype": 42})
x = np.arange(5)
labels = [f"{c}\n(n={int((cls == c).sum())})" for c in CLASSES]


def boxes(ax, vals):
    for i, c in enumerate(CLASSES):
        ax.boxplot(vals[cls == c], positions=[i], widths=.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=SEQ[i], color=SEQ[i]), medianprops=dict(color=INK, lw=1.8),
                   whiskerprops=dict(color=SEQ[i]), capprops=dict(color=SEQ[i]))
    ax.set_xticks(x, labels)


fig, ax = plt.subplots(2, 2, figsize=(10, 7))
boxes(ax[0, 0], P["rmse_true"]); ax[0, 0].set_ylabel("reconstruction RMSE (percentage points)")
ax[0, 0].set_title("(a) Reconstruction error by eumelanin class", loc="left", fontsize=10)
real_b = [N["real"][c]["both"] for c in CLASSES]
ci = np.array([N["real"][c]["both_ci"] for c in CLASSES])
ax[0, 1].bar(x - .2, real_b, .38, color=C2, label="ISSA spectra")
ax[0, 1].errorbar(x - .2, real_b, yerr=[np.array(real_b) - ci[:, 0], ci[:, 1] - np.array(real_b)],
                  fmt="none", ecolor=INK, lw=1, capsize=2)
ax[0, 1].bar(x + .2, [N["inverse_by_class"]["fit_pass_both"][c] for c in CLASSES], .38, color=C1,
             label="BioSkin fits to the same spectra")
ax[0, 1].text(4, 2, "0 / 0", ha="center", fontsize=8, color=INK)
ax[0, 1].set_xticks(x, labels); ax[0, 1].set_ylim(0, 122); ax[0, 1].set_yticks(range(0, 101, 20))
ax[0, 1].set_ylabel("pass both validation stages (%)"); ax[0, 1].legend(fontsize=8, loc="upper left", ncol=2)
ax[0, 1].set_title("(b) Two-stage validation outcome", loc="left", fontsize=10)
for lo_, hi_, lab in [(1.3, 6.3, "light"), (11, 16, "moderate"), (18, 43, "dark")]:
    ax[1, 0].axhspan(lo_, hi_, color=GRID, alpha=.7, lw=0, zorder=0)
    ax[1, 0].text(4.45, np.sqrt(lo_ * hi_), lab, fontsize=7.5, color=INK2, va="center")
boxes(ax[1, 0], 100 * ph[:, 0]); ax[1, 0].set_yscale("log"); ax[1, 0].set_xlim(-.5, 5.2)
ax[1, 0].set_yticks([1, 3, 10, 30], ["1", "3", "10", "30"]); ax[1, 0].minorticks_off()
ax[1, 0].axhline(100 * hi[0] ** 3, color=INK, ls=":", lw=1)
ax[1, 0].text(-.4, 100 * hi[0] ** 3 * 1.06, "training-domain maximum (43%)", fontsize=7.5, color=INK2)
ax[1, 0].set_ylim(0.2, 60)
ax[1, 0].set_ylabel("fitted melanin volume fraction (%)")
ax[1, 0].set_title("(c) Melanin (shaded: Jacques 1998 ranges)", loc="left", fontsize=10)
ax[1, 1].axhspan(57, 72, color=GRID, alpha=.9, lw=0, zorder=0)
boxes(ax[1, 1], ph[:, 2]); ax[1, 1].axhline(100, color=INK, ls="--", lw=1)
ax[1, 1].set_ylabel("fitted epidermal thickness (µm)")
ax[1, 1].set_title("(d) Thickness (shaded: facial LC-OCT, 57–72 µm)", loc="left", fontsize=10)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"bioskin_inverse_fit_issa.{ext}"), dpi=200)
plt.close(fig)

# sampled (original 4,000) vs fitted-to-ISSA parameter distributions; replaces their fig:parameters
pd.DataFrame({"melanin": ph[:, 0], "blood": ph[:, 1], "thickness_um": ph[:, 2], "mi_class": cls,
              "participant": pid}).to_csv(os.path.join(OUT, "fitted_parameters_issa.csv"), index=False)
dlo, dhi = normalized_to_physical(lo[None])[0], normalized_to_physical(hi[None])[0]
# every fraction shown in percent; thickness stays in micrometres
scale = np.array([100, 100, 1, 100, 100])
names = ["(a) melanin volume fraction (%)", "(b) blood volume fraction (%)", "(c) epidermal thickness (µm)",
         "(d) eumelanin fraction (%)", "(e) blood oxygenation (%)"]
fig, ax = plt.subplots(1, 5, figsize=(13, 2.9))
for j, a in enumerate(ax):
    log = j in (0, 1)
    l_, h_ = dlo[j] * scale[j], dhi[j] * scale[j]
    bins = np.geomspace(l_, h_, 31) if log else np.linspace(l_, h_, 31)
    # fraction of each set per bin (not density), so bar height means the same thing on log and linear axes
    a.hist(orig_phys[:, j] * scale[j], bins=bins, weights=np.full(len(orig_phys), 100 / len(orig_phys)), color=C2,
           alpha=.55, lw=0, label="original 4,000 sample")
    if j < 3:
        a.hist(ph[:, j] * scale[j], bins=bins, weights=np.full(len(ph), 100 / len(ph)), histtype="step", color=C1,
               lw=1.8, label="fits to 7,923 ISSA spectra")
    else:
        a.axvline(FIX[j] * scale[j], color=C1, lw=1.8, label="value fixed in fits")
        a.set_ylim(0, 8); a.legend(fontsize=7, loc="upper left")
    if log:
        a.set_xscale("log")
        ticks = [t for t in (0.1, 0.3, 1, 3, 10, 30) if l_ <= t <= h_]
        a.set_xticks(ticks, [f"{t:g}" for t in ticks]); a.minorticks_off()
    a.set_xlim(l_, h_); a.set_title(names[j], loc="left", fontsize=9)
ax[0].legend(fontsize=7, loc="upper left"); ax[0].set_ylabel("% of set per bin")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"bioskin_parameters_sampled_vs_fitted.{ext}"), dpi=200)
plt.close(fig)

fig, ax = plt.subplots(1, 3, figsize=(12, 3.8), sharex=True, sharey=True)
t = np.linspace(0, 2 * np.pi, 200)
L = np.linalg.cholesky(ref["pca_score_cov"]) @ np.vstack([np.cos(t), np.sin(t)]) * np.sqrt(chi2.ppf(.95, 2))
ell = L + ref["pca_score_mean"][:, None]
for a, (key, title, col) in zip(ax, [("original", "(a) Original 4,000 sample", C2),
                                     ("empirical", "(b) Sampled from fitted parameters", C1),
                                     ("fits", "(c) Per-spectrum BioSkin fits", C1)]):
    a.scatter(sc_issa[:, 0], sc_issa[:, 1], s=2, color="#b9b8b3", lw=0, rasterized=True, label="ISSA")
    a.scatter(pca_sc[key][:, 0], pca_sc[key][:, 1], s=2, color=col, lw=0, alpha=.5, rasterized=True)
    a.plot(ell[0], ell[1], "--", color=INK, lw=1)
    cov = N["pca"]["inverse_fits" if key == "fits" else key]
    a.set_title(f"{title}\n{cov:.1f}% inside ISSA 95% ellipse", loc="left", fontsize=9.5)
    a.set_xlabel("PC1 score")
ax[0].set_ylabel("PC2 score")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, f"pca_issa_bioskin_sci.{ext}"), dpi=200)
plt.close(fig)
print("figures written to", OUT)
