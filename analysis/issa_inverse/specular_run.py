"""Full inversion with the specular term ISSA's SCI geometry includes and BioSkin's R_d does not.

ISSA: Konica Minolta CM-2600d / CM-700d, d/8, specular component included (SCI). By reciprocity d/8 is
8 deg in, hemispherical out, so the surface term is near-normal Fresnel F(8 deg) ~ R0. BioSkin's R_d is
subsurface only; BioSkin adds specular as a separate step (bioskin/lighting/light_transport.add_specular_reflectance).

Model compared to ISSA:  R_SCI = F + (1 - F) * R_d(decoder).
Because (1 - F) is constant over wavelength, minimising ||F + (1-F) p - y|| is the same as fitting the decoder
to (y - F) / (1 - F); the true RMSE is (1 - F) times the fitted one. Primary config otherwise unchanged
(full domain, ratio/oxy fixed, smoothed). F = 0.028 is normal-incidence Fresnel at n = 1.4; 0.040 is BioSkin's R0.

    venv/Scripts/python.exe -I analysis/issa_inverse/specular_run.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from full_run import fit_general, FIX, K, STEPS, SEED, OUT, CLASSES
from inverse_pilot import Decoder, training_box, normalized_to_physical, evaluate, load_issa_face, mi_class, UPSTREAM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"), allow_pickle=False) as z:
    ref = dict(z)
s1 = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))["stage1_threshold"]
d = load_issa_face()
issa, cls = d["spectra"], mi_class(d["mi"])
dec = Decoder(dev); lo, hi, _ = training_box(dev)
fixed = {j: (v - lo[j]) / (hi[j] - lo[j]) for j, v in FIX.items()}
res = pd.read_csv(os.path.join(OUT, "full_fits.csv"))

runs = [(0.028, None), (0.034, None), (0.040, None), (0.028, 100)]
rows = []
for F, cap in runs:
    h = hi.copy()
    if cap:
        h[2] = (cap * 1e-4 - 0.001) / 0.034
    tag = f"spec{int(round(F * 1000)):03d}" + (f"_cap{cap}" if cap else "")
    target = (issa - 100 * F) / (1 - F)
    f = fit_general(dec, target, lo, h, [0, 1, 2], fixed, K, STEPS, True, SEED)
    ph = normalized_to_physical(lo + f["u"] * (h - lo))
    pred = 100 * F + (1 - F) * f["pred"]                       # SCI-consistent model spectrum, percent
    rmse = np.sqrt(((pred - issa) ** 2).mean(1))
    ev = evaluate(np.clip(pred, 1e-3, 99.9), ref, s1)
    res[f"{tag}_rmse"], res[f"{tag}_both"] = rmse, ev["final"]
    for j, n in enumerate(["melanin", "haemoglobin", "thickness_um"]):
        res[f"{tag}_{n}"] = ph[:, j]
        res[f"{tag}_athi_{n}"] = f["u"][:, j] > 1 - 1e-3
    below = (issa < 100 * F).any(1)
    for c in CLASSES + ["all"]:
        m = np.ones(len(issa), bool) if c == "all" else cls == c
        rows.append({"F": F, "cap_um": cap or "", "class": c, "n": int(m.sum()),
                     "rmse_med": np.median(rmse[m]), "melanin_med": np.median(ph[m, 0]),
                     "hb_med": np.median(ph[m, 1]), "thick_med": np.median(ph[m, 2]),
                     "thick_le100_pct": 100 * np.mean(ph[m, 2] <= 100),
                     "hb_at_ceiling": int((f["u"][m, 1] > 1 - 1e-3).sum()),
                     "thick_at_ceiling": int((f["u"][m, 2] > 1 - 1e-3).sum()),
                     "mel_at_ceiling": int((f["u"][m, 0] > 1 - 1e-3).sum()),
                     "both_pct": 100 * ev["final"][m].mean(), "spectra_below_F": int(below[m].sum())})
    print(f"{tag}: median RMSE {np.median(rmse):.3f} pp, thickness {np.median(ph[:, 2]):.0f} um, "
          f"pass {100 * ev['final'].mean():.1f}%")

# F = 0 row from the primary run, for the table
for c in CLASSES + ["all"]:
    m = np.ones(len(res), bool) if c == "all" else (res.mi_class == c).values
    P = "primary_smoothed"
    rows.append({"F": 0.0, "cap_um": "", "class": c, "n": int(m.sum()), "rmse_med": res[P + "_rmse"][m].median(),
                 "melanin_med": res[P + "_melanin"][m].median(), "hb_med": res[P + "_haemoglobin"][m].median(),
                 "thick_med": res[P + "_thickness_um"][m].median(),
                 "thick_le100_pct": 100 * (res[P + "_thickness_um"][m] <= 100).mean(),
                 "hb_at_ceiling": int(res[P + "_athi_haemoglobin"][m].sum()),
                 "thick_at_ceiling": int(res[P + "_athi_thickness_um"][m].sum()),
                 "mel_at_ceiling": int(res[P + "_athi_melanin"][m].sum()),
                 "both_pct": 100 * res[P + "_both"][m].mean(), "spectra_below_F": 0})
t = pd.DataFrame(rows).sort_values(["class", "cap_um", "F"], key=lambda s: s.map(
    {c: i for i, c in enumerate(CLASSES + ["all"])}) if s.name == "class" else s.astype(str))
t.to_csv(os.path.join(OUT, "specular_runs.csv"), index=False)
res.to_csv(os.path.join(OUT, "full_fits.csv"), index=False)
pd.set_option("display.width", 250)
print(t.round(3).to_string(index=False))
