"""Single global additive offset (specular-included floor) shared by all spectra: scan on the pilot 200.

ISSA is SP/SCI (specular included); BioSkin decoder output is diffuse. Fitting decoder + a to ISSA is
the same as fitting the decoder to ISSA - a, so each offset value is an ordinary primary-config refit.

    venv/Scripts/python.exe -I analysis/issa_inverse/offset_scan.py
"""
import json, os, sys
import numpy as np, pandas as pd, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from full_run import fit_general, FIX, STEPS, SEED, OUT, CLASSES
from inverse_pilot import Decoder, training_box, normalized_to_physical, evaluate, load_issa_face, mi_class, UPSTREAM

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"), allow_pickle=False) as z:
    ref = dict(z)
s1 = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))["stage1_threshold"]
d = load_issa_face(); pilot = pd.read_csv(os.path.join(HERE, "out", "pilot_fits.csv")).issa_row.values
tgt, cls = d["spectra"][pilot], mi_class(d["mi"][pilot])
dec = Decoder(dev); lo, hi, _ = training_box(dev)
fixed = {j: (v - lo[j]) / (hi[j] - lo[j]) for j, v in FIX.items()}
rows = []
for a in [0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5]:
    f = fit_general(dec, tgt - a, lo, hi, [0, 1, 2], fixed, 32, STEPS, True, SEED)
    ph = normalized_to_physical(lo + f["u"] * (hi - lo))
    ev = evaluate(np.clip(f["pred"] + a, 1e-3, 99.9), ref, s1)
    g = pd.DataFrame({"c": cls, "rmse": f["rmse"], "mel": ph[:, 0], "hb": ph[:, 1], "th": ph[:, 2], "both": ev["final"],
                      "th_hi": f["u"][:, 2] > 1 - 1e-3, "neg": (tgt - a).min(1) <= 0})
    for c in CLASSES + ["all"]:
        m = np.ones(len(g), bool) if c == "all" else (g.c == c).values
        rows.append({"offset_pp": a, "class": c, "rmse": g.rmse[m].median(), "melanin": g.mel[m].median(),
                     "hb": g.hb[m].median(), "thick": g.th[m].median(), "thick_at_350": int(g.th_hi[m].sum()),
                     "both": int(g.both[m].sum()), "n": int(m.sum()), "target_nonpos": int(g.neg[m].sum())})
t = pd.DataFrame(rows); t.to_csv(os.path.join(OUT, "offset_scan.csv"), index=False)
pd.set_option("display.width", 200)
for c in CLASSES + ["all"]:
    print(t[t["class"] == c].round(3).to_string(index=False), "\n")
