"""Co-author question (2026-10-25): does the BioSkin encoder agree with the gradient fits on ISSA?

The encoder maps RGB (3 numbers) to the 5 decoder parameters. It was trained on RGB computed from the
subsurface-only (diffuse) Monte Carlo spectra, so for a fair test the ISSA surface term is removed first:
R_d = (R - F) / (1 - F), F = 0.028. The 400-700 nm ISSA spectrum is extended to the decoder's 380-778 nm visible
range by holding the end values, converted to RGB with BioSkin's own colour code, and passed to the encoder.

    venv/Scripts/python.exe -I analysis/issa_inverse/encoder_vs_fits.py
"""
import json, os, sys

import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from inverse_pilot import (Decoder, training_box, normalized_to_physical, evaluate, load_issa_face, mi_class,
                           UPSTREAM, WL_DEC, WL_ISSA, CLASSES)
import bioskin.spectrum.color_spectrum as color

F = 0.028
OUT = os.path.join(HERE, "paper")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"), allow_pickle=False) as z:
    ref = dict(z)
s1 = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))["stage1_threshold"]
d = load_issa_face(); issa, cls = d["spectra"], mi_class(d["mi"])
dec = Decoder(dev); lo, hi, _ = training_box(dev)
fits = pd.read_csv(os.path.join(OUT, "fitted_parameters_issa.csv"))      # written by paper_numbers.py

rd = np.clip((issa / 100 - F) / (1 - F), 1e-4, 1)
vis = WL_DEC[:200]
spec = np.stack([np.interp(vis, WL_ISSA, s) for s in rd])
cs = color.ColorSpectrum(dev, 200)
with torch.no_grad():
    rgb = cs.spectrum_to_rgb(torch.tensor(spec, dtype=torch.float64, device=dev)).float()
    x = dec.net.encoder(rgb)
    pred = 100 * F + (1 - F) * dec.pct31(x, smooth=True).cpu().numpy()
x = x.cpu().numpy()
ph = normalized_to_physical(x)
rmse = np.sqrt(((pred - issa) ** 2).mean(1))
ev = evaluate(np.clip(pred, 1e-3, 99.9), ref, s1)
outside = (x < lo - 1e-6) | (x > hi + 1e-6)

res = {"n": len(issa),
       "encoder_rmse_median": float(np.median(rmse)),
       "fit_rmse_median": 0.8698,
       "encoder_pass_both_pct": 100 * float(ev["final"].mean()),
       "pct_outside_training_domain_any": 100 * float(outside.any(1).mean()),
       "pct_outside_by_param": {n: 100 * float(outside[:, j].mean()) for j, n in
                                enumerate(["melanin", "blood", "thickness", "ratio", "oxygenation"])},
       "encoder_ratio_range": [float(ph[:, 3].min()), float(ph[:, 3].max())],
       "encoder_oxy_range": [float(ph[:, 4].min()), float(ph[:, 4].max())],
       "spearman_vs_fit": {n: float(pd.Series(ph[:, j]).corr(fits[c], method="spearman"))
                           for j, (n, c) in enumerate([("melanin", "melanin"), ("blood", "blood"),
                                                       ("thickness", "thickness_um")])},
       "median_ratio_encoder_over_fit": {n: float(np.median(ph[:, j] / fits[c]))
                                         for j, (n, c) in enumerate([("melanin", "melanin"), ("blood", "blood"),
                                                                     ("thickness", "thickness_um")])},
       "by_class": {c: {"encoder_rmse": float(np.median(rmse[cls == c])),
                        "encoder_melanin_pct": float(100 * np.median(ph[cls == c, 0])),
                        "fit_melanin_pct": float(100 * np.median(fits.melanin[cls == c])),
                        "encoder_pass": float(100 * ev["final"][cls == c].mean())} for c in CLASSES}}
# Blood is masked by epidermal melanin: per-class agreement, and how much +50% blood moves the fitted spectrum.
enc = pd.DataFrame({"melanin": ph[:, 0], "blood": ph[:, 1], "thickness_um": ph[:, 2]})


def decode(df):
    x = np.column_stack([df.melanin ** (1 / 3), df.blood ** (1 / 4), (df.thickness_um * 1e-4 - 0.001) / 0.034,
                         np.full(len(df), .725), np.full(len(df), .95)]).astype(np.float32)
    with torch.no_grad():
        return (1 - F) * dec.pct31(torch.tensor(x, device=dev), smooth=True).cpu().numpy()


more_blood = fits.copy(); more_blood["blood"] *= 1.5
d_blood = np.abs(decode(more_blood) - decode(fits)).mean(1)
for c in CLASSES:
    m = cls == c
    res["by_class"][c].update({
        "spearman_melanin": float(enc.melanin[m].corr(fits.melanin[m], method="spearman")),
        "spearman_blood": float(enc.blood[m].corr(fits.blood[m], method="spearman")),
        "spectrum_change_pp_for_50pct_more_blood": float(np.median(d_blood[m]))})
json.dump(res, open(os.path.join(OUT, "encoder_vs_fits.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
