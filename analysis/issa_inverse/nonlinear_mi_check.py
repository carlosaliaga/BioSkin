"""Co-author question (2026-10-25): is the BVCM's melanin effect nonlinear, or is it the specular floor?

ISSA is measured with the specular component included (SCI), so every spectrum carries a roughly flat surface
term F ~ 0.028. On the logit scale, a constant additive floor is invisible for bright skin and dominant for dark
skin: it makes dark spectra brighter than a model that is linear in MI predicts, at every wavelength. That is
the residual pattern seen for EH.

Rough check, in the same spirit as his (least squares, not the Gibbs sampler), on all 7,923 facial spectra:
  A  logit(R)                      ~ 1 + MI + EI                 (the BVCM mean, linear)
  B  logit(R)                      ~ 1 + spline(MI) + EI         (nonlinear-in-MI check)   
  C  logit((R - F) / (1 - F))      ~ 1 + MI + EI                 (linear, specular floor removed)
  D  logit((R - F) / (1 - F))      ~ 1 + spline(MI) + EI
MI and EI are always the indices of the measured spectrum. Each model's residual covariance is estimated from
its own residuals (empirical, not the GP covariance), and its Stage-2-style threshold is the 95th percentile of
its own distances, so the four are compared on equal terms.

    venv/Scripts/python.exe -I analysis/issa_inverse/nonlinear_mi_check.py
"""
import json, os, sys

import numpy as np, pandas as pd
from patsy import dmatrix

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from inverse_pilot import load_issa_face, mi_class, CLASSES

F = 0.028
d = load_issa_face()
R, mi, ei = d["spectra"] / 100, d["mi"], d["ei"]
cls = mi_class(mi)
logit = lambda p: np.log(p / (1 - p))
X_lin = np.column_stack([np.ones_like(mi), mi, ei])
X_spl = np.column_stack([np.asarray(dmatrix("cr(mi, df=5) - 1", {"mi": mi})), ei])   # natural cubic spline in MI

floor = np.clip((R - F) / (1 - F), 1e-3, None)
n_clipped = int(((R - F) <= 1e-3).any(1).sum())
models = {"A linear, measured": (logit(R), X_lin), "B spline MI, measured": (logit(R), X_spl),
          "C linear, floor removed": (logit(floor), X_lin), "D spline MI, floor removed": (logit(floor), X_spl)}

rows, out = [], {"spectra_clipped_by_floor": n_clipped}
for name, (Y, X) in models.items():
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ B
    S = np.cov(E, rowvar=False)
    D = np.sqrt(np.einsum("ij,jk,ik->i", E, np.linalg.inv(S), E))
    thr = np.percentile(D, 95)
    for c in CLASSES:
        m = cls == c
        Em = E[m]
        mean_share = float((Em.mean(0) ** 2).sum() / (Em ** 2).mean(0).sum())   # class-mean residual share
        rows.append({"model": name, "class": c, "median_D_over_thr": float(np.median(D[m]) / thr),
                     "pass_pct": 100 * float((D[m] <= thr).mean()), "class_mean_share_pct": 100 * mean_share,
                     "mean_logit_residual": float(Em.mean())})
t = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(f"spectra with R <= F somewhere (floor-removed value clipped): {n_clipped}\n")
print(t.pivot(index="class", columns="model", values="pass_pct").loc[CLASSES].round(1), "\n")
print(t.pivot(index="class", columns="model", values="class_mean_share_pct").loc[CLASSES].round(0), "\n")
print(t.pivot(index="class", columns="model", values="median_D_over_thr").loc[CLASSES].round(2))
t.to_csv(os.path.join(HERE, "paper", "nonlinear_mi_check.csv"), index=False)
