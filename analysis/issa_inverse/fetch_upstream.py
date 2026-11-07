"""Download the lasisilab/SkinSim_Quarto files this analysis needs, into analysis/issa_inverse/upstream/.

Their repository carries no licence, so the files are fetched rather than redistributed here.
`check_bioskin_rerun.evaluate()` and `reference_state.npz` are the two-stage validation used unchanged
by every script in this folder; the rest is kept for provenance.

    venv/Scripts/python.exe -I analysis/issa_inverse/fetch_upstream.py
"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/lasisilab/SkinSim_Quarto/main/"
FILES = [
    "code/check_bioskin_rerun.py",            # their evaluate(): stage 1 GMM + stage 2 BVCM Mahalanobis
    "code/bioskin_inputs.py",                  # imported by check_bioskin_rerun.py
    "code/bvcm_gibbs.py",
    "output/corrected_validation/reference_state.npz",   # fitted GMM, BVCM betas, covariance, thresholds, PCA
    "output/corrected_validation/gmm_mi_ei_reference.npz",  # frozen stage-1 mixture (from 2026-10-25)
    "output/corrected_validation/summary.json",          # stage-1 threshold, published counts, ISSA workbook hash
    "figures/bioskin_properties_distribution.pdf",
    "README.md",
]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upstream")

for path in FILES:
    dest = os.path.join(OUT, *path.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(BASE + path) as r, open(dest, "wb") as f:
        f.write(r.read())
    print("fetched", path)
