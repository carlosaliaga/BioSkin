"""Shared ISSA reference preparation, mirroring SkinSim_Quarto `notebooks/bvcm.ipynb`.

Every step below is transcribed from their notebook. Where their code leaves
something ambiguous it is marked DEVIATION or FLAG in the comments and listed in
`analysis/two_stage_deviations.md` — nothing is silently reinterpreted.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ISSA_XLSX = os.path.join(HERE, "issa", "ISSA_17_Jan_2025_Yan_Lu.xlsx")
FACE_SITE_CODES = [2, 3, 4, 5, 6, 9]
IDX_560, IDX_680 = 16, 28          # their column indices into the 31-point grid

logit = lambda p: np.log(p / (1 - p))
inv_logit = lambda l: 1 / (1 + np.exp(-l))


def load_issa_face():
    """ISSA face sites, savgol-derivative outliers removed, MI/EI, their 80/20 split."""
    issa = pd.read_excel(ISSA_XLSX, sheet_name="ISSA")
    wavelengths = issa.iloc[10, 17:48].values.astype(float)
    sites = issa.iloc[11:, 6].values.astype(int)
    spectra = issa.iloc[11:, 17:48].to_numpy(dtype=float)

    face = spectra[np.isin(sites, FACE_SITE_CODES), :]
    face_sites = sites[np.isin(sites, FACE_SITE_CODES)]

    drv = savgol_filter(face, window_length=5, polyorder=3, axis=1, deriv=1) / np.gradient(wavelengths)
    outlier = np.any(np.abs(drv - drv.mean(axis=0)) > 5 * drv.std(axis=0), axis=1)
    clean = face[~outlier, :]

    mi = 100 * np.log10(100 / clean[:, IDX_680])
    ei = 100 * np.log10(clean[:, IDX_680] / clean[:, IDX_560])
    X = np.column_stack((np.ones(clean.shape[0]), mi, ei))

    # their seed 123 is set once at the top of the notebook and nothing consumes
    # the RNG before this call, so the split reproduces exactly
    np.random.seed(123)
    n = clean.shape[0]
    train_idx = np.random.choice(n, size=int(0.8 * n), replace=False)
    test_idx = np.setdiff1d(np.arange(n), train_idx)

    return {
        "wavelengths": wavelengths, "spectra": clean, "sites": face_sites[~outlier],
        "mi": mi, "ei": ei, "X": X,
        "train_idx": train_idx, "test_idx": test_idx,
        "n_face_raw": int(face.shape[0]), "n_outliers": int(outlier.sum()),
    }


def candidate_mi_ei(spectra_pct):
    mi = 100 * np.log10(100 / spectra_pct[:, IDX_680])
    ei = 100 * np.log10(spectra_pct[:, IDX_680] / spectra_pct[:, IDX_560])
    return mi, ei


def to_issa_grid(spectra_unit):
    """Our 380-998nm/2nm spectra in [0,1] -> ISSA 400-700nm/10nm grid, in percent."""
    our_wl = 380.0 + np.arange(spectra_unit.shape[1]) * 2.0
    issa_wl = np.arange(400.0, 701.0, 10.0)
    return 100.0 * np.asarray(spectra_unit, dtype=float)[:, np.isin(our_wl, issa_wl)]
