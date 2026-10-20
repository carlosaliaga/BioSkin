"""Full ISSA inversion (7,923 face spectra) + real-ISSA validation breakdown + scale-factor probe.

Decisions (2026-10-19 discussion):
  primary   = full training domain, melanin type ratio / oxygenation FIXED at mid-domain
              (0.725, 0.95; unidentifiable from 400-700 nm), double-SG-smoothed decoder output
  secondary = same on raw decoder output; all-5-free (cost of fixing); thickness caps 250 / 100 um
  probe     = pilot 200 only: free gain s, and free linear gain s0 + s1*(lambda-550)/150 (geometry test)
Summaries are by MI class and clustered by participant (origin dataset x subject number).

    venv/Scripts/python.exe -I analysis/issa_inverse/full_run.py
"""

import json, os, sys, time

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from inverse_pilot import (Decoder, training_box, normalized_to_physical, near_optimal_spread, mi_class,
                           evaluate, load_issa_face, CLASSES, NAMES, WL_ISSA, UPSTREAM, ROOT)
from analysis.issa_reference import ISSA_XLSX, FACE_SITE_CODES

OUT = os.path.join(HERE, "full_out")
FIX = {3: 0.725, 4: 0.95}
K, STEPS, SEED = 16, 1500, 20261019
OFFSET_MAX = 6.0
T = torch.tensor((WL_ISSA - 550) / 150, dtype=torch.float32)


def load_meta():
    """Same rows, same order as load_issa_face(), plus participant / instrument columns."""
    from scipy.signal import savgol_filter
    issa = pd.read_excel(ISSA_XLSX, sheet_name="ISSA")
    wl = issa.iloc[10, 17:48].values.astype(float)
    sites = issa.iloc[11:, 6].values.astype(int)
    spectra = issa.iloc[11:, 17:48].to_numpy(dtype=float)
    meta = issa.iloc[11:, [1, 2, 3, 7, 8]].copy()
    meta.columns = ["origin", "subject", "ethnicity", "instrument", "spin_spex"]
    face = np.isin(sites, FACE_SITE_CODES)
    f = spectra[face]
    drv = savgol_filter(f, window_length=5, polyorder=3, axis=1, deriv=1) / np.gradient(wl)
    keep = ~np.any(np.abs(drv - drv.mean(0)) > 5 * drv.std(0), axis=1)
    m = meta[face][keep].reset_index(drop=True)
    m["participant"] = m.origin.astype(str) + "_" + m.subject.astype(str)
    return m, f[keep]


def fit_general(dec, tgt, lo, hi, free, fixed_unit, K, steps, smooth, seed, gain=None, lr=0.05):
    """Projected Adam over free unit-box dims (+ optional gain). gain: None | 'const' | 'linear'."""
    dev = dec.device
    N, nf = tgt.shape[0], len(free)
    ng = {None: 0, "const": 1, "linear": 2, "offset": 1, "offset_const": 2}[gain]
    u0 = qmc.Sobol(nf + ng, scramble=True, seed=seed).random(N * K)
    v = torch.tensor(u0, dtype=torch.float32, device=dev).requires_grad_(True)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
    w_t = torch.tensor(hi - lo, dtype=torch.float32, device=dev)
    base = torch.zeros(5, device=dev)
    for j, val in fixed_unit.items():
        base[j] = val
    free_t = torch.tensor(free, device=dev)
    t = T.to(dev)
    y = torch.tensor(tgt, dtype=torch.float32, device=dev).repeat_interleave(K, 0)

    def predict(v):
        u = base.expand(v.shape[0], 5).clone()
        u[:, free_t] = v[:, :nf]
        p = dec.pct31(lo_t + u * w_t, smooth)
        if gain in ("const", "linear"):
            s0 = 0.5 + v[:, nf:nf + 1]                     # gain 0.5 - 1.5
            s1 = (v[:, nf + 1:nf + 2] - 0.5) if ng == 2 else 0.0   # tilt -0.5 - +0.5 across 400-700
            p = p * (s0 + s1 * t)
        elif gain == "offset":
            p = p + OFFSET_MAX * v[:, nf:nf + 1]          # additive floor 0 - 6 pp (specular-included)
        elif gain == "offset_const":
            p = (0.5 + v[:, nf + 1:nf + 2]) * p + OFFSET_MAX * v[:, nf:nf + 1]
        return p, u

    opt = torch.optim.Adam([v], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p, _ = predict(v)
        ((p - y) ** 2).mean(1).sum().backward()
        opt.step(); sched.step()
        with torch.no_grad():
            v.clamp_(0, 1)
    with torch.no_grad():
        p, u = predict(v)
        rmse = ((p - y) ** 2).mean(1).sqrt()
    r = rmse.view(N, K).cpu().numpy()
    b = r.argmin(1)
    sel = torch.tensor(np.arange(N) * K + b, device=dev)
    return {"u_all": u.view(N, K, 5).cpu().numpy(), "rmse_all": r, "rmse": r.min(1),
            "u": u[sel].cpu().numpy(), "pred": p[sel].cpu().numpy(), "v": v[sel].detach().cpu().numpy()}


def cluster_boot(values, groups, stat=np.mean, B=2000, seed=0):
    """Percentile CI resampling participants, not spectra."""
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by = [np.flatnonzero(inv == g) for g in range(len(ug))]
    out = []
    for _ in range(B):
        pick = rng.integers(0, len(ug), len(ug))
        out.append(stat(values[np.concatenate([idx_by[g] for g in pick])]))
    return np.percentile(out, [2.5, 97.5])


def gain_probe(dec, lo, hi, fixed_unit, issa, cls, ref, s1_thr, gains):
    """Pilot 200 only: does a free gain / tilt / additive floor remove the thickness inflation?"""
    pilot = pd.read_csv(os.path.join(HERE, "out", "pilot_fits.csv")).issa_row.values
    probe = {}
    for gain in gains:
        f = fit_general(dec, issa[pilot], lo, hi, [0, 1, 2], fixed_unit, 32, STEPS, True, SEED, gain=gain)
        ph = normalized_to_physical(lo + f["u"] * (hi - lo))
        ev = evaluate(np.clip(f["pred"], 1e-3, 99.9), ref, s1_thr)
        g = pd.DataFrame({"mi_class": cls[pilot], "rmse": f["rmse"], "melanin": ph[:, 0], "hb": ph[:, 1],
                          "thick": ph[:, 2], "both": ev["final"]})
        if gain in ("offset", "offset_const"):
            g["offset_pp"] = OFFSET_MAX * f["v"][:, 3]
            g["gain"] = 0.5 + f["v"][:, 4] if gain == "offset_const" else 1.0
        elif gain:
            g["s0"] = 0.5 + f["v"][:, 3]
            g["s1"] = f["v"][:, 4] - 0.5 if gain == "linear" else 0.0
            g["gain400"], g["gain700"] = g.s0 - g.s1, g.s0 + g.s1
        g.to_csv(os.path.join(OUT, f"gain_probe_{gain or 'none'}.csv"), index=False)
        agg = {"rmse": "median", "melanin": "median", "hb": "median", "thick": "median", "both": "sum"}
        if gain in ("offset", "offset_const"):
            agg.update({"offset_pp": "median", "gain": "median"})
        elif gain:
            agg.update({"gain400": "median", "gain700": "median"})
        probe[gain or "none"] = g.groupby("mi_class").agg(agg).loc[CLASSES]
        print(f"\ngain={gain}\n", probe[gain or "none"].round(3).to_string())
    return probe


def main():
    probe_only = "--probe-only" in sys.argv
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"), allow_pickle=False) as z:
        ref = dict(z)
    s1_thr = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))["stage1_threshold"]

    d = load_issa_face()
    meta, spec_check = load_meta()
    assert np.array_equal(spec_check, d["spectra"])
    issa, mi, ei = d["spectra"], d["mi"], d["ei"]
    cls = mi_class(mi)
    pid = meta.participant.values
    S = {"n": len(issa), "participants": int(meta.participant.nunique())}

    if probe_only:
        dec = Decoder(dev)
        lo, hi, _ = training_box(dev)
        fixed_unit = {j: (val - lo[j]) / (hi[j] - lo[j]) for j, val in FIX.items()}
        probe = gain_probe(dec, lo, hi, fixed_unit, issa, cls, ref, s1_thr, ("offset", "offset_const"))
        S = json.load(open(os.path.join(OUT, "summary.json")))
        S["gain_probe"].update({k: v.round(4).to_dict() for k, v in probe.items()})
        json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
        return
    # ------------------------------------------------ 1. their evaluate() on all real ISSA spectra
    e = evaluate(issa, ref, s1_thr)
    real = pd.DataFrame({"mi_class": cls, "stage1": e["stage1"], "both": e["final"], "participant": pid,
                         "train": np.isin(np.arange(len(issa)), ref["train_indices"])})
    rows = []
    for c in CLASSES:
        m = real.mi_class == c
        rows.append({"class": c, "spectra": int(m.sum()), "participants": int(real.participant[m].nunique()),
                     "stage1 %": 100 * real.stage1[m].mean(), "both %": 100 * real.both[m].mean(),
                     "both % CI (participant bootstrap)": "%.1f-%.1f" % tuple(
                         100 * cluster_boot(real.both[m].values.astype(float), real.participant[m].values)),
                     "both % train rows": 100 * real.both[m & real.train].mean()})
    real_tab = pd.DataFrame(rows)
    real_tab.to_csv(os.path.join(OUT, "real_issa_validation_by_class.csv"), index=False)
    print(real_tab.round(1).to_string(index=False))
    S["real_issa_both_pct"] = 100 * float(real.both.mean())
    # instrument / geometry breakdown
    S["instrument_by_class"] = pd.crosstab(cls, meta.instrument.astype(str) + "/" + meta.spin_spex.astype(str)).to_dict()

    # ------------------------------------------------ 2. full inversion
    dec = Decoder(dev)
    lo, hi, _ = training_box(dev)
    fixed_unit = {j: (val - lo[j]) / (hi[j] - lo[j]) for j, val in FIX.items()}

    def capped(um):
        h = hi.copy(); h[2] = (um * 1e-4 - 0.001) / 0.034
        return h

    variants = {
        "primary_smoothed": dict(free=[0, 1, 2], fixed=fixed_unit, smooth=True, hi=hi),
        "primary_raw": dict(free=[0, 1, 2], fixed=fixed_unit, smooth=False, hi=hi),
        "free5_smoothed": dict(free=[0, 1, 2, 3, 4], fixed={}, smooth=True, hi=hi),
        "cap250_smoothed": dict(free=[0, 1, 2], fixed=fixed_unit, smooth=True, hi=capped(250)),
        "cap100_smoothed": dict(free=[0, 1, 2], fixed=fixed_unit, smooth=True, hi=capped(100)),
    }
    res = pd.DataFrame({"issa_row": np.arange(len(issa)), "MI": mi, "EI": ei, "mi_class": cls,
                        "participant": pid, "origin": meta.origin.values, "site": d["sites"],
                        "instrument": meta.instrument.astype(str).values, "spin_spex": meta.spin_spex.astype(str).values,
                        "real_stage1": e["stage1"], "real_both": e["final"]})
    S["fit_seconds"] = {}
    for name, cfg in variants.items():
        t0 = time.time()
        f = fit_general(dec, issa, lo, cfg["hi"], cfg["free"], cfg["fixed"], K, STEPS, cfg["smooth"], SEED)
        S["fit_seconds"][name] = time.time() - t0
        h = cfg["hi"]
        x = lo + f["u"] * (h - lo)
        ph = normalized_to_physical(x)
        ev = evaluate(np.clip(f["pred"], 1e-3, 99.9), ref, s1_thr)
        res[f"{name}_rmse"] = f["rmse"]
        for j, n in enumerate(NAMES[:3]):
            res[f"{name}_{n}"] = ph[:, j]
            res[f"{name}_atlo_{n}"] = f["u"][:, j] < 1e-3
            res[f"{name}_athi_{n}"] = f["u"][:, j] > 1 - 1e-3
        if name == "free5_smoothed":
            res[f"{name}_ratio"], res[f"{name}_oxy"] = ph[:, 3], ph[:, 4]
        res[f"{name}_stage1"], res[f"{name}_both"] = ev["stage1"], ev["final"]
        res[f"{name}_fitMI"], res[f"{name}_fitEI"] = ev["mi"], ev["ei"]
        if name == "primary_smoothed":
            spread, n_near, n_basins = near_optimal_spread(f["u_all"], f["rmse_all"])
            res["primary_basins"] = n_basins
            for j, n in enumerate(NAMES[:3]):
                res[f"primary_spread_unit_{n}"] = spread[:, j]
            np.save(os.path.join(OUT, "primary_pred.npy"), f["pred"])
        print(f"{name}: {S['fit_seconds'][name]:.0f}s median RMSE {np.median(f['rmse']):.3f}  "
              f"both-pass {100 * ev['final'].mean():.1f}%")
    res.to_csv(os.path.join(OUT, "full_fits.csv"), index=False)

    # ------------------------------------------------ 3. gain probe on the pilot 200
    probe = gain_probe(dec, lo, hi, fixed_unit, issa, cls, ref, s1_thr,
                       (None, "const", "linear", "offset", "offset_const"))
    S["gain_probe"] = {k: v.round(4).to_dict() for k, v in probe.items()}
    json.dump(S, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
