"""Inverse-fit pilot: can the BioSkin decoder reproduce individual ISSA facial spectra?

For each ISSA spectrum, find decoder parameters minimising RMSE (reflectance %) on the
31 ISSA wavelengths (400-700 nm / 10 nm, which sit exactly on the decoder's 380-998 / 2 nm
grid, so no interpolation). Box constraint = the decoder's Monte Carlo training support,
nothing else. K multi-start projected-Adam runs per spectrum, all batched on the GPU.

Also runs, as baselines / feasibility probes:
  - the RGB encoder path (ISSA -> RGB -> encoder -> decoder), to justify not using it;
  - their two-stage validation (`upstream/code/check_bioskin_rerun.evaluate`, unchanged,
    with their saved `reference_state.npz`) on the fits and on two forward samples;
  - timing probes for the full 7,923-spectrum inversion.

Nothing in bioskin/ is modified. Downloaded upstream code is imported, not edited.

    venv/Scripts/python.exe -I analysis/issa_inverse/inverse_pilot.py
"""

import argparse, json, os, sys, time

import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter
from scipy.stats import qmc, gaussian_kde

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
UPSTREAM = os.path.join(HERE, "upstream")
sys.path.insert(0, ROOT)
sys.path.append(os.path.join(UPSTREAM, "code"))  # appended, so it cannot shadow anything

from analysis.issa_reference import load_issa_face
from bioskin.bioskin import load_model_and_trained_weights
import bioskin.spectrum.color_spectrum as color
from check_bioskin_rerun import evaluate  # theirs, unchanged

SIM = ("dataset_wide_thickness_wide_spectrum_380_1000nm_2nm_1M_photons_{}_test_"
       "me[0.001-0.430]_bl[0.000-0.200]_th[0.000-1.000]_eu[0.650-0.800]_ox[0.900-1.000].pt")
WL_DEC = 380.0 + 2.0 * np.arange(310)
WL_ISSA = np.arange(400.0, 701.0, 10.0)
IDX31 = np.searchsorted(WL_DEC, WL_ISSA)
CLASSES = ["EML", "EMIL", "EMI", "EMIH", "EH"]
MI_EDGES = [-np.inf, 25, 50, 75, 100, np.inf]
PILOT_N = {"EML": 45, "EMIL": 50, "EMI": 45, "EMIH": 45, "EH": 10_000}  # EH: take all
NAMES = ["melanin", "haemoglobin", "thickness_um", "melanin_type_ratio", "oxygenation"]


# ---------------------------------------------------------------- parameter spaces
def physical_to_normalized(p):
    """Physical (mel, hb, thick_cm, ratio, oxy) -> decoder input (params_io unwarp)."""
    p = np.asarray(p, float)
    return np.column_stack([p[:, 0] ** (1 / 3), p[:, 1] ** (1 / 4),
                            (p[:, 2] - 0.001) / 0.034, p[:, 3], p[:, 4]])


def normalized_to_physical(n):
    n = np.asarray(n, float)
    return np.column_stack([n[:, 0] ** 3, n[:, 1] ** 4, 1e4 * (0.001 + 0.034 * n[:, 2]),
                            n[:, 3], n[:, 4]])  # thickness reported in micrometres


def mi_class(mi):
    return np.array(CLASSES)[np.digitize(mi, MI_EDGES[1:-1])]


def sg_matrix(n=310):
    """Double Savitzky-Golay (15, 3) as used for the shared packages, as a matrix: y_s = y @ S."""
    eye = np.eye(n)
    return savgol_filter(savgol_filter(eye, 15, 3, axis=1), 15, 3, axis=1)


# ---------------------------------------------------------------- decoder
class Decoder:
    def __init__(self, device):
        model, self.params, ok = load_model_and_trained_weights(
            os.path.join(ROOT, "pretrained_models", "BioSkin"), device)
        assert ok and self.params.D_skin == 5
        self.net = getattr(model, "module", model)
        for w in self.net.parameters():
            w.requires_grad_(False)
        self.device = device
        S = sg_matrix()
        self.S31 = torch.tensor(S[:, IDX31], dtype=torch.float32, device=device)
        self.idx31 = torch.tensor(IDX31, device=device)

    def full(self, x):                       # (N,5) normalized -> (N,310) fraction
        return self.net.decoder(x)

    def pct31(self, x, smooth=False):        # (N,5) normalized -> (N,31) percent
        y = self.full(x)
        y = y @ self.S31 if smooth else y[:, self.idx31]
        return 100.0 * y


def training_box(device):
    p = torch.load(os.path.join(ROOT, "datasets", SIM.format("params")), weights_only=False).numpy()
    n = physical_to_normalized(p)
    return n.min(0), n.max(0), p


# ---------------------------------------------------------------- fitting
def fit(dec, target_pct, lo, hi, K=32, steps=3000, lr=0.05, smooth=False, seed=0):
    """Multi-start projected Adam in the unit box. Returns u (N,K,5) and rmse (N,K)."""
    dev = dec.device
    N = target_pct.shape[0]
    u0 = qmc.Sobol(5, scramble=True, seed=seed).random(N * K)
    u = torch.tensor(u0, dtype=torch.float32, device=dev).requires_grad_(True)
    lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
    w_t = torch.tensor(hi - lo, dtype=torch.float32, device=dev)
    tgt = torch.tensor(target_pct, dtype=torch.float32, device=dev).repeat_interleave(K, 0)
    opt = torch.optim.Adam([u], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 1e-3)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        mse = ((dec.pct31(lo_t + u * w_t, smooth) - tgt) ** 2).mean(1)
        mse.sum().backward()
        opt.step(); sched.step()
        with torch.no_grad():
            u.clamp_(0.0, 1.0)
    with torch.no_grad():
        mse = ((dec.pct31(lo_t + u * w_t, smooth) - tgt) ** 2).mean(1)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return u.detach().cpu().numpy().reshape(N, K, 5), mse.sqrt().cpu().numpy().reshape(N, K)


def local_halfwidth(dec, u_best, lo, hi, delta_pp=0.5, smooth=False):
    """Linearised half-width (unit-box fraction) of each parameter keeping RMSE change <= delta.

    ||J du||^2 / 31 <= delta^2  ->  max |du_i| = delta * sqrt(31 * [(J^T J)^-1]_ii).
    Also returns singular values of J (sensitivity of the 31-pt curve to each unit direction).
    """
    dev = dec.device
    lo_t = torch.tensor(lo, dtype=torch.float32, device=dev)
    w_t = torch.tensor(hi - lo, dtype=torch.float32, device=dev)
    f = lambda u: dec.pct31((lo_t + u * w_t)[None], smooth)[0]
    J = torch.func.vmap(torch.func.jacrev(f))(torch.tensor(u_best, dtype=torch.float32, device=dev))
    J = J.double().cpu().numpy()                                   # (N,31,5)
    JtJ = np.einsum("nwi,nwj->nij", J, J)
    inv = np.linalg.pinv(JtJ)
    hw = delta_pp * np.sqrt(31 * np.clip(np.einsum("nii->ni", inv), 0, None))
    sv = np.linalg.svd(J, compute_uv=False)
    return hw, sv


def near_optimal_spread(u, rmse, tol_abs=0.05, tol_rel=0.02, sep=0.05):
    """Spread of restarts whose RMSE is within tol of the best; count of distinct basins."""
    N = u.shape[0]
    spread = np.zeros((N, 5)); n_near = np.zeros(N, int); n_basins = np.zeros(N, int)
    for i in range(N):
        best = rmse[i].min()
        m = rmse[i] <= best + max(tol_abs, tol_rel * best)
        ui = u[i, m]
        n_near[i] = m.sum()
        spread[i] = ui.max(0) - ui.min(0)
        centres = []
        for v in ui[np.argsort(rmse[i, m])]:
            if all(np.abs(v - c).max() > sep for c in centres):
                centres.append(v)
        n_basins[i] = len(centres)
    return spread, n_near, n_basins


# ---------------------------------------------------------------- encoder baseline
def encoder_baseline(dec, issa_pct):
    """ISSA % (400-700/10) -> 380-778/2 fraction (edge hold) -> RGB -> encoder -> decoder."""
    vis = WL_DEC[:200]
    spec = np.stack([np.interp(vis, WL_ISSA, s / 100.0) for s in issa_pct])
    cs = color.ColorSpectrum(dec.device, 200)
    with torch.no_grad():
        rgb = cs.spectrum_to_rgb(torch.tensor(spec, dtype=torch.float64, device=dec.device))
        x = dec.net.encoder(rgb.float())
        pred = dec.pct31(x).cpu().numpy()
    return x.cpu().numpy(), pred


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {"device": str(dev), "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
               "K": args.K, "steps": args.steps, "seed": args.seed}

    # --- reference and data
    with np.load(os.path.join(UPSTREAM, "output/corrected_validation/reference_state.npz"),
                 allow_pickle=False) as z:
        ref = dict(z)
    up_sum = json.load(open(os.path.join(UPSTREAM, "output/corrected_validation/summary.json")))
    s1_thr = up_sum["stage1_threshold"]
    d = load_issa_face()
    assert d["spectra"].shape == (7923, 31) and np.array_equal(d["wavelengths"], WL_ISSA)
    issa, mi_all, ei_all = d["spectra"], d["mi"], d["ei"]
    cls_all = mi_class(mi_all)
    counts = {c: int((cls_all == c).sum()) for c in CLASSES}
    summary["issa_class_counts"] = counts

    # their reference reproduces on real ISSA (sanity)
    r = evaluate(issa, ref, s1_thr)
    summary["issa_all_two_stage_pass_pct"] = 100 * float(r["final"].mean())
    summary["issa_train_stage2_pass_pct_of_s1"] = 100 * float(
        r["final"][ref["train_indices"]].sum() / r["stage1"][ref["train_indices"]].sum())

    # --- pilot selection
    rng = np.random.default_rng(args.seed)
    pilot = np.concatenate([rng.choice(np.flatnonzero(cls_all == c), min(PILOT_N[c], counts[c]),
                                       replace=False) for c in CLASSES])
    tgt, mi, ei, cls = issa[pilot], mi_all[pilot], ei_all[pilot], cls_all[pilot]
    weights = np.array([counts[c] / (cls == c).sum() for c in cls])  # reweight to ISSA mix
    summary["pilot_n"] = int(len(pilot))
    summary["pilot_class_n"] = {c: int((cls == c).sum()) for c in CLASSES}

    # --- decoder and domain
    dec = Decoder(dev)
    lo, hi, sim_phys = training_box(dev)
    summary["domain_normalized"] = {"lo": lo.tolist(), "hi": hi.tolist()}
    pl, ph = normalized_to_physical(lo[None])[0], normalized_to_physical(hi[None])[0]
    summary["domain_physical"] = {n: [float(a), float(b)] for n, a, b in zip(NAMES, pl, ph)}

    # sanity: our lean decoder + SG matrix reproduce the shared corrected package
    cp = pd.read_csv(os.path.join(ROOT, "analysis/corrected/corrected_skin_props_network_ready.csv"),
                     header=None).to_numpy(np.float32)[:, :5]
    sm = pd.read_csv(os.path.join(ROOT, "analysis/corrected/corrected_spectral_reflectance_smoothed.csv"),
                     header=None).to_numpy()
    with torch.no_grad():
        xs = torch.tensor(cp, device=dev)
        ours_s = dec.pct31(xs, smooth=True).cpu().numpy()
        ours_r = dec.pct31(xs, smooth=False).cpu().numpy()
    summary["package_repro_max_abs_pp"] = float(np.abs(ours_s - 100 * sm[:, IDX31]).max())
    rs, rr = evaluate(ours_s, ref, s1_thr), evaluate(ours_r, ref, s1_thr)
    summary["package_two_stage_smoothed"] = [int(rs["stage1"].sum()), int(rs["final"].sum())]
    summary["package_two_stage_raw"] = [int(rr["stage1"].sum()), int(rr["final"].sum())]
    orig_phys = normalized_to_physical(cp)

    # --- inversion
    fits = {}
    for smooth in (False, True):
        key = "smoothed" if smooth else "raw"
        t0 = time.time()
        u, rmse = fit(dec, tgt, lo, hi, args.K, args.steps, smooth=smooth, seed=args.seed)
        fits[key] = (u, rmse, time.time() - t0)
        print(f"fit {key}: {fits[key][2]:.1f}s  median best RMSE {np.median(rmse.min(1)):.3f} pp")
    summary["fit_seconds"] = {k: v[2] for k, v in fits.items()}

    # convergence check: half and double the step budget on the raw fit
    conv = {}
    for st in (args.steps // 2, args.steps * 2):
        _, rm = fit(dec, tgt, lo, hi, args.K, st, seed=args.seed)
        conv[st] = float(np.median(rm.min(1)))
    conv[args.steps] = float(np.median(fits["raw"][1].min(1)))
    summary["convergence_median_best_rmse"] = conv
    # restart-count check: how often does the best of the first k restarts reach the overall best?
    rm = fits["raw"][1]
    summary["restarts_needed"] = {k: float(np.mean(rm[:, :k].min(1) <= rm.min(1) + 0.02))
                                  for k in (1, 2, 4, 8, 16, 32) if k <= args.K}

    u, rmse, _ = fits["raw"]
    best = rmse.argmin(1)
    u_best = u[np.arange(len(u)), best]
    x_best = lo + u_best * (hi - lo)
    phys = normalized_to_physical(x_best)
    rmse_best = rmse.min(1)
    with torch.no_grad():
        pred_raw = dec.pct31(torch.tensor(x_best, dtype=torch.float32, device=dev)).cpu().numpy()
        us = fits["smoothed"][0][np.arange(len(u)), fits["smoothed"][1].argmin(1)]
        pred_sm = dec.pct31(torch.tensor(lo + us * (hi - lo), dtype=torch.float32, device=dev),
                            smooth=True).cpu().numpy()

    spread, n_near, n_basins = near_optimal_spread(u, rmse)
    hw, sv = local_halfwidth(dec, u_best, lo, hi)
    at_bound = (u_best < 1e-3) | (u_best > 1 - 1e-3)

    enc_x, enc_pred = encoder_baseline(dec, tgt)
    enc_rmse = np.sqrt(((enc_pred - tgt) ** 2).mean(1))

    # nearest Monte Carlo spectrum (internal-only baseline; MC set is not shareable)
    sim_spec = 100 * torch.load(os.path.join(ROOT, "datasets", SIM.format("spec")),
                                weights_only=False).numpy()[:, IDX31]
    nn_rmse = np.sqrt(((tgt[:, None, :] - sim_spec[None]) ** 2).mean(2)).min(1)

    # their two-stage on: real pilot spectra, BioSkin fits (raw / smoothed), encoder recon
    two = {}
    for name, S in [("issa_observed", tgt), ("fit_raw", pred_raw), ("fit_smoothed", pred_sm),
                    ("encoder", np.clip(enc_pred, 1e-3, 99.9))]:
        e = evaluate(S, ref, s1_thr)
        two[name] = {"stage1": e["stage1"], "final": e["final"], "dist": e["spectral_distance"],
                     "mi": e["mi"], "ei": e["ei"]}

    # --- per-spectrum table
    df = pd.DataFrame({"issa_row": pilot, "site": d["sites"][pilot], "MI": mi, "EI": ei,
                       "mi_class": cls, "weight": weights,
                       "rmse_fit_raw": rmse_best, "rmse_fit_smoothed": fits["smoothed"][1].min(1),
                       "rmse_encoder": enc_rmse, "rmse_nearest_mc": nn_rmse,
                       "fit_MI": two["fit_raw"]["mi"], "fit_EI": two["fit_raw"]["ei"],
                       "n_near_optimal": n_near, "n_basins": n_basins})
    for j, n in enumerate(NAMES):
        df[n] = phys[:, j]
        df[f"enc_{n}"] = normalized_to_physical(enc_x)[:, j]
        df[f"spread_unit_{n}"] = spread[:, j]
        df[f"halfwidth05_unit_{n}"] = np.minimum(hw[:, j], 1.0)
        df[f"at_bound_{n}"] = at_bound[:, j]
    for k, v in two.items():
        df[f"{k}_stage1"], df[f"{k}_pass"], df[f"{k}_D"] = v["stage1"], v["final"], v["dist"]
    df["jac_cond"] = sv[:, 0] / sv[:, -1]
    df.to_csv(os.path.join(args.out, "pilot_fits.csv"), index=False)
    np.savez(os.path.join(args.out, "pilot_arrays.npz"), u=u, rmse=rmse, target=tgt, pred_raw=pred_raw,
             pred_smoothed=pred_sm, pred_encoder=enc_pred, sv=sv, lo=lo, hi=hi, pilot=pilot)

    # --- forward probes (small batches, their evaluate unchanged)
    fwd = {}
    kde = gaussian_kde(u_best.T, weights=weights)
    for name, U in [("empirical_kde_4000", np.clip(kde.resample(4000, seed=args.seed).T, 0, 1)),
                    ("sobol_4096", qmc.Sobol(5, scramble=True, seed=args.seed).random(4096))]:
        X = torch.tensor(lo + U * (hi - lo), dtype=torch.float32, device=dev)
        out = {}
        for smooth in (False, True):
            t0 = time.time()
            with torch.no_grad():
                S = dec.pct31(X, smooth).cpu().numpy()
            t1 = time.time()
            e = evaluate(S, ref, s1_thr)
            t2 = time.time()
            out["smoothed" if smooth else "raw"] = {
                "n": len(S), "stage1_pct": 100 * float(e["stage1"].mean()),
                "both_pct": 100 * float(e["final"].mean()),
                "decode_s": t1 - t0, "evaluate_s": t2 - t1}
        fwd[name] = out
        if name.startswith("sobol"):
            np.save(os.path.join(args.out, "sobol_u.npy"), U)
    summary["forward_probes"] = fwd

    # --- timing probe at full inversion size (a few steps only)
    full_rows = 7923 * args.K
    probe = np.repeat(issa[:7923], 1, 0)
    t0 = time.time()
    fit(dec, probe, lo, hi, args.K, 50, seed=1)
    per_step = (time.time() - t0) / 50
    summary["full_inversion_probe"] = {"rows": full_rows, "sec_per_step": per_step,
                                       "est_minutes_raw": per_step * args.steps / 60}
    print(json.dumps(summary["full_inversion_probe"], indent=1))

    summary["two_stage_on_pilot"] = {k: {"stage1": int(v["stage1"].sum()), "both": int(v["final"].sum())}
                                     for k, v in two.items()}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1, default=float)
    np.save(os.path.join(args.out, "orig_phys.npy"), orig_phys)
    print(json.dumps({k: summary[k] for k in summary if k not in ("domain_normalized",)}, indent=1,
                     default=float))


if __name__ == "__main__":
    main()
