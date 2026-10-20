# Inverse fitting BioSkin to ISSA: feasibility report, pilot and full run

The analysis log, in the order it was done. See `README.md` for how to run it.
- Code: `inverse_pilot.py` and `make_report_figs.py` (pilot); `full_run.py`, `offset_scan.py`, `full_figs.py`
  and `specular_run.py` (full run); `paper_numbers.py` (paper).
- Outputs, including the figures embedded below: `out/` and `full_out/`. They are not committed; run the
  scripts to regenerate them.
- Their code and reference: `fetch_upstream.py` downloads them unmodified into `upstream/`.
- This analysis does not modify `bioskin/`.

**Section 7 (specular term) supersedes section 6.4; section 6 supersedes the pilot numbers where they differ.** Sections 1–5 are the original
feasibility pass, kept as run.

## Verdict (updated after the full run)

1. **Reconstruction.** On all 7,923 face spectra (2,095 participants), BioSkin fits at a **median RMSE of
   0.90 pp** (0.87 pp with the specular term of section 7). Only melanin, blood and thickness are free; ratio
   and oxygenation are fixed. It is *not* comparable to either BVCM number.
   The BVCM's 1.43 pp is a population prediction from MI/EI; its 0.20 pp is in-sample with 10 free deviation
   coefficients per spectrum.
2. **The 18.3% headline measures where the decoder is sampled.** Their unchanged `evaluate()` passes the same
   decoder at 18%, 31% or 87% depending only on the sampling distribution (section 4). The 87% figure is
   partly circular and is an upper reference only.
3. **Their test does not accept real dark skin.** On the real spectra, 0 of 15 EH (13 participants) and
   71% of EMIH (participant-bootstrap CI 62–80%) pass, against 94% for EMIL. The acceptance region is
   calibrated on a reference that is 81% EMIL, so it measures closeness to the typical ISSA face.
4. **The thickness inflation was mostly a specular mismatch, and it is now resolved (section 7).** ISSA is d/8
   with specular included (SCI). BioSkin's R_d is subsurface only: this is confirmed from the training data and
   from the code. Adding near-normal Fresnel, R = F + (1 − F)·R_d with F = 0.028 (n = 1.4):
   - the fit **improves** (0.90 → 0.87 pp);
   - median thickness falls 130 → 100 µm (EMIH 262 → 132; EH 332 → 121);
   - melanin lands in the literature ranges (EMI 15%, EMIH 22%);
   - the blood and thickness ceiling artefacts mostly vanish.

   What remains: the darkest skin now saturates melanin at the 0.43 training ceiling (8 of 15 EH).
5. **The paper's 18.3% has the same mismatch built in.** Their test scores diffuse BioSkin spectra against SCI
   measurements. Adding the specular term to the *same* 4,000 spectra raises the pass rate to **29.2%** (F = 0.028;
   33.6% at F = 0.040).
6. **Raw decoder output is rougher than skin.** This is a BioSkin weakness, to disclose: raw fits pass stage 2 far
   less often than smoothed fits of the same quality.

## 1. Repo audit

| question | answer |
|---|---|
| Encoder input | **RGB only** (`Skin2ReflectanceNN.fc_enc_in = Linear(D_rgb=3, …)`). The encoder takes no spectrum. |
| Decoder | 5 → 620 → 620 → 310 MLP (tanh, sigmoid out), 380–998 nm at 2 nm. **Fully differentiable**. Gradients via autograd work out of the box. Use non-AO `BioSkin` (the AO model's SAM loss ignores magnitude). |
| Wavelengths | All 31 ISSA wavelengths fall exactly on the decoder grid (indices 10, 15, …, 160). No interpolation is needed. |
| Training domain | From the curated MC parameter tensor `datasets/*_params_test_*.pt`. Melanin 0.001–0.429, blood 0.001–0.200, thickness 10–350 µm, melanin type ratio 0.650–0.800, oxygenation 0.900–1.000. The `train/` tensors are not on disk. The test file's name encodes the same bounds, so I treat it as the domain. This box is the only constraint used. |

Order caveat (already known): slot 3 = melanin type ratio and slot 4 = oxygenation. `check_decoder_inputs()` guards this.

**ISSA data.** It is not in their repo (their `data/` is gitignored), but it is **not access-restricted**: it
is the public figshare release (doi:10.6084/m9.figshare.28228571.v4, CC BY 4.0), already at
`analysis/issa/`. Its SHA-256 matches the workbook hash in their `summary.json` exactly. Their cleaning reproduces 7,923 face spectra.

**Their validation code.** `code/check_bioskin_rerun.py::evaluate()` plus their saved
`output/corrected_validation/reference_state.npz` (stage-1 cutoff 2.4477, stage-2 cutoff 9.41141). I import
`evaluate` unchanged. Fed my re-decode of the shared corrected package, it returns their published
**1,167 / 733** exactly, and my decoder reproduces the shared spectra to within 2e-5 pp.

## 2. Method: gradient fitting through the decoder (recommended)

- Loss: MSE in reflectance % on the 31 ISSA wavelengths only, with no padding and no resampling.
- Parameters live in the unit box mapped onto the training domain. Projected Adam (lr 0.05, cosine decay) runs with
  **K = 32 Sobol-scrambled restarts per spectrum**, all spectra and restarts batched on the GPU.
- Primary fit uses the raw decoder. A second fit includes the double Savitzky–Golay (15, 3) used for the shared packages, as a
  fixed 310×310 linear map.
- Convergence: median best RMSE is identical at 1,500, 3,000 and 6,000 steps (0.7226 pp).
  16 restarts reach the 32-restart optimum on 99.5% of spectra.

Why not the encoder: it sees 3 numbers for 5 parameters, and it needs ISSA extrapolated to 380–780 nm and
projected to RGB. On the pilot it is **2–3× worse** (median RMSE 0.8–2.0 pp by class). It also returns
out-of-domain parameters on every spectrum: oxygenation 0.39–0.59 against a trained 0.90–1.00, and melanin ratio down to 0.30. It is
not usable for inversion, and not even as an initialiser.

## 3. Pilot (200 spectra, MI-stratified, seed 20261019)

All 15 EH spectra (the darkest in ISSA, MI up to 114) plus 45/50/45/45 random from EML/EMIL/EMI/EMIH.
Population summaries below are reweighted to ISSA's class mix (626/6,419/734/129/15).

### Reconstruction

| class | n | MI | RMSE med (p90, max) pp | rel. RMSE | smoothed fit | encoder | thick ≤100 µm |
|---|---|---|---|---|---|---|---|
| EML | 45 | 20–25 | **1.33** (1.75, 2.08) | 3.6% | 1.29 | 2.00 | 1.58 |
| EMIL | 50 | 25–50 | **0.93** (1.39, 2.13) | 3.3% | 0.90 | 1.93 | 1.21 |
| EMI | 45 | 51–73 | **0.52** (1.14, 1.61) | 4.1% | 0.48 | 0.86 | 1.33 |
| EMIH | 45 | 75–99 | **0.37** (0.65, 1.21) | 4.6% | 0.36 | 0.83 | 1.61 |
| EH | 15 | 101–114 | **0.32** (0.54, 0.62) | 5.3% | 0.25 | 0.80 | 1.62 |

Reference points: BVCM population level 1.43 pp (test); BVCM plus per-spectrum GP deviation 0.20 pp.
Absolute error falls with MI because reflectance falls; relative error is flat to slightly rising. MI and
EI of the fits match the data to a median of 0.6 and 0.8 index units.

![](out/fig1_reconstruction_by_class.png)

Misfit is structured, not noise. For light skin the decoder overshoots 460–510 nm by up to +2.3 pp
and undershoots the 600–650 nm red edge by up to −2 pp. The model cannot make the red edge as steep as real light facial
skin while keeping the blue right.

![](out/fig2_example_fits.png)

### Fitted parameters vs. the original sample and the training domain

ISSA-weighted 5th / 50th / 95th percentiles of the best fits:

| parameter | pilot fits | original 4,000 sample (median) | comment |
|---|---|---|---|
| melanin fraction | 0.018 / **0.038** / 0.079 | uniform 0.001–0.43 (0.21) | 81% of the original draws exceed the ISSA p95 of 0.079 |
| blood fraction | 0.004 / **0.012** / 0.046 | uniform 0.001–0.20 (0.10) | 77% of the original draws exceed the ISSA p95 |
| thickness µm | 78 / **130** / 253 | uniform 50–250 (152) | only 15% of fits are ≤100 µm |
| melanin type ratio | bimodal at 0.65 / 0.80 | uniform 0.65–0.80 | not identifiable |
| oxygenation | mostly at 0.90 / 1.00 | uniform 0.90–1.00 | not identifiable |

Only **4.1%** of the original 4,000 draws have *both* melanin and blood inside the ISSA 95th percentiles.

![](out/fig3_parameter_histograms.png)

### Literature comparison and the melanin / thickness offset

- **Thickness.** The fits put facial epidermis at 130 µm median, and at 210 → 265 → 337 µm for EMI → EMIH → EH.
  That is far above the ≤100 µm optical-method benchmark, and it rises with MI, which epidermis does not.
- **Melanin.** Free-thickness fits give 2.1% (EML), 3.8% (EMIL), 7.4% (EMI), 9.5% (EMIH) and 13% (EH). The commonly quoted
  epidermal melanosome fractions are about 1.3–6.3% (light), 11–16% (moderate) and 18–43% (dark) (Jacques, OMLC;
  please verify the citation). So for darker skin, **melanin is under-estimated and thickness over-estimated**:
  the decoder gets the needed attenuation from path length rather than concentration.
- **Refit with thickness ≤100 µm.** Melanin moves up (EMI 14%, EMIH 43%), much closer to the literature, but
  the fit fails. RMSE goes to 1.2–1.6 pp in every class, 29 of 45 EMIH fits pin melanin at the 0.43 training
  ceiling, and two-stage passes drop from 119 to 67 of 200. A 250 µm cap (your original range) costs little: the median rises from
  0.37 to 0.40 pp for EMIH and from 0.32 to 0.48 pp for EH.

So BioSkin "thickness" should not be read as anatomical epidermal thickness. This is the most important
interpretive flag for the paper. Within this decoder, melanin and thickness are partly exchangeable (fig. 4).

![](out/fig4_melanin_thickness.png)

### Non-uniqueness and bounds

- **Restarts.** Light skin has a single basin (median 1 distinct solution within 0.05 pp of best). Darker skin has several:
  median 2 (EMI), 5 (EMIH) and 3 (EH). Across those basins melanin barely moves (≤1% of its domain). Blood, thickness and
  ratio/oxygenation are what change.
- **Local identifiability** (linearised range keeping RMSE within 0.5 pp, as a fraction of the domain): melanin
  ±3–8%; blood ±6% for light skin rising to ±66% for dark; thickness ±7% rising to ±43%; ratio and oxygenation
  **±100%** in every class.

  ![](out/fig5_identifiability.png)
- **Bound hits** (best fit within 0.1% of a domain edge). Melanin: 0/200. Blood: 22 (17 at the 0.20 upper
  bound, almost all EMIH). Thickness: 11 at the 350 µm upper bound, 6 of them among the 15 EH. Ratio: 152 and oxygenation: 151.
  Those two are flat directions, so they are meaningless rather than domain-limited. On the meaningful parameters, **29 of 200 fits hit a
  bound, concentrated in MI > 75**. The darkest ISSA skin sits at the edge of the decoder domain.

### Do the fits pass their two-stage test?

| class | ISSA observed | raw fit | smoothed fit | thickness ≤100 µm fit |
|---|---|---|---|---|
| EML (45) | 40 | 31 | 39 | 25 |
| EMIL (50) | 47 | 49 | 49 | 41 |
| EMI (45) | 43 | 38 | 45 | 1 |
| EMIH (45) | 30 | **1** | 19 | 0 |
| EH (15) | **0** | 0 | 0 | 0 |
| total | 160 | 119 | 152 | 67 |

Two findings matter for the paper:

1. **Not one real EH spectrum passes their test** (5 of 15 clear stage 1, 0 clear both). The framework cannot
   accept the darkest skin in its own reference data, so no simulated dark spectrum can pass either.
2. For EMIH, raw fits at 0.37 pp RMSE pass 1 of 45, against 30 of 45 for the spectra they fit. Stage 2 is rejecting
   decoder fine structure (the roughness issue in `analysis/section_4_3_revision.md`), not colour. Smoothing recovers most of it.

## 4. Forward sanity checks (small batches, their `evaluate()` unchanged)

| candidate set | n | stage 1 | both, raw decoder | both, SG-smoothed |
|---|---|---|---|---|
| real ISSA face spectra | 7,923 | — | 93.3% (no decoder) | — |
| original corrected sample (uniform physical) | 4,000 | 30.7% / 29.2% | 11.4% | **18.3%** (= paper) |
| Sobol over full training domain\* | 4,096 | 47.0% / 45.9% | 18.2% | 31.4% |
| empirical KDE of pilot fits (ISSA-weighted) | 4,000 | 88.8% / 88.3% | 77.1% | **86.5%** |

\* Sobol is uniform in decoder-normalised space (melanin^⅓, blood^¼), which already favours lighter skin than uniform physical sampling.

The same decoder goes from 18% to 87% depending only on where it is sampled. The empirical row is partly
circular, because its parameters were fitted to ISSA. It is an upper reference, not a validation result. Both pipelines
run end to end.

## 5. Wall-clock estimates (RTX 5090, measured)

| run | estimate |
|---|---|
| Pilot: 200 spectra × 32 restarts × 3,000 steps | 15 s per variant (measured) |
| Full inversion: 7,923 × 32 restarts (253k rows) | 0.050 s/step → **~1.3 min** at 1,500 steps, 2.5 min at 3,000 |
| Full set of variants (raw, smoothed, 250 µm cap, 100 µm cap) plus Jacobians | **< 15 min** |
| Forward decode + their evaluate, 100k samples | **< 1 s** compute; writing a 100k × 310 CSV (~0.4 GB) dominates at ~1 min |

## 6. Full run: all 7,923 face spectra

Configuration, as agreed:
- **primary:** full training domain; melanin type ratio and oxygenation fixed at 0.725 / 0.95; SG-smoothed decoder output.
- **sensitivities:** raw output; all 5 parameters free; thickness capped at ≤250 µm and at ≤100 µm.
- **fitting:** K = 16 restarts, 1,500 steps.

Each variant took about 32 s on the RTX 5090, about 3 min end to end. Summaries are by MI class. Confidence
intervals come from resampling participants (origin dataset × subject number; 2,095 participants).

![](full_out/fig6_full_run.png)

### 6.1 Their test on the real ISSA spectra

| class | spectra | participants | stage 1 | both stages | CI (participant bootstrap) |
|---|---|---|---|---|---|
| EML | 626 | 360 | 95.0% | 89.8% | 86.9–92.3 |
| EMIL | 6,419 | 1,930 | 98.5% | 94.2% | 93.4–95.0 |
| EMI | 734 | 375 | 98.0% | 94.8% | 93.1–96.5 |
| EMIH | 129 | 74 | 92.2% | **71.3%** | 62.3–79.7 |
| EH | 15 | 13 | 33.3% | **0.0%** | 0–0 |

Restricting to the BVCM's own training rows gives the same numbers (EMIH 71.6%, EH 0%). The test rejects
dark spectra it was fitted on. Ten of the 15 EH spectra fail stage 1, because the GMM ellipses do not reach MI > 100.
Overall, 93.3% of real spectra pass.

This is the expected behaviour of a density-based acceptance region on a reference whose dark tail is sparse,
which is why I would frame it as **coverage**, not as a bug. The consequence is the same: neither real nor
simulated dark skin can pass.

### 6.2 Reconstruction

| class | primary (smoothed) | participant-level median | raw | all 5 free | cap 250 µm | cap 100 µm |
|---|---|---|---|---|---|---|
| EML | 1.49 | 1.49 | 1.51 | 1.30 | 1.49 | 1.63 |
| EMIL | 0.90 | 0.89 | 0.95 | 0.82 | 0.90 | 1.20 |
| EMI | 0.54 | 0.66 | 0.59 | 0.49 | 0.54 | 1.39 |
| EMIH | 0.38 | 0.41 | 0.40 | 0.36 | 0.43 | 1.67 |
| EH | 0.31 | 0.31 | 0.34 | 0.25 | 0.55 | 1.71 |
| all | **0.90** | | 0.95 | 0.82 | 0.91 | 1.29 |

Values are median RMSE in pp. Participant-bootstrap CIs on the primary medians are narrow (±0.02–0.06 pp).

- **Fixing ratio and oxygenation costs 0.02–0.19 pp, mostly for light skin.** Left free, they sit on a bound in
  87% (ratio) and 83% (oxygenation) of fits. So they soak up a little structured misfit, but are not identified.
- **The 250 µm cap** is neutral up to EMI, and costs 0.05 pp for EMIH and 0.24 pp for EH. The darkest spectra
  are limited by the decoder domain.
- **MI and EI** of the fits match the data to a median of 0.73 and 0.41 index units.
- **Non-uniqueness is small once ratio and oxygenation are fixed.** The mean number of distinct near-optimal
  basins is 1.1 for EML/EMIL, 1.5–1.8 for EMI/EMIH and 2.1 for EH. Melanin varies by under 1% of its range across
  them; blood varies by up to 23% of its range for EH.

### 6.3 Their test on the fitted spectra

| class | real ISSA | primary (smoothed) | raw | cap 250 µm | cap 100 µm |
|---|---|---|---|---|---|
| EML | 89.8% | 93.1% | 77.5% | 93.1% | 93.3% |
| EMIL | 94.2% | 98.4% | 97.1% | 98.6% | 98.2% |
| EMI | 94.8% | 97.8% | 73.2% | 99.9% | 43.5% |
| EMIH | 71.3% | 48.1% | **5.4%** | 67.4% | 46.5% |
| EH | 0.0% | 0.0% | 0.0% | 40.0% | 0.0% |
| all | 93.3% | 96.9% | 91.6% | 97.6% | 91.7% |

- **Raw vs smoothed.** Same-quality fits drop from 48% to 5% for EMIH and from 98% to 73% for EMI when not
  smoothed. Raw decoder output is rougher than real skin (residual rms second difference 0.054 vs 0.020), and
  stage 2 charges that roughness to a very small noise budget. This is a BioSkin limitation, to disclose.
- **Better fits do not always pass more.** The cap-250 fits are *worse* for EH, yet 40% pass, against 0% for
  primary. They pass because their MI/EI drift into the GMM ellipses. This is another sign that pass/fail tracks
  closeness to the typical face rather than fidelity to the measured spectrum.

### 6.4 Fitted parameters, and why thickness is not interpretable yet

Primary fits, 5th / 50th / 95th percentiles over all spectra:
- **melanin** 0.019 / 0.044 / 0.082; by class 2.4%, 4.2%, 7.4%, 9.4%, 12.7%.
- **blood** 0.004 / 0.011 / 0.046.
- **thickness** 68 / 130 / 264 µm; by class 101, 126, 203, 262, 332 µm. Only 22% of fits are ≤100 µm.

Several knobs are compensating rather than measuring:
- **Blood.** Half of the EMIH fits (65 of 129) put blood at the 0.20 ceiling, which is not physiological.
- **Thickness.** 89 fits sit at the 350 µm ceiling.

**Measurement-model mismatch.** Every ISSA spectrum is SP/SCI: spectrophotometer, specular component *included*.
The BioSkin packages are labelled "diffuse". If the decoder output carries no surface (Fresnel) reflection, every
ISSA spectrum contains a roughly wavelength-flat floor that BioSkin cannot produce. That floor is a large share
of dark-skin reflectance, which is 3–10%.

Tests on the pilot 200:

| probe | effect |
|---|---|
| free multiplicative gain, per spectrum | improves RMSE (EMIL 0.98 → 0.64), but the gain is erratic across classes (0.90–1.50, pinned at 1.5 for EMIH). EMIH/EH thickness is unchanged (260 / 316 µm). **Not the explanation.** |
| free gain + linear tilt | further RMSE gain, uninterpretable (gain 0.68–1.73 at 400 nm). **Too flexible to test anything.** |
| free additive floor, per spectrum | EMI: thickness 198 → 85 µm, melanin 7.6% → 18%, RMSE 0.53 → 0.33. EMIH: 265 → 93 µm, 9.4% → 29%, RMSE 0.37 → 0.32. The floor is 2.9–3.2 pp for those classes, but pins at 6 pp for EML (absorbing other misfit there). |
| **single global floor 0–5 pp** (`offset_scan.csv`, fig. 6c) | overall RMSE is nearly flat from 0 to 2.5 pp (0.72 → 0.75). Over that span thickness falls (overall 167 → 119 µm; EMI 198 → 110; EMIH 265 → 162; EH 332 → 170) and melanin rises (EMIH 9% → 19%; EH 13% → 27%). It also removes the EMIH blood-at-ceiling artefact (0.20 → 0.002). |

Conclusions:
- The spectra **cannot determine** the thickness / melanin split while a specular floor of unknown size is
  unmodelled. They are nearly indifferent to a 0–2.5 pp floor.
- The real ISSA data cap a uniform floor at about 2.2 pp: 0.1% of spectra are below 2.17% at 400 nm. Normal-incidence
  Fresnel for n ≈ 1.4–1.55 is 2.8–4.6%, so a floor of roughly 2 pp is physically plausible.
- At a 2–2.5 pp floor, melanin for the darker classes lands in the commonly quoted 11–16% (moderate) to 18–43%
  (dark) ranges. Thickness drops a lot but stays above 100 µm for MI > 50.

So the specular floor is a **third candidate explanation**, alongside melanin absorption scaling (ln 10) and
missing carotenoids. It is the only one that can be tested without the Monte Carlo code: check whether the MC
reflectance includes the surface Fresnel term.

## 7. The specular term (resolves section 6.4)

### Evidence that BioSkin R_d excludes specular and ISSA includes it

- **ISSA.** Konica Minolta CM-2600d / CM-700d, CIE d/8, specular component included (SCI), per the ISSA
  paper (Lu et al., Scientific Data 2025).
- **The BioSkin training data has no Fresnel floor.** Many training spectra fall well below 2.8% in
  400–700 nm, which is impossible if a ~2.8% surface term were included.
- **The BioSkin code treats specular as a separate post-process.** `bioskin/lighting/light_transport.add_specular_reflectance`
  is called only by `apps/manifold_sampling.py`, which writes it to a separate `..._diffuse_and_specular` output.
  The shared packages are the diffuse output.
- **The original BioSkin simulation agrees.** It used a cosine-weighted source with no entry Fresnel; the
  specular term is carried by a separate BRDF lobe rather than by the subsurface integral. The convention
  for measured SCI data is to add the model's own surface term, never to subtract a constant from the
  measurement.

**Which F.** d/8 is 8° incidence by reciprocity, so the surface term is near-normal Fresnel: 0.028 at n = 1.4,
0.034 at n = 1.45, and 0.040 at n = 1.5 (BioSkin's own R0).

BioSkin's own `add_specular_reflectance` is the wrong term here, for two reasons:
- It returns a **hemispherical** average, F̄ = 0.106, which is right for a renderer under diffuse light but
  not for d/8. It would put a 10.6% floor under spectra that go down to about 2%.
- Its `integrate_fresnel` weights by cos θ·sin θ while sampling uniformly in cos θ. That counts sin θ twice; the
  correct cosine-weighted average is 0.086. **This is a side bug in core BioSkin, not fixed here.**

### Full inversion with R_SCI = F + (1 − F)·R_d

Script: `specular_run.py`; table: `full_out/specular_runs.csv`. The fit equals the primary config otherwise
(full domain, ratio and oxygenation fixed, smoothed, all 7,923 spectra).

| | F = 0 (sec. 6) | **F = 0.028** | F = 0.034 | F = 0.040 | F = 0.028, thickness ≤ 100 µm |
|---|---|---|---|---|---|
| median RMSE, pp | 0.902 | **0.870** | 0.876 | 0.891 | 0.971 |
| median thickness, µm | 130 | **100** | 93 | 88 | 98 |
| fits ≤ 100 µm | 22% | **50%** | 58% | 64% | 100% |
| thickness at 350 µm ceiling | 89 | 21 | 15 | 5 | — |
| blood at 0.20 ceiling | 99 | 13 | 13 | 21 | 10 |
| melanin at 0.43 ceiling | 0 | 71 | 118 | 166 | 142 |
| fits passing their test | 96.9% | 97.6% | 97.7% | 97.6% | 97.4% |
| ISSA spectra below F somewhere | — | 18 | 33 | 61 | 18 |

By class at F = 0.028, compared with F = 0:

| class | RMSE pp | melanin | thickness µm | melanin at ceiling | fits passing |
|---|---|---|---|---|---|
| EML | 1.49 → 1.46 | 2.4% → 4.2% | 101 → 72 | 0 | 93 → 95% |
| EMIL | 0.90 → 0.87 | 4.2% → 5.7% | 126 → 101 | 0 | 98 → 99% |
| EMI | 0.54 → 0.51 | 7.4% → 15% | 203 → 101 | 19 / 734 | 98 → 98% |
| EMIH | 0.38 → 0.42 | 9.4% → 22% | 262 → 132 | 44 / 129 | **48 → 74%** |
| EH | 0.31 → 0.39 | 12.7% → 43% (ceiling) | 332 → 121 | 8 / 15 | 0 → 0% |

Reading:
- **The specular term removes most of the thickness inflation and improves the fit.** The literature thickness
  cap now costs 0.10 pp (0.87 → 0.97); without the specular term it cost 0.39 pp (0.90 → 1.29).
- **Melanin moves into the literature ranges** (Jacques, OMLC News, January 1998; verified against the source):
  light 1.3–6.3% (EML/EMIL 4–6%), moderate 11–16% (EMI 15%), dark 18–43% (EMIH 22%).
- **The open problem is now the melanin ceiling.** The 0.43 upper bound binds for a third of EMIH and half of
  EH. It is the sampling range of the training set, not a physical limit. In those two classes the fits split
  into two groups, which the spectra cannot tell apart:
  - melanin at the bound with a thin epidermis (median 64 µm for EMIH, 94 µm for EH);
  - lower melanin with a thicker epidermis (153 and 179 µm).
- **The data cannot choose F.** RMSE is flat across 0.028–0.034. F = 0.028 is the principled value (n = 1.4).
- **Remaining convention gap.** BioSkin's R_d was simulated with a cosine-lobe source and no entry Fresnel, while
  d/8 is collimated-8° by reciprocity. The entry geometry matters by itself and cannot be corrected without
  re-simulating. F + (1 − F)·R_d is the standard first-order correction, not an exact one.
- **What stays unexplained.** EML misfit (1.46 pp, structured at 460–510 and 600–650 nm) is barely touched by the
  specular term. The carotenoid hypothesis remains the candidate there.

### Their validation with the specular term added to the candidate spectra

`full_out/forward_with_specular.csv`; same spectra, their `evaluate()` unchanged.

| candidate set | output | F = 0 (as published) | F = 0.028 | F = 0.040 |
|---|---|---|---|---|
| original 4,000 (paper) | smoothed | 29.2% / **18.3%** | 56.0% / **29.2%** | 58.6% / 33.6% |
| original 4,000 (paper) | raw | 30.7% / 11.4% | 55.3% / 21.3% | 57.4% / 24.6% |
| Sobol 4,096 | smoothed | 45.9% / 31.4% | 53.1% / 35.6% | 49.5% / 34.8% |
| Sobol 4,096 | raw | 47.0% / 18.2% | 53.1% / 26.3% | 49.7% / 26.8% |

Each cell is stage 1 / both stages.

**For the paper:** Section 4.4 compares a diffuse-only simulation against SCI measurements. Adding the
measurement's own surface term is a comparison-convention fix, not a tuning choice, and it raises the published
18.3% to 29.2% before any change to sampling.

## 8. Status (2026-10-20)

**Decisions.**
- F = 0.028 is primary in the paper.
- The 0.43 melanin bound is the training sampling range, not a physical limit.
- Melanin absorption is stated explicitly in the paper: μa_eu = 6.6×10¹¹ λ^−3.33 and μa_pheo = 2.9×10¹⁵ λ^−4.75 cm⁻¹
  (Donner & Jensen 2006, as in Aliaga et al. 2023).

**Done.** A replacement Section 4.3 was sent to the co-authors on 2026-10-20. Every number and figure in it is
generated by `paper_numbers.py`. The manuscript material itself is kept out of this repository until publication.

**Still open.**
1. Confirm that the original Monte Carlo used the melanin coefficients directly as Napierian μa (no ln 10).
2. Side bug: `integrate_fresnel` double-counts sin θ (F̄ 0.106 vs 0.086). It only affects renders that use
   `add_specular_reflectance`, not this analysis. Not fixed here because it is core BioSkin.
3. Internal only: the pilot's nearest-Monte-Carlo baseline (`rmse_nearest_mc`) uses the non-shareable training set
   and must not be published.
