# PPE2 — Differential Sparse PPE for Fault Localization

Reproduction and assessment of the "differential sparse PPE" scheme for
improving the localization accuracy of fiber-longitudinal power profile
estimation (PPE / LPM), as discussed in a Gemini dialogue and related to:

* T. Sasai et al., *Linear Least Squares Estimation of Fiber-Longitudinal
  Optical Power Profile*, JLT 2023 (LS-PPE model)
* R. Shinzaki et al. (Fujitsu), *Sparse Modeling Analysis for Efficient
  Localization of Power Anomalies in Transmission Links*, OECC 2024
  (generalized Lasso with a prior-knowledge J-matrix)
* H. Ishihara et al. (NTT), *Robust Fibre Longitudinal Power Monitoring
  with Few Measurements using Two-stage Sparse Regularization*, ECOC 2025

## The scheme under test

1. **Reference stage** (healthy link, plenty of data): average LS-PPE over
   many captures to get a low-noise reference profile `x_ref`.
2. **Fault stage** (few captures): instead of re-estimating the absolute
   profile, solve for the *change* against the reference. Because the RP1
   perturbation model `y ≈ G γ'` is linear in the power profile, the fault
   signature lives in the residual `r = y_fault − G x_ref`.
3. **Sparsity prior**: a single lumped loss makes the *relative* change
   `u = Δx / x_ref` an exact step function, so total-variation (L1 on the
   first difference) regularization collapses the estimate onto one
   change point (generalized Lasso, solved with ADMM along a λ path).
4. **Refinement**: a matched-filter scan of candidate fault positions on a
   100-m grid (RP1 step signatures) refines position and re-fits the loss
   magnitude without Lasso shrinkage bias.

Two corrections vs. the dialogue's formulation were needed in practice:

* You cannot literally subtract measurements `Δy = y_ref − y_fault` taken
  with different transmitted data patterns; the correct differential is
  `r = y_fault − G_fault x_ref` (subtract the reference *profile* through
  the current capture's own perturbation operator).
* A plain first-difference penalty `‖D Δx‖₁` is not the right sparsity
  basis, because `Δx` decays with fiber loss and jumps at EDFAs. Working
  in the relative domain `u = Δx / x_ref` (equivalently the Fujitsu
  attenuation-weighted J-matrix) makes the fault the only change point.

## Simulation

Single-polarization split-step NLSE, 64-GBd 16QAM, 3 × 50 km SSMF
(0.2 dB/km, β₂ = −21.4 ps²/km, γ = 1.3 /W/km), 6 dBm launch, EDFA ASE
plus 18-dB receiver SNR, 8192 symbols per capture. A 1.0-dB lumped loss
is inserted at 72.35 km (off-grid). PPE grid: 1 km.

```
pip install -r requirements.txt
python3 run_experiment.py        # ~2.5 min, writes results/*.png
```

## Results (4 fault trials, single post-fault capture each)

| method                                   | mean abs err | max abs err |
|------------------------------------------|--------------|-------------|
| naive profile subtraction (LS)           | ~40 km       | 75 km       |
| differential sparse (gen-Lasso, 1-km grid)| **0.9 km**  | 1.15 km     |
| matched-filter refinement (100-m grid)   | 1.5 km       | 4.2 km      |

Estimated loss magnitude from the matched-filter refit: 0.85–0.95 dB
(true 1.0 dB).

## Verdict on the dialogue's claims

* **Core idea — correct.** Differential estimation against a low-noise
  reference plus a step-sparsity prior is sound, is essentially the
  published Fujitsu/NTT approach, and in this reproduction improves
  single-capture localization from tens of km (noise-dominated naive
  subtraction) to ~1 km.
* **"50–200 m accuracy, purely noise-limited" — not supported.** The
  matched-filter correlation surface is nearly flat over several km:
  adjacent candidate signatures decorrelate only through chromatic
  dispersion acting on the signal bandwidth, so the CD × bandwidth²
  physics that sets Δz_min still bounds how sharply a step can be
  pinpointed. Published experimental numbers (1–4 km for 0.7–3 dB
  events) and this simulation (~1 km for 1 dB) agree; the claimed
  10–50× improvement over the naive baseline is real, but the absolute
  50–200 m figure is optimistic by roughly an order of magnitude at
  these baud rates.

## Layout

* `ppe/link_sim.py` — split-step link simulator (spans, EDFAs, anomaly)
* `ppe/ppe_core.py` — G-matrix, LS-PPE, ADMM generalized Lasso,
  differential sparse localization, matched-filter refinement
* `run_experiment.py` — end-to-end experiment and figures
* `results/` — generated figures
