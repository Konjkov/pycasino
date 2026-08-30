name: vmc-efficiency
description: >
  Use this skill when working on VMC sampling efficiency in PyCasino: the VMC time step
  (theory, sum rule, Thomas-Fermi scaling, non-Gaussian corrections), the acceptance-ratio
  target (RGG theorem, CBCS vs EBES differences), the decorrelation period (AR(1) thinning),
  and CBCS vs EBES proposals. Covers the theory (Lee/Conduit/Nemec/Lopez Rios/Drummond,
  Roberts-Gelman-Gilks), PyCasino's implementation choices, empirical measurements (Series A-F),
  and where CASINO's implementations are weak.

# VMC Efficiency: Theory, Implementation, and Empirical Validation

Three coupled choices govern how fast a VMC run reaches a given error bar: the time step,
the decorrelation period, and CBCS vs EBES. This file collects the theoretical foundation,
PyCasino's implementation logic, and the empirical measurements that validate them.

## 1. The Efficiency Functional
From Lee et al., Phys. Rev. E 83, 066706 (2011).
Statistical error `Δ = σ₀·(n/n_corr)^(-1/2)`, run time `T = n·T_iter`.
Efficiency `E = 1 / (Δ² · T) = 1 / (σ₀² · n_corr · T_iter)` is independent of `n`.
With a decorrelation loop of length `p`:
`T_iter(p) = p·T_move + T_energy`
`n_corr(p) = (1+ρ^p)/(1−ρ^p)` (AR(1) thinning law, where `ρ = (τ−1)/(τ+1)`).

*Implementation note:* An implementation detecting "all `p` moves rejected" can skip the energy 
evaluation: `T_iter = p·T_move + [1−(1−a)^p]·T_energy`.

## 2. Decorrelation Period

### 2.1 PyCasino Implementation & CASINO Defects
`VMC_DECORR_PERIOD = 0` triggers `Casino.optimize_decorr_period`.
It measures the correlation time of the production block (`correlation = (energy_sem/energy_std)²`),
inverts it through the AR(1) law to find `ρ`, and minimizes `T_iter(p)` capped at 100.
`T_move`/`T_energy` are timed on the last block only (excluding JIT compilation).
The skip factor `1−(1−a)^p` is deliberately omitted in PyCasino because we measure `T_energy` 
at the operating `d`, where the factor is ≈ 0.87, unlike CASINO which measures it at `d=1`.

**Defects in CASINO's implementation:**
1. **Measured on too few moves:** CASINO measures on 200–500 moves (`vmc.f90:277`). 
   For multi-electron systems, the statistical error `σ_τ` is huge (e.g., ≈21 on Be). 
2. **Timer failures reported as `1`:** `vmc.f90:339` sets `corper=1` if 
   `time_energy == 0` (a single-precision timer difference over 500 moves). 
   Diagnostic: `DEBUG_OPT_CORPER` at `vmc.f90:38`.

### 2.2 Precision of τ and Measured Values
For an AR(1) series, `σ_τ = τ^(3/2) / sqrt(n)`. The windowed integrated estimator used by 
CASINO is `sqrt(12) ≈ 3.5` times noisier than the AR(1) estimator.
±0.5 is over-precision. The optimum is flat; being off by a factor of two costs 9–25%. 
A sane spec is ±25–50% on τ, which needs only ~2·10⁴ samples.

**Measured values in this project:**
CBCS, Slater–Jastrow, `vmc_decorr_period : 10`, 10⁶ steps: 
residual τ_d = 1.31 (He), 3.54 (N), 4.26 (Be) → τ_raw = 10.0, 34.4, 41.8.
Lee et al. give CBCS `p_opt` = 8…36. Our computed 20–23 for Be/N sits inside it. 
For EBES, Lee et al. give `p_opt` = 3 almost everywhere (CASINO's default).

## 3. The Time Step and the Acceptance Curve

### 3.1 The Theoretical Split
The step size law splits into two independent steps:
1. **Sampling theory:** `ts₅₀·√⟨T⟩ = √3·erfinv(½) = 0.826` (Gaussian limit, uniform cube proposal).
2. **Quantum chemistry:** `⟨T⟩` scaling. By the virial theorem and Thomas-Fermi theory, 
   `⟨T⟩ ∝ Σ_a Z_a^(7/3)`. Thus, `ts₅₀ ∝ N^(1/2) · z_TF^(−2/3)`, where `z_TF = (ΣZ^(7/3)/ΣZ)^(3/4)`.
   This proves `α = ½` (extensivity) and `β = ⅔` (Thomas-Fermi) without empirical fitting.

### 3.2 The Non-Gaussian Correction
`ts₅₀·√⟨T⟩ = 0.826` assumes `X = ln(Ψ′²/Ψ²)` is Gaussian. It is a scale mixture of Gaussians 
because `T_D` (drift kinetic energy) fluctuates. By Jensen's inequality, the spread of `T_D` 
raises acceptance at a fixed mean, requiring a larger step.
The excess scales as `Var(T_D)/⟨T⟩²`. Since `T_D` is dominated by electrons near nuclei, 
`CV²(T_D) ∝ 1/nuclei`. 

*Empirical validation:* Inverting the measured excess per system confirms the `1/nuclei` scaling:
| System | excess | nuclei | sd(T_D)/⟨T⟩ | CV²·nuclei |
|---|---|---|---|---|
| Be²⁺ | 2.87% | 1 | 0.397 | 0.158 |
| He   | 5.01% | 1 | 0.525 | 0.275 |
| Kr   | 6.32% | 1 | 0.589 | 0.348 |
| C₆H₆ | 1.16% | 6.18| 0.253 | 0.394 |

### 3.3 Empirical Constants and Scaling Laws (Series B & C)
Fitting the prefactor and non-Gaussian correction across 17 systems:
`a = 1.0959·√N·(1 + 0.0295/nuclei)·z_TF^(−2/3)` (rms 1.10%, max 2.93% for Kr).
The fitted 1.096 sits between TF-calibrated values. Errors: `C = 1.0959 ± 0.0090`, `k = 0.0295 ± 0.0102`.

* **Series B (CnHn - Settling α = ½):** Fitting `a = C·√N·(1 + c/N)` gives `C = 0.3533`, `c = 0.359`. 
  The sum rule predicts `C = 0.3525` (0.23% error). Both exponent and prefactor are derived.
* **Series C (Ions - N=2 isoelectronic):** The failure of the old `√N·z_TF^(−2/3)` form (which assumed N=ΣZ) 
  proved that electron count must not appear in the formula for ions. Over 17 systems, the old form 
  was 7.86% rms against 1.10% on the 15 neutrals.

### 3.4 The Shape of the Acceptance Curve
The acceptance curve is genuinely non-Gaussian. For a symmetric proposal at stationarity, 
`X` obeys `P(−x) = e^x P(x)`. The curve is well-described by:
`p(ts) = C / (C + exp(u) − 1)`, where `u = ts·N/ts0`.
*Empirical fit:* This 2-parameter form fits the CBCS data 4–8× better than a Gaussian model 
(rms residual 0.0011–0.0022 vs 0.0034–0.0090).

### 3.5 Code Implementation (`approximate_step_size`)
The CBCS branch calculates the Thomas-Fermi kinetic energy, applies the `1/nuclei` 
non-Gaussian correction, and returns the 50% anchor step. `optimize_vmc_step` then probes 
10 step sizes, fits the 2-parameter curve, and solves for the target acceptance.

## 4. Which Acceptance to Target?

### 4.1 The RGG Theorem & CBCS vs EBES
For a product target in `d → ∞`, the rescaled process converges to a Langevin diffusion 
with speed `h(ℓ) = ℓ²·a(ℓ)`. Maximizing `h` gives an optimal acceptance of `0.234`.
50% is a 1D intuition; in high dimensions, the optimum sits deep in the low-acceptance tail.

* **CBCS:** Approximately satisfies RGG hypotheses. The true optimum is below 50%. 
* **EBES:** Fails RGG hypotheses. The acceptance in EBES does not collapse as the step grows. 
  The `D`-max runs away to absurd steps. The true efficiency optimum for EBES is measurably 
  *above* 50% (typically 0.70–0.80).

### 4.2 Empirical Sweeps (Series E & F)
* **Series E (EBES optimum):** Swept 19 targets over 8 systems. EBES optimum is 0.69–0.80, 
  drifting up with the depth of the core. The 50% rule costs a factor 1.9–2.9. 
  The profile (m4) moves the optimum down to 0.58–0.68 by equalizing per-electron acceptance.
* **Series F (The profile ceiling):** The exact profile `⟨|∇lnΨ|²|r⟩` buys only ~1% efficiency 
  at the optimum, but the formula degrades severely off-optimum. 
  The two-parameter formula in the code: `I(r) = min(Z², (0.815·Z^⅓ / r)²) + 1.256²`.

### 4.3 PyCasino Implementation
PyCasino uses a module constant `ACCEPTANCE_TARGET = 0.70`. 
This is a minimax compromise chosen because the efficiency basin is flat, and a single constant 
must serve both methods. `optimize_vmc_step` maps the initial 50% guess to this target via:
`step *= (erfinv(1−ACCEPTANCE_TARGET)/erfinv(1−acceptance))²`.

## 5. CBCS vs EBES: The EBES Step Size

### 5.1 Theoretical Derivation
EBES moves one electron. `Var(X|R,i) = 4(ts²/3)·|∇ᵢlnΨ|²`. 
Averaging over `i` gives `⟨|∇ᵢlnΨ|²⟩ = 2⟨T⟩/N`.
Thus, `ts₅₀(EBES)·√(⟨T⟩/N) = 0.826`, meaning `ts₅₀(EBES) = √N · ts₅₀(CBCS)`.

### 5.2 Empirical Measurements (Series D)
Measured 31 systems. Prediction: `ts₅₀(EBES) = √N · ts₅₀(CBCS)`.
| System | N | measured ratio | √N |
|---|---|---|---|
| H-pp | 1 | 1.00 | 1.00 |
| He   | 2 | 1.42 | 1.41 |
| Be   | 4 | 3.23 | 2.00 |
| Ne   | 10| 5.56 | 3.16 |
| Ar   | 18| 9.17 | 4.24 |
| Kr   | 36| 14.35| 6.00 |

*Conclusion:* `√N` holds only for one-shell systems. The excess tracks the number of occupied shells.
Kurtosis confirms it: −0.33 on one-shell systems vs 4.87 on Ar, 4.63 on Kr.

### 5.3 Why the Gaussian Assumption Fails for EBES
The sum rule fixes the *mean* variance, but acceptance is a nonlinear functional of the variance. 
At a step where the average electron sits at 50%, a core electron is rejected almost always, 
and a valence one almost always accepted. This creates a mixture of normals with different 
variances (high kurtosis). The correction grows with the number of occupied shells, not with N.

## 6. Open Theoretical Questions
1. What is the exact log-ratio distribution `P(X)` that yields `odds ∝ e^u − 1` for CBCS?
2. For CBCS, is the 2-parameter acceptance curve form exact, or only an excellent approximation?
3. Does the exact Thomas-Fermi gradient expansion close the remaining 1% residual in the 
   kinetic energy scaling?
4. Is there a better cheap proxy than `D` for CBCS optimization (e.g., mean square change of `E_L`)?
5. For EBES, is solving `E_i[2Φ(−σᵢ/2)] = ½` the accepted way to state the time step, 
   and has the shell-structure dependence been formally reported in literature?

## 7. Code, Data, and Script Pointers

### PyCasino Code
* `casino/pycasino.py`: `ACCEPTANCE_TARGET`, `approximate_step_size`, `optimize_vmc_step`, 
  `optimize_decorr_period`, `vmc_energy_accumulation`, `vmc_step_graph`, `vmc_corr_graph`.
* `casino/vmc.py`: `vmc_step_profile` (method 4), `simple_random_step` (CBCS), 
  `one_electron_step` (EBES), `gibbs_random_step`.
* `casino/slater.py`: `SlaterState.ratio_1e`, `accept_1e` (EBES determinant updates).
* `casino/sem.py`: `correlated_sem` (pyblock reblocking).

### Experimental Data & Scripts
* Time step data: `examples/time_step/{CBCS,EBES,Biased}/*.dat`, `fit.log`, `single_atom.dat`.
* Efficiency sweeps: `examples/step_profile/` 
  * `acceptance.py` (Series E campaign), `acceptance_report.py`
  * `tabulated.py` (Series F, measured-profile experiment)
  * `casino_scan.sh` (CASINO control)
* Results directories: `examples/step_profile/acceptance_1e6/`, `casino_{ne,ch4,o3}/`, `tabulated/`.

### CASINO Source References
* `vmc.f90`: `eff_estimate`, `DEBUG_OPT_CORPER`, `equilibration` (corper calibration).
* `numerical.f90`: `correlation_time`, `correlation_time_alt`.
* `esdf_key.f90`: keyword documentation.

### Papers & References
* `pdfs/casino_recomendations.pdf` (Lee et al. 2011, central reference).
* `pdfs/casino_overview.pdf` (Needs et al., JCP 152, 154106 (2020)).
* Roberts, Gelman & Gilks (1997) - RGG optimal scaling theorem.