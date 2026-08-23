---
name: vmc-efficiency
description: >
  Use this skill when working on VMC sampling efficiency in PyCasino: the VMC time step
  (DTVMC, OPT_DTVMC, approximate_step_size, optimize_vmc_step), the acceptance-ratio
  target, the "50% rule" and the measured 0.70 optimum (ACCEPTANCE_TARGET), the
  position-dependent step profile (VMC_METHOD 4), the decorrelation period (VMC_DECORR_PERIOD,
  optimize_decorr_period), correlation times and their precision, the diffusion-constant
  criterion, and CBCS vs EBES (VMC_METHOD). Covers the theory (Lee/Conduit/Nemec/
  Lopez Rios/Drummond PRE 83 066706, and the Roberts-Gelman-Gilks optimal-scaling
  result), the measured acceptance curves in examples/time_step and the efficiency sweeps
  in examples/step_profile, the empirical step-size
  scaling law with electron count and nuclear charge, and the open questions worth
  putting to Drummond. Also covers where CASINO's own implementations of these are weak.
---

# VMC efficiency: time step, acceptance target, decorrelation period

Three coupled choices govern how fast a VMC run reaches a given error bar: the time step,
the decorrelation period, and (upstream of both) CBCS vs EBES. This file collects the
theory, what has been measured in this project, and what is still open.

## 1. The efficiency functional

From Lee, Conduit, Nemec, López Ríos, Drummond, *Strategies for improving the efficiency
of quantum Monte Carlo calculations*, Phys. Rev. E **83**, 066706 (2011), arXiv:1006.1798.
Local copies: `pdfs/casino_recomendations.pdf`, `pdfs/Alternative sampling/1006.1798.pdf`.

Statistical error `Δ = σ₀·(n/n_corr)^(-1/2)`, run time `T = n·T_iter`, so the efficiency

```
E = 1 / (Δ² · T) = 1 / (σ₀² · n_corr · T_iter)          (their Eq. 2)
```

is independent of `n`. `σ₀` is fixed by the system and the trial function; everything a
sampling algorithm can do lives in the product `n_corr · T_iter`. With a decorrelation
loop of length `p`:

```
T_iter(p) = p·T_move + T_energy                          (Eq. 5)
n_corr(p) = 1 + 2(n_corr−1)^p / ((n_corr+1)^p − (n_corr−1)^p)   (Eq. 9)
```

Eq. 9 is the AR(1) thinning law: assuming `A_k = exp(−αk)` gives
`exp(−α) = (n_corr−1)/(n_corr+1) = ρ`, hence `n_corr(p) = (1+ρ^p)/(1−ρ^p)`.
Footnote [27] adds the refinement that an implementation which detects "all `p` moves
rejected" can skip the energy: `T_iter = p·T_move + [1−(1−a)^p]·T_energy`.

CASINO implements exactly this in `vmc.f90:983` (`eff_estimate`); the line above it,
commented out, is the older naive law `1 + (τ−1)/p`.

## 2. Decorrelation period

### PyCasino implementation

`VMC_DECORR_PERIOD = 0` triggers `Casino.optimize_decorr_period` (`casino/pycasino.py`),
called at the end of `vmc_energy_accumulation`:

- correlation time of the **production block**, `correlation = (energy_sem/energy_std)²`,
  i.e. the square of the ratio of the reblocked error to the naive one — already computed
  for the FINAL RESULT print, so it is free;
- inverted through `ρ = ((τ_d−1)/(τ_d+1))^(1/d)` where `d` is the period the block ran at;
- `d_opt = argmin_d (1+ρ^d)/(1−ρ^d) · (d·T_move + T_energy)`, capped at 100;
- `T_move`/`T_energy` timed on the **last** block only, so the JIT compilation of the
  first block is excluded;
- the answer is broadcast, so it applies to the optimization walk in the same cycle and
  to every later cycle.

The skip factor of footnote [27] is deliberately omitted: we measure `T_energy` at the
operating `d`, where `1−(1−a)^d ≥ 0.87`, unlike CASINO which measures it at `d = 1` where
the factor is ≈ `a` ≈ 0.5.

The inversion is self-consistent — the same `d_opt` comes out whatever `d` the measurement
was taken at:

| | τ at d=1 | d=3 | d=10 | d=20 | → d_opt |
|---|---|---|---|---|---|
| He | 10.0 | 3.42 | 1.31 | 1.04 | 9 |
| N | 34.4 | 11.49 | 3.54 | 1.91 | 20 |
| Be | 41.8 | 13.96 | 4.26 | 2.25 | 23 |

Limitation: for `runtype : vmc` the number is printed but not applied, because the
measurement necessarily follows the block.

### How precisely can τ be known

For an AR(1) series estimated by the lag-1 autocorrelation,
`σ_ρ = sqrt((1−ρ²)/n)`, and `τ = (1+ρ)/(1−ρ)` gives

```
σ_τ = τ^(3/2) / sqrt(n)        =>   n = 4τ³ for σ_τ = 0.5
```

Verified against synthetic AR(1) to two digits (τ=42, n=10⁶: predicted 0.28, measured 0.27).
The windowed integrated estimator with window `W = 3τ` that CASINO uses
(`numerical.f90:4658 correlation_time`, error `τ·sqrt((4W+2)/n)`) is `sqrt(12) ≈ 3.5` times
noisier, i.e. needs 12× the samples.

Cost scales as the **cube** of the answer, so a dedicated calibration run is hopeless:

| | τ_raw | n for σ_τ = 0.5 (AR1) | CASINO's σ_τ at n=2000 |
|---|---|---|---|
| He | 10.0 | 4.8·10³ | 2.5 |
| N | 34.4 | 1.7·10⁵ | 15.7 |
| Be | 41.8 | 3.1·10⁵ | 20.9 |

But ±0.5 is over-precision. The optimum is flat — with `T_energy/T_move = 5`, being off by
a factor of two costs 9–25% (τ=42: 0.5·d_opt → ×1.09, 2·d_opt → ×1.14). A sane spec is
±25–50% on τ, which needs only ~2·10⁴ samples.

Measuring on the thinned production block (10⁶ stored samples at d=10) gives
σ_stat(τ_raw) = 0.04 (He), 0.09 (N), 0.11 (Be) — better than a d=1 measurement on 10⁶ raw
moves (0.11/0.70/0.94), because 10⁶ stored at d=10 is 10⁷ raw moves. These are statistical
only; the single-exponential assumption in the inversion is the larger error, so quote ±20%.

### Two defects in CASINO's version

1. **τ measured on 200–500 moves.** `vmc.f90:277`:
   `nequil_add = max(0.1·nequil, 200/p_move)`, so with the usual
   `vmc_equil_nstep : 5000` it is exactly 500 (visible in the logs as
   "Finding optimal inner loop length (500 additional moves)"). With 4 processes,
   n = 2000. Per the table above σ_τ ≈ 21 on Be. The only knob is `vmc_equil_nstep`;
   reaching σ_τ = 0.5 would need 1.2·10⁵ for He and 8.8·10⁶ for Be. So the estimate
   cannot be fixed by parameters — only by moving the measurement onto the production
   block.
2. **The reported `1` values are timer failures, not measurements.** `vmc.f90:339`:
   `if(any(corr_tau_err<0).or.time_move==0.or.time_energy==0) corper=1`, and
   `time_energy = abs(t/(nequil_add·nprocs) − time_move)` — a difference of two nearly
   equal single-precision CPU times over 500 moves. In
   `examples/stowfn/He/HF/QZ4P/CBCS/Jastrow_emin_eff/out` the two cycles that report `1`
   both print `Done. [total CPU time: 0s]`. So the observed He sequence 1, 4, 8, 10, 1 is
   three measurements scattered around τ≈10 ± 2.5 plus two failures. (Note `corper=1` is
   also the *correct* answer when `T_energy → 0`, which makes the failure invisible.)

Diagnostic: `DEBUG_OPT_CORPER` at `vmc.f90:38` is a compile-time `PARAMETER`; flipping it
and rebuilding prints `corr_tau ± err`, `time_move`, `time_energy` and `Eff(p)` for every
`p`. It is the only way to tell a failure from a measurement.

### Measured values in this project

CBCS, Slater–Jastrow, `vmc_decorr_period : 10`, 10⁶ steps: residual τ_d = 1.31 (He),
3.54 (N), 4.26 (Be) → τ_raw = 10.0, 34.4, 41.8. Lee et al. Table II gives CBCS
`p_opt` = 8 (N pp), 8 (O), 36 (NiO), 13 (N₂H₄), 36 (HEG), 36 (diamond) — so 8…36, and
our computed 20–23 for Be/N sits inside it. For EBES their Table I gives `p_opt` = 3
almost everywhere, which is where CASINO's default of 3 comes from: it is an **EBES**
calibration, applied to CBCS in our runs. `E_(p=1)/E_opt` reaches 0.13, i.e. skipping
decorrelation loops costs up to a factor 7 in CBCS.

## 3. The time step and the acceptance curve

### The empirical form

Found in this project (see `examples/time_step/CBCS/forum.txt`, posted to the CASINO forum
as Vladimir_Konjkov); implemented in `Casino.optimize_vmc_step`:

```
p(ts) = (exp(a/ts0) − 1) / (exp(a/ts0) + exp(ts·N/ts0) − 2)
      = C / (C + exp(u) − 1),   C = exp(a/ts0) − 1,  u = ts·N/ts0
```

equivalently **odds of rejection = (e^u − 1)/(e^(u₅₀) − 1)**. `a` is the value of `ts·N`
at 50% acceptance. Data: `examples/time_step/{CBCS,EBES,Biased}/*.dat`, x-axis is
`step_size × electron count`, columns are STO/GTO × Slater/Jastrow/backflow (the three
wave-function levels agree to <1%, so the curve is a property of the sampler, not of the
correlation factor).

`optimize_vmc_step` probes 10 step sizes over `xdata = linspace(0, 2, 11)` in units of
`approximate_step_size`, fits this 2-parameter form, and **solves** for the target. CASINO
instead bisects on a noisy function. The fit is the better-conditioned approach: it uses
all 10 points and the known global shape, and it makes retargeting free (any point on the
fitted curve, including the diffusion maximum, needs no extra sampling).

### Why the fit is as good as it is

rms residual over 21 points (p from 1 down to <0.05), same number of parameters:

| | this form (2 par) | 2Φ(−s/2) (1 par) | 2Φ(−(cx)^γ/2) (2 par) |
|---|---|---|---|
| He | 0.0021 | 0.0251 | 0.0090 |
| N | 0.0017 | 0.0199 | 0.0084 |
| Kr | 0.0011 | 0.0135 | 0.0066 |
| O3 | 0.0022 | 0.0082 | 0.0034 |

4–8× better than an equally parameterized Gaussian model, so this is not a
reparametrization of the standard result and the log-ratio distribution is genuinely
non-Gaussian.

Route towards a derivation (partly conjecture, flagged as such):

- For a **symmetric** proposal at stationarity, `X = ln(π(R')/π(R))` obeys the exact
  identity `P(−x) = e^x P(x)`. Consequences: `E[e^X] = 1`; the cumulant generating
  function satisfies `K(λ) = K(1−λ)` exactly, so it is minimized at `λ = 1/2`; and
  **`A = 2·P(X > 0)`**. (For Gaussian `X ~ N(−s²/2, s²)` this reproduces `A = 2Φ(−s/2)`,
  the Roberts–Gelman–Gilks form.) So the whole acceptance curve is a functional of one
  scalar distribution that is heavily constrained — which is why so few parameters suffice.
- If `X` is a sum of `N` per-electron contributions, Cramér plus `K(λ)=K(1−λ)` give
  `P(X>0) ≍ F^N/√N` with `F = E[e^(x/2)]`, the **Bhattacharyya fidelity** between the
  one-electron density and its proposal-averaged translate. Hence `ln A` linear in `N`,
  which is what the data show.
- For a cusped/exponentially decaying density, `−ln F` crosses over from `(Zδ)²/2` at small
  displacement to `≈ Zδ` at large — the origin of both the `Z` dependence and the mixed
  linear/quadratic structure. This is the loose end.

**Open**: which `P(X)` reproduces `odds ∝ e^u − 1` exactly, and hence what `ts0` *is*.
This is the question posed on the forum and still unanswered.

### Step-size scaling: N or Z?

For neutral atoms the two cannot be separated, because `N = Z`:

| | He | Be | N | Ne | Ar | Kr |
|---|---|---|---|---|---|---|
| N | 2 | 4 | 7 | 10 | 18 | 36 |
| a = ts₅₀·N | 1.025 | 0.904 | 0.822 | 0.768 | 0.688 | 0.605 |

`a` drifts only 1.7× while `N` changes 18×, which is why `1/(neu+ned)` is a good initial
guess and why `ts₅₀ ∝ N^(-1.18)` for the atomic series. **The two molecules in the data
break the degeneracy**: B2H6 (N=16, max Z=5) has `a = 1.858` versus Ar (N=18) 0.688 — 2.7×
apart at the same electron count. Least squares over all eight systems for
`ts₅₀ = C·N^(−α)·Z_eff^(−β)`:

| Z_eff definition | α (N) | β (Z) | max deviation |
|---|---|---|---|
| max Z | 0.429 | 0.761 | 10.2% |
| **ΣZ²/ΣZ** | **0.534** | **0.646** | **1.8%** |
| mean Z | 0.668 | 0.494 | 14.3% |
| sqrt(ΣZ³/ΣZ) | 0.496 | 0.689 | 2.4% |

So it is **not** the maximum nuclear charge — that is 10% off and loses to the
electron-weighted mean `Z_eff = ΣZ_a²/ΣZ_a`, which fits all eight systems to 1.8% with
`C = 1.161`. Clean exponents `α = 1/2, β = 2/3` give 4.4%.

Two consequences:

- `α ≈ 1/2` is exactly the CLT scaling of the spread of `X` (a sum of `N` per-electron
  terms), i.e. the Roberts–Gelman–Gilks `N^(-1/2)`. **The `1/N` law is an artefact of
  neutral atoms having `N = Z`.** An earlier claim in this project that the data rule out
  `N^(-1/2)` was based on the atomic series alone and is wrong.
- ~~`Z_eff` is nuclear and `N` is the electron count, so the law applies to **ions**
  unchanged.~~ **Refuted by series C, §3.7.** That separation is the *residue* of substituting
  `N = ΣZ` into `N/√(ΣZ^(7/3))`, so it marks the neutrality assumption rather than avoiding it.
  On Be²⁺ the separated form is 29% wrong.

**Current, since 2026-07-31**: the CBCS branch of `approximate_step_size` returns

```python
kinetic_energy = (0.7687 * Z ** (7 / 3) - 0.5 * Z**2 + 0.2699 * Z ** (5 / 3)).sum()
nuclei = (Z ** (7 / 3)).sum() ** 2 / (Z ** (14 / 3)).sum()
return np.sqrt(3) * erfinv(1 / 2) * (1 + 0.045 / nuclei) / np.sqrt(kinetic_energy)
```

0.88% rms over seventeen systems, spread 0.982–1.018 (worst He −1.84%, Kr +1.65%). Only the last
constant is fitted — see §3.5 for the sum rule, §3.6 for the correction, §3.7 for the ions. Two
earlier versions: `1.096·(1+0.030/nuclei)·N^(−1/2)·z_eff^(−2/3)` (1.10% on the fifteen neutrals but
**7.9% once ions are included**), and before that `1.118·N^(−1/2)·(ΣZ²/ΣZ)^(−2/3)` at 2.26%.
Before either, `xdata = linspace(0, 2, 11)` in units of `1/N` put the 50% point anywhere from 0.61
(Kr) to 1.86 (B2H6) — for B2H6 all 11 probes sat above 47% acceptance, so `ts0` was badly
conditioned, and benzene (predicted `a = 2.26`) would have fallen outside the scan entirely.
Now the 50% point lands at 1.00 ± 4% for all eight systems and the scan spans acceptance
1.0 → 0.20. This changes only the conditioning, not the answer: `curve_fit` solves for the
physical step whatever the units.

`vmc_step_graph` deliberately does **not** follow: its axis stays `step_size × electrons`,
the convention of every file in `examples/time_step` (verified: `EBES/Ne.dat` runs
0.86859 = 0.08686×10 to 17.37, i.e. the range comes from `approximate_step_size` but the axis
does not). Normalizing the graph by the law the graph exists to measure would collapse every
system onto one curve, hide the scaling, and put new files on a different axis from the old
ones. Range from the current best guess, axis from a fixed convention.

The constant is not a rounding of 1: fitting `lnC, α, β` freely gives
`C = 1.163 ± 0.012`, i.e. **14.4σ from 1.000**, on data whose individual `ts₅₀` are known to
0.08–0.23%. Imposing `C = 1` drives α to 0.492 but the residual rms goes from 0.83% to 5.41%.

### Series A, measured 2026-07-30: the separable form is falsified

The α/β split cannot be read off the original eight systems, because six of them are neutral
atoms with `log N = log z_eff` identically: they correlate at −0.789, pin the sum
`α + β = 1.181` superbly, and leave the split to B2H6 and O3 alone. So a **10-electron
isoelectronic series at fixed `N`** was measured to get β on its own — HF, H₂O and NH₃ added to
the existing Ne (`a = ts₅₀·N` on the `step_size × electrons` axis):

| | z_eff | a | x0 | a/x0 | rms |
|---|---|---|---|---|---|
| Ne | 10.00 | 0.76835 ± 0.00151 | 0.75607 | 1.016 | 0.00164 |
| HF | 8.20 | 0.86844 ± 0.00173 | 0.85460 | 1.016 | 0.00177 |
| H₂O | 6.60 | 0.99329 ± 0.00133 | 0.97624 | 1.017 | 0.00126 |
| NH₃ | 5.20 | 1.15072 ± 0.00111 | 1.12422 | 1.024 | 0.00099 |

**Within the series the power law is essentially exact**: `β = 0.6177 ± 0.0030`, residuals
−0.001 / −0.013 / +0.012 / −0.003 %, sequential pairwise slopes 0.617 / 0.619 / 0.617 (no
curvature). Out-of-sample prediction worked twice: the code's round-exponent law hit HF to
−0.22%, and the three-point fit predicted NH₃ at 1.1510 against 1.15072 measured — **−0.02%**.

**But that exponent does not transfer.** A global fit gives `β = 0.6405 ± 0.0013`
(`C = 1.1015, α = 0.5201 ± 0.0008`, 1.66% rms over the fifteen systems now measured) — ~7σ from
the series-A value. Forcing `β = 0.6177` and refitting `C, α` on the
other eight leaves *structured* residuals, not scatter:

He −0.62%, Be −1.10%, N −0.88%, Ne −1.35%, Ar −2.16%, Kr −2.87%, B2H6 +1.26%, O3 +2.88%

The atomic series drifts monotonically with size across 2.3%, and the molecules go the other
way. That trend stands on its own, independent of O3 being the largest outlier. So
`C·N^(−α)·z_eff^(−β)` is **an approximation good to 1–3%, not a law**: separable in a family of
fixed `N`, not across families. Any conclusion that relied on separability is void — in
particular `α = 1.181 − β = 0.563` is not a valid inference.

Where to look next: the shape parameter `a/x0` is *also* not universal, and drifts the same way.
Atoms: He 0.935, Be 0.975, N 0.997, Ne 1.016, Ar 1.036, Kr 1.071. Molecules run higher at
comparable `N` — B2H6 (16e) 1.109 vs Ar (18e) 1.036, O3 (24e) 1.163 vs Kr (36e) 1.071. Inside
series A it is nearly constant, 1.016–1.024 — exactly where `a` also behaves cleanly. So `a` and
`a/x0` look driven by one and the same missing parameter that `z_eff = ΣZ²/ΣZ` fails to capture.

### The missing variable is charge spread

`z_eff = ΣZ²/ΣZ` is one moment of the charge distribution; it cannot tell CH₄ from a fictitious
uniform system of the same mean. The next moment,

```
h = (ΣZ³/ΣZ²) / z_eff        h = 1 for any single-element system
```

is identically 1 for all six atoms and for O₃, and rises with the H content — B2H6 1.306, CH₄
1.375, NH₃ 1.280, H₂O 1.180, HF 1.086, CₙHₙ 1.110. Adding `h^(−γ)` to the fit over all fifteen
systems:

| model | rms | α | β | γ |
|---|---|---|---|---|
| `C·N^(1−α)·z^(−β)` | 1.66% | 0.5201 ± 0.0008 | 0.6405 ± 0.0013 | — |
| `C·N^(1−α)·z^(−β)·h^(−γ)` | **0.45%** | 0.5213 ± 0.0008 | 0.6584 ± 0.0016 | 0.1069 ± 0.0047 |

γ is 23σ from zero and cuts the residual by 3.7×, and both exponents move *toward* the round ½
and ⅔ in the code rather than away. Worst cases left are Kr −1.20% and Ar −0.86%, the two heaviest
atoms, where `h ≡ 1` and so γ cannot be what is missing. The lesson generalizes: each added
moment of `{Z_a}` buys about a factor of 3, which says the true law is a functional of the whole
nuclear charge distribution and no finite truncation of it will be exact.

### 3.5 The law splits into two independent steps

Before any of the detail below: the whole thing is **two steps that share nothing**, and their
errors add in quadrature, which is how we know the split is real and not just a narrative.

| | contains | rms |
|---|---|---|
| **1. sampling theory** — `ts₅₀·√⟨T⟩ = √3·erfinv(½)·(1 + correction)` | 0.826, the non-Gaussian term, `l = 0` only | 0.46% |
| **2. quantum chemistry** — `⟨T⟩` for the system | virial, Thomas–Fermi, β = ⅔, `z_eff`, charge spread `h`, "moments of Z" | 1.47% in `⟨T⟩` → 0.74% in `ts` |
| quadrature `√(0.46² + 0.74²)` | | 0.87% |
| **measured together** | | **0.85%** |

Residual correlation between the two −0.33. The entire wave function reaches step 1 through the
single scalar `⟨T⟩`; step 1 knows no chemistry and step 2 knows no Monte Carlo.

Two consequences worth stating plainly:

- **Everything about β, `z_eff` and `h` is step 2** — i.e. quantum chemistry, measured through
  acceptance curves, when `⟨T⟩` for all these systems sits in the ORCA output. That was the wrong
  instrument, and the "missing variable" it kept implying was an artefact of using it.
- **`α = ½` is not a sampling result.** At fixed composition `⟨T⟩ ∝ N` by extensivity, so
  `ts ∝ N^(−1/2)` follows from step 2. RGG reaches the same exponent by the same mechanism, so
  there is no conflict, but "series B confirmed RGG" was overstated: it confirmed that `⟨T⟩` is
  extensive. The sum rule is the stronger statement — it fixes the constant too, not just the
  exponent.

### 3.5.1 Step 1 in detail, and where 0.826 comes from

Displace all `N` electrons at once, `δr_{iα} ~ U[−ts, ts]` independently, so
`E[δr_{iα}δr_{jβ}] = (ts²/3)δ_{ij}δ_{αβ}`. Expand `X ≡ ln(Ψ′²/Ψ²)` to second order at **fixed** `R`:

```
X = 2·δR·F + δR·H·δR + O(δR³),      F = ∇lnΨ,  H = ∇∇lnΨ
```

There are two local kinetic energies, equal in mean and **not** in variance:

```
T_L(R) = −½·∇²Ψ/Ψ                       laplacian form; what wfn.kinetic_energy returns
T_D(R) = ½·Σᵢ|∇ᵢlnΨ|² = ½·F·F           drift form; the sum rule needs THIS one
```

`⟨T_L⟩ = ⟨T_D⟩ = ⟨T⟩` by integration by parts. Averaging over `δR` at fixed `R`:

```
Var(X|R) = 4·(ts²/3)·F·F = (8/3)·ts²·T_D(R)
E[X|R]   = (ts²/3)·Tr H = −(2ts²/3)·[T_L(R) + T_D(R)]
```

Averaging those over `R` gives `σ² = (8/3)ts²⟨T⟩` and mean `= −σ²/2` — the Metropolis constraint
`E[e^X] = 1` falls out by itself, which is the check that the expansion is right.

If `T_D` were constant, `X` is one Gaussian, detailed balance gives `A = 2Φ(−σ/2)`, and `A = ½`:

```
σ = 2Φ⁻¹(¾)   →   ts₅₀·√⟨T⟩ = Φ⁻¹(¾)·√(3/2) = √3·erfinv(½) = 0.826078
```

(`Φ⁻¹(p) = √2·erfinv(2p−1)`, and the `√2` cancels against `√(3/2)`.) **This is a pure number with
no physics in it** — three conventions are baked in, and each moves it:

| | `ts·√⟨T⟩` |
|---|---|
| target 50%, uniform cube (ours) | **0.8261** |
| target 30% | 1.2694 |
| target 23.4% (RGG optimum) | 1.4576 |
| Gaussian proposal of sd `ts` | 0.4769 |
| uniform *ball* of radius `ts` | 1.0665 |

General target `A`: `√3·erfinv(1−A)`. And asking why the prefactor of the *final* law is not 1 is
malformed — `ts` is in bohr, so it is dimensional.

### 3.5.2 Step 2: β = ⅔ is Thomas–Fermi

There is no separate "law in Z". The sum rule fixes everything at once:

```
ts₅₀ = 0.826 / √⟨T⟩            (Gaussian limit)
⟨T⟩  = |E|                     (virial)
```

so any Z dependence must come through `⟨T⟩` and nothing else. Thomas–Fermi gives
`T_TF[ρ] = C_F∫ρ^(5/3)`, hence `Z^(7/3)` per atom, and `⟨T⟩` is additive over nuclei because it is
collected in the core region where bonding barely reaches. Measured on the fifteen systems:

```
⟨T⟩ = 0.5358 · Σ_a Z_a^(2.380 ± 0.004)      rms 1.0%      (TF: 7/3 = 2.333)
```

The 2% excess is the Scott and Weizsäcker corrections — TF is itself asymptotic in Z. Substituting,
with `N = ΣZ` for a neutral system:

```
a = ts₅₀·N = K·√N·(ΣZ^(7/3)/ΣZ)^(−1/2) = K·√N·z_TF^(−2/3),   z_TF ≡ (ΣZ^(7/3)/ΣZ)^(3/4)
```

`z_TF = Z` identically for a single element, so it changes nothing for atoms; it differs only for
mixed systems (CH₄ 4.277 vs 4.000, B2H6 3.698 vs 3.500, CₙHₙ 5.406 vs 5.286). **β = ⅔ is now
derived: half of the TF exponent 4/3.** Refitting β freely with the two candidate moments:

| moment | β free | γ on `h` |
|---|---|---|
| `z_eff = ΣZ²/ΣZ` | 0.6405 ± 0.0013 (**20σ from ⅔**) | +0.1069 ± 0.0047 (23σ) |
| `z_TF` | **0.6623 ± 0.0014 (3σ from ⅔)** | −0.0308 ± 0.0045 (−7σ) |

So the charge spread `h` was **a patch over the wrong moment**: it collapses and changes sign once
`ΣZ²` is replaced by `ΣZ^(7/3)`. The earlier reading — "each extra moment of {Z_a} buys a factor of
3, so the law is a functional of the whole charge distribution" — was wrong in its premise; there
was one right moment, not a series. What survives of it is narrower and true: `⟨T⟩ = C_F∫ρ^(5/3)`
is a *nonlinear functional* of the density, not a moment of it, so the residual 1% is structural.

With both exponents pinned at their derived values, only the prefactor and the non-Gaussian
correction are free:

```
a = 1.0959·√N·(1 + 0.0295/nuclei)·z_TF^(−2/3)        rms 1.10%, max 2.93% (Kr)
```

**Where the prefactor comes from.** It is not free either — `C = 0.826/√c` where `⟨T⟩ = c·ΣZ^(7/3)`:

| `c` | source | `C` |
|---|---|---|
| 0.7687 | TF textbook, `E_TF = −0.7687·Z^(7/3)` | 0.9421 |
| 0.5866 | fit with the exponent forced to 7/3 | 1.0785 |
| 0.5358 | fit with the exponent free (2.380) | 1.1284 |
| | **fitting `a` directly** | **1.0959 ± 0.0090** |

So the fitted 1.096 sits between the two TF-calibrated values, i.e. it is predicted to ~2–3% and no
better — the textbook constant alone gives 19% rms, because `E_TF` overshoots badly at these Z
(Kr: 3290 vs 2752 Ha, 20% high). Do not present 1.096 as derived.

**How well the two constants are known.** The formal `curve_fit` errors (±0.0012 and ±0.0015) are
worthless: χ² = 121 on 13 dof, i.e. the model misses by 1.2% where the data are good to 0.15%, so
the fit is dominated by model error, not sampling error. Honest numbers:

| | C | k |
|---|---|---|
| formal | 1.0920 ± 0.0012 | 0.0400 ± 0.0015 |
| inflated by √(χ²/dof) = 3.1 | ± 0.0036 | ± 0.0047 |
| **equal weights** (model error dominates, so `σ ∝ a`) | **1.0959 ± 0.0090 (0.8%)** | **0.0295 ± 0.0102 (35%)** |
| leave-one-out sd | ± 0.0036 | ± 0.0080 |
| bootstrap over systems, 68% CI | [1.0929, 1.0996] | [0.0214, 0.0370] |

`corr(C, k) = −0.93`, so they are not independently determined. Note the **weighting choice moves
`k` by 26%, more than any of the quoted errors** — statistical weights over-weight CH₄/NH₃/B2H6
purely because their curves were sampled harder, which is irrelevant when the residual is
systematic. Equal weighting is the defensible choice and also fits better (1.10% vs 1.20% rms), so
the code carries 1.096 and 0.030.

`k` survives at about 3σ (bootstrap `P(k ≤ 0) = 0.2%`) but nothing more: the correction is
1.0%±1.0% of the step for a single-nucleus system and 0.5%±0.2% for benzene, while dropping it
entirely costs only 1.44% vs 1.10% rms. **It is at the edge of being worth having**; the sign and
the `1/nuclei` scaling are the results, the magnitude is soft. Kr and He alone move it by +20% and
−12%, because at `nuclei = 1` they are the whole lever arm.

One-parameter forms, for comparison: `a = 1.114·N/√(ΣZ^(7/3))` gives 1.58% with *no* fitted
exponent at all, and even beats `a = 0.849·N/√|E_HF|` (1.89%) — the TF estimate of `⟨T⟩` predicts
the step better than the true HF energy does, because its fitted constant absorbs the non-Gaussian
excess that the exact energy leaves exposed. `N/√(ΣZ²)` gives 11.1%, which is how badly the second
moment fails once it is not propped up by a free exponent.

### 3.6 The non-Gaussian correction: it is Var(T_D), and it is derived

`ts₅₀·√⟨T⟩ = 0.826` assumes `X` is Gaussian. It is not, and the excess is the last fitted constant
— but it is not a constant at all. From §3.5.1, at fixed `R` the variance is `(8/3)ts²·T_D(R)`, and
`T_D` **fluctuates over configuration space**. So `X` is a *scale mixture* of Gaussians:

```
A(ts) = ⟨ 2Φ(−ts·√(2·T_D(R)/3)) ⟩
```

`2Φ(−ts√(2T/3))` is convex in `T`, so by Jensen the spread of `T_D` **raises** acceptance at fixed
mean, and a larger step is needed to come back to 50%. **The excess is positive by construction** —
predicted, not fitted — and it measures +2.9…+6.3%. Expanding to second order:

```
Δts/ts = (1 + s²)/8 · Var(T_D)/⟨T⟩² = 0.18187 · Var(T_D)/⟨T⟩²,   s = Φ⁻¹(¾)
```

Checked against direct numerical solution of `A(ts) = ½` for gamma-distributed `T_D`
(exact/predicted = 1.000, 1.006, 1.020, 1.061 at CV = 0.1, 0.2, 0.4, 0.6 — second order, so it
under-predicts by ~6% at CV = 0.6, ~10% at 0.8).

So **`k = 0.182·CV²(T_D)·nuclei`**, and `k = 0.045` means `sd(T_D)/⟨T⟩ = 0.497` at one nucleus.
Inverting the measured excess per system:

| | excess | nuclei | `sd(T_D)/⟨T⟩` | `CV²·nuclei` |
|---|---|---|---|---|
| Be²⁺ | 2.87% | 1 | 0.397 | 0.158 |
| Li⁺ | 3.79% | 1 | 0.457 | 0.208 |
| He | 5.01% | 1 | 0.525 | 0.275 |
| Ne | 5.14% | 1 | 0.532 | 0.283 |
| Kr | 6.32% | 1 | 0.589 | 0.348 |
| C₂H₂ | 2.80% | 2.06 | 0.392 | 0.317 |
| C₆H₆ | 1.16% | 6.18 | 0.253 | 0.394 |

**Why `1/nuclei`.** `T_D` is dominated by whichever electron is near a nucleus (`|∇lnΨ| ~ Z`
there), so `n` equivalent nuclei give `n` independent contributions and `CV² ∼ 1/n`. Adding
*electrons* to an atom adds outer shells contributing almost nothing — which is exactly what the
atoms show: `N` sweeps 2 → 36 and the excess stays 4.4–6.3%, `excess × N` scattering by 88% against
17% for `excess × nuclei`.

**Why it also falls with Z** (series C, §3.7): at frozen `N = 2` and `nuclei = 1` the excess still
drops 5.01 → 3.79 → 2.87%, because a two-electron ion becomes hydrogenic — `lnΨ → −Zr` makes
`|∇lnΨ| → Z` *exactly*, so `CV(T_D) → 0`. Counting nuclei is therefore a crude surrogate for what
actually matters, the dispersion of `T_D`; hence the 35% uncertainty on `k`.

**Practical.** `wfn.kinetic_energy` (`wfn.py:210`) returns `(T − F@F)/2`, i.e. the *laplacian* form
`T_L`. Its mean is right, its variance is not the one needed. `T_D = F@F/2` is already computed on
that same line and simply not returned.

### Only l = 0 enters, so the law cannot see geometry

The sum rule is a **full trace**:

```
⟨Σᵢ|∇ᵢ lnΨ|²⟩ = Tr ⟨∇_α lnΨ ∇_β lnΨ⟩ = 2⟨T⟩
```

CBCS displaces every electron isotropically, so only the trace of that rank-2 tensor survives — a
scalar, `l = 0`. Every multipole of the density (`l = 1` dipole, `l = 2` shape and anisotropy)
cancels identically. Prediction: the law must be blind to molecular geometry.

Series B *is* that test, and it passes: C₂H₂ (D∞h, linear), C₄H₄ (Td, a 3-D cage) and C₆H₆ (D6h,
planar) are maximally different shapes at identical CH stoichiometry, and one two-constant law fits
all three to 0.15%. Bond lengths, angles and planarity never appear.

This also disposes, from theory rather than data, of the "π electrons / shape anisotropy"
explanation once floated for the C₂H₂ anomaly: it placed an `l = 2` effect where only the trace
remains. O₃, strongly bent, sits in the same band as spherical atoms.

**Where `l > 0` would enter:** an *anisotropic* step — a separate `dtvmc` per axis, or along the
principal axes of `Q_αβ = ⟨Σᵢ∇_α lnΨ ∇_β lnΨ⟩` with `Tr Q = 2⟨T⟩`. Then the traceless part of `Q`,
an `l = 2` quadrupole, is what gets optimized, and the efficiency gain scales with the ratio of
eigenvalues of `Q`: identically nothing for atoms, potentially real for linear molecules, slabs and
surfaces. Neither CASINO nor PyCasino has an anisotropic step. Unlike β this follows from structure
already derived, not from a fit. Second entry point: the non-Gaussian `1/N` term, whose fourth
cumulant involves `⟨|∇lnΨ|⁴⟩`, which is not the trace of any rank-2 tensor — but that is a
percent-level effect decaying as `1/N`.

### 3.7 Series C: ions, and what they broke

Measured 2026-07-31. He / Li⁺ / Be²⁺, all `N = 2` with a single nucleus, so the non-Gaussian term
cancels *exactly* in the ratios and the comparison is clean.

| | Z | a | a/x0 | excess |
|---|---|---|---|---|
| He | 2 | 1.02552 ± 0.00312 | 0.9274 | +5.01% |
| Li⁺ | 3 | 0.63736 ± 0.00339 | 0.9195 | +3.79% |
| Be²⁺ | 4 | 0.46062 ± 0.00391 | 0.9126 | +2.87% |

Ratios to He under four candidate forms:

| form | Li⁺ | Be²⁺ |
|---|---|---|
| `√N·z_TF^(−2/3)` — what was in the code | **−18.6%** | **−28.7%** |
| `N/√(ΣZ^(7/3))`, substitution not made | −0.26% | +0.83% |
| sum rule `N/√\|E_HF\|` | −1.17% | −2.04% |
| hydrogenic `N/Z` | −6.8% | −10.2% |

**The failure is a derivation error, not a surprise.** Going from `N/√(ΣZ^(7/3))` to
`√N·z_TF^(−2/3)` substitutes `N = ΣZ`, i.e. neutrality; for Be²⁺, `N = 2` but `ΣZ = 4`. So the
electron count must simply not appear in the formula, and it no longer does. Over all seventeen
systems the old form is 7.86% rms against 1.10% on the fifteen neutrals alone.

Caveat: the numbers above use Hartree–Fock-limit energies for the ions (−7.236415, −13.611299);
the cc-pVQZ `orca.out` files were not yet in the tree, so the sum-rule row may shift slightly.

### Series B settles α: it is exactly ½, and the excess was a finite-size term

Any CₙHₙ has *identically* `z_eff = 222/42 = 74/14 = 5.286` and `h = 1.110`, so the whole series
varies `N` alone. Measured 2026-07-30, all with `cusp_correction : F` (see
`cusp-shift-not-implemented`), benzene at 10⁵ moves per point and the other two at 10⁶:

| | N | a | σ(a) | a/x0 | E_HF (Ha) |
|---|---|---|---|---|---|
| C₂H₂ | 14 | 1.35606 | 0.15% | 1.1221 | −76.85439 |
| C₄H₄ | 28 | 1.89315 | 0.15% | 1.1545 | −153.66234 |
| C₆H₆ | 42 | 2.31002 | 0.17% | 1.1449 | −230.79413 |

(σ(a) inflated by `√(χ²/dof)`; the fit's own error is 3–4× smaller, so the limit is the model, not
the sampling. This is why 10⁵ moves per point suffice — the extra decade buys nothing.)

A pure power law `a ∝ N^(1−α)` gives **α = 0.5157 ± 0.0025**, but the pairwise exponents drift
monotonically — C₂H₂–C₄H₄ 0.5186, C₂H₂–C₆H₆ 0.5151, C₄H₄–C₆H₆ 0.5092 — heading for ½. Fitting
instead

```
a = C·√N·(1 + c/N)      C = 0.3533,  c = 0.359
```

gives χ² = 0.06 against 1.66 for the pure power law (1 dof each). So **α = ½ exactly**, as RGG
requires, and the apparent excess is a finite-size term. RGG is an `N → ∞` theorem; nothing in it
promised the exponent at N = 14. Two points can never see this — the third was decisive.

**Caution: `1/N` here is not the electron count.** Series B varies `N` and the number of nuclei
*together* (`n = N/7` along CₙHₙ), so it cannot separate them, and reading `c/N` as a
central-limit term in the electron count is wrong — see §3.6, where the whole set shows the
correction scaling with the nuclei, not the electrons. Nothing above depends on which it is:
`α = ½` comes from the sum rule and Thomas–Fermi, not from the correction term.

**The prefactor is not fitted either.** The sum rule `ts₅₀·√⟨T⟩ = 0.826` plus `⟨T⟩ = |E|` and
`|E|/N = 5.4903` Ha, constant to 0.1% across the series, forces

```
a = ts₅₀·N = 0.826·√N / √(|E|/N) = 0.3525·√N
```

against the fitted `C = 0.3533` — **0.23%**. Both the exponent and the prefactor of the CₙHₙ law
are derived, not measured.

And `c` is the *same* physics, not a second fact: `ts₅₀·√|E|` measures 0.8492 / 0.8381 / 0.8356 at
N = 14/28/42, approaching 0.826 from above. The sum rule is exact only for a Gaussian
`ln(Ψ′²/Ψ²)`, and what dies here is that non-Gaussianity — but as one over the *nuclei*, see §3.6.

Still outstanding:

- **N/Z decoupling**, N=2 isoelectronic: He 2/0.5124, Li⁺ 3/0.3943, Be²⁺ 4/0.3275 — the test is
  whether they reproduce the β found at N=10, which after the above is doubtful and therefore
  informative either way. H⁻ 1/0.8018 would extend the lever arm but HF gives hydrogen a negative
  electron affinity (HF limit ≈ −0.4879 Ha against −0.5 for H), so it is marginally bound, needs
  an augmented diffuse basis, and is a bonus point rather than a load-bearing one.
- CH₄ (`z_eff` 4.00) completes series A: series-A fit predicts `a = 1.3532`, the code's round
  exponents 1.4030.

None of this touches the code: 1–3% is irrelevant to placing a scan window, and the round
exponents ½ and ⅔ stay adequate.

No Jastrow or backflow is needed — the curve agrees to <1% across all three wave-function levels
in every one of the eight files. Basis quality is a precondition, not a variable to control for:
the law exists to guess a step for production calculations, which have a converged basis by
assumption, so use whatever the system would normally get. The exception is H⁻, marginally bound
at HF, where an inadequate set collapses the second electron and the resulting density is not the
one being modelled — check `E_HF` there. `vmc_step_graph` exists for exactly this measurement;
see the memory `vmc-step-scaling-tests` for the run plan and its cost.

Genuinely untested domain: **pseudopotentials**. All eight calibration systems are all-electron.
For a pseudoatom `atom_charges` holds the effective charge, which is the natural continuation
(the valence electron sees that charge and the rejecting core is gone), but whether the same
exponents and prefactor carry over has not been checked.

## 4. Which acceptance to target, and why max D

### The theorem behind it

For a product target in `d → ∞` with proposal variance `σ² = ℓ²/d`, the rescaled process
converges weakly to a Langevin diffusion whose speed is `h(ℓ) = ℓ²·a(ℓ)` (Roberts, Gelman
& Gilks, *Weak convergence and optimal scaling of random walk Metropolis algorithms*,
Ann. Appl. Probab. **7**, 110 (1997)). In that limit the correlation time of **every**
observable is inversely proportional to `h`, so maximizing the diffusion constant is
exactly optimal and observable-independent. Maximizing `h` gives `s = ℓ√I ≈ 2.38` and

```
optimal acceptance -> 2Φ(−1.19) = 0.234
```

That is the answer to "why not 50%": 50% is a one-dimensional intuition (the 1D optimum is
≈0.44); in high dimension the acceptance collapses faster than the step grows, so the
optimum sits deep in the low-acceptance tail.

The conditions of the theorem are also the failure modes: factorizable target, a single
length scale, smoothness, and steps small compared to the scale on which the observable
varies. Known counterexamples:

- **Redundant directions.** Lee et al.: rigidly translating all electrons of a HEG gives a
  large `D` and terrible sampling. `D` measures `|ΔR|²` in the full 3N space, but `E_L`
  depends on relative coordinates far more than on the centre of mass.
- **Scale separation.** `D` is dominated by valence electrons while `n_corr` of `E_L` is
  dominated by the slowest mode; the variance of `E_L` comes mostly from near the nuclei.
  This is the likely reason the measured `a_opt` drifts down to 9–17% for large
  heterogeneous systems.
- **EBES.** Their Table I: `E_Dmax/E_opt` = 0.52, 0.40, 0.39, **0.09**, 0.96, 0.66. The
  acceptance in EBES does not collapse as the step grows (a single electron can be thrown
  far and still occasionally land somewhere with density), so `ts²·p` has no useful
  interior maximum and `D`-max runs away to absurd steps. Consistent with our data: the
  2-parameter form above **fails to fit the EBES curves** (`examples/time_step/EBES`,
  degenerate `ts0`, D-maximum outside the sampled range), i.e. the shape is CBCS-specific.
- For **CBCS** it works: `E_Dmax/E_opt` = 0.90, 0.82, 1.00, 1.00, 0.88, 0.79.

So: max D is a proxy, justified by a theorem whose hypotheses CBCS approximately satisfies
and EBES does not.

### Numbers from our own curves

Maximizing `x²·p(x)` on the fitted curve (`x = ts·N`), which is exact for the *shape*:

| | a (50% point) | x at max D | p there | D(50%)/D_max | step ratio |
|---|---|---|---|---|---|
| He | 1.025 | 2.336 | 0.173 | 0.558 | 2.28 |
| N | 0.822 | 1.782 | 0.182 | 0.584 | 2.17 |
| Kr | 0.605 | 1.244 | 0.192 | 0.614 | 2.06 |
| B2H6 | 1.858 | 3.724 | 0.198 | 0.629 | 2.00 |
| O3 | 1.356 | 2.626 | 0.205 | 0.649 | 1.94 |

17–20% across He→Kr and both molecules, because the optimum depends only on the shape and
the shape is universal in `u` with `a/ts0 ≈ 1`. The three independent estimates —
RGG 0.234, this curve 0.18, Lee et al. measured 0.27–0.32 for small atoms — all say the
same thing: well below 50%. Lee et al. `E_50%/E_opt` for CBCS: 0.87, 0.82, 0.70, 0.59,
0.54, 0.25 (degrading with system size).

**Caveat, important before acting on this.** `ts²·p` is a proxy for `D`, not `D`. With a
uniform proposal, larger `|Δ|` is preferentially rejected, so the true `D` at large step is
lower and the true optimum lies **above** 18% — which is exactly why Lee et al. measure
27–32% on small atoms. The honest route is to measure the mean square displacement
directly. That is nearly free here: `random_walk` already returns `position`, and the
squared displacement between consecutive rows (zero when the row is NaN, i.e. rejected) is
available in the same 10 probe runs `optimize_vmc_step` already performs. Then fit/solve as
now — this is `OPT_DTVMC : 2`, currently `raise NotImplementedError` in
`vmc_energy_accumulation`. Retargeting also requires widening `xdata` past 2, since the
D maximum lies at 1.24–3.72.

### Series E, measured 2026-08-19: for EBES the optimum is 0.7, and neither 0.5 nor 0.234

Everything above argues about `D`, which is a proxy. This series measures the thing itself.
`examples/step_profile/acceptance.py OUTDIR STEPS [SYSTEM ...]` sweeps the nineteen targets
0.05…0.95 of `Casino.vmc_corr_graph` over eight systems (He, Be, N, Ne, CH₄, Ar, C₂H₂, O₃; gto,
Slater, no Jastrow), for `vmc_method : 1` and for the position-dependent step profile
`vmc_method : 4`, one file per system per method; `acceptance_report.py` reads it. 10⁶ steps a
point, in `examples/step_profile/acceptance_1e6`.

The figure of merit is `ms_indep`: the wall time of one independent sample with the decorrelation
period already minimized out of it (§2), so at fixed wall time the error bar is
`√(var·ms_indep/T)` and `var` belongs to the wave function rather than to the step. This is why
neither `corr_E` nor `D` is the answer on its own — a longer step buys correlation time back at
the price of variance, and `D` counts motion `E_L` does not feel.

| | He | Be | N | Ne | CH₄ | Ar | C₂H₂ | O₃ |
|---|---|---|---|---|---|---|---|---|
| m1, best target | 0.50 | 0.75 | 0.80 | 0.80 | 0.85 | 0.85 | 0.80 | 0.80 |
| m1, acceptance there | 0.497 | 0.692 | 0.742 | 0.748 | 0.799 | 0.790 | 0.741 | 0.744 |
| m1, cost of the 50% target | 1.00 | 2.94 | 2.38 | 2.07 | 2.37 | 2.85 | 1.92 | 2.23 |
| m4, best target | 0.50 | 0.65 | 0.60 | 0.65 | 0.65 | 0.70 | 0.65 | 0.60 |
| m1/m4 `ms_indep`, own optima | 1.03 | 1.00 | 1.25 | 1.15 | 1.45 | 1.35 | 1.23 | 1.24 |

Three readings, in order of how firm they are.

1. **The EBES optimum is 0.69–0.80 and drifts up with the depth of the core.** He, which has no
   core, is the one system whose optimum really is a half — i.e. the 50% rule is the special case
   of a single occupied shell, exactly the regime where §5's `correction` column is 1.06. The
   heavier the core, the further the optimum moves up, because at a step set by the mean electron
   the core is rejected regardless and the walk pays for it in `T_move` (§5's mixture mechanism,
   here in its efficiency form). The 50% rule costs a factor 1.9–2.9 on everything but He.
2. **RGG's 0.234 is not merely unreachable in EBES, it is the wrong direction.** The measured
   optimum sits *above* 50%, where the theorem asks for far below it. No contradiction: the RGG
   limit prices a move at a constant, while `ms_indep` prices it at `p·T_move + T_energy` with `p`
   re-minimized at every point, and the target it optimizes is a mixture over `|∇ᵢlnΨ|²`, not a
   product measure. Both hypotheses fail in the same direction.
3. **The profile (m4) moves the optimum back down, to 0.58–0.68**, which is what it should do if
   it works: equalizing the per-electron acceptance removes the mixture, and the optimum walks
   back toward the one-shell value. He, where the profile is inactive, does not move.

### CASINO confirms it, in its own units and with its own optimizer

`examples/step_profile/casino_scan.sh SYSTEM` runs one CASINO VMC per time step, `vmc_method 1`,
`opt_dtvmc 0`, on the grid of the m1 sweep above, each row paired with the decorrelation period
that minimized `ms_indep` there; 10⁶ steps, `nproc=4`. The figure of merit is the `Efficiency
(au^-2 s^-1)` CASINO prints itself, `1/(err²·CPU)` — nothing of ours enters it. Results in
`examples/step_profile/casino_{ne,ch4,o3}`:

| | Ne | CH₄ | O₃ |
|---|---|---|---|
| best efficiency, at acceptance | 776 @ 0.699 | 1060 @ 0.799 | 94.7 @ 0.745 |
| the 50% row | 328 @ 0.506 | 254 @ 0.506 | 43.8 @ 0.507 |
| what `opt_dtvmc : 1` actually gives | 291 @ 0.506 | 278 @ 0.495 | 32.9 @ 0.497 |
| cost of the default | 2.67× | 3.82× | 2.88× |

The `auto` row doubles as a cross-check that the two codes mean the same thing by DTVMC: CASINO's
own optimizer lands at acceptance 0.495–0.506, i.e. on the 50% row of *our* grid, so the variance
per component agrees even though the proposals differ (Gaussian vs rectangular).

### The consequence in the code: `ACCEPTANCE_TARGET = 0.70`

`casino/pycasino.py` now carries a module constant, and `optimize_vmc_step` maps to it
(`step *= (erfinv(1−ACCEPTANCE_TARGET)/erfinv(1−acceptance))²`, which is the old line when the
constant is ½, so the fixed-point scheme is untouched). `vmc_step_graph` and `vmc_corr_graph`
re-anchor their nineteen-point grids through `erfinv(1 − ACCEPTANCE_TARGET)`, so every file
measured before the change is still read the same way. `approximate_step_size` is deliberately
left at its 50% anchor: it is only the starting guess, the map corrects it in one iteration, and
its `1 + 0.080/nuclei^0.82` patch was fitted at 50%.

0.70 is not any system's optimum — it is the minimax over the sixteen sweeps, chosen because the
basin is flat and a single constant has to serve both methods:

| target | 0.50 | 0.55 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.85 |
|---|---|---|---|---|---|---|---|---|
| worst of the 16 | 2.94 | 2.55 | 2.09 | 1.52 | **1.38** | 1.73 | 1.81 | 3.29 |
| mean of the 16 | 1.70 | 1.60 | 1.43 | 1.20 | **1.17** | 1.16 | 1.26 | 1.73 |

0.75 ties on the mean and loses on the worst case (Be under the profile); below 0.65 and above
0.80 both halves degrade together. **A user's run therefore no longer samples at the step CASINO
would choose for the same input** — the `opt_dtvmc : 1` branch says so in a comment, and any
DTVMC-level comparison with CASINO reference output must account for it.

Caveats worth carrying:

- **CBCS is untested.** Everything here is EBES. For CBCS §4's argument (and Lee et al.'s
  measurements) still says the optimum is *below* 50%, so the same constant is very likely wrong
  there in the opposite direction. The constant is applied to both.
- **Timing comparability is the weak point of the m1/m4 column.** `us_energy` is
  method-independent by construction, and it differs between the two files by 21% on Ne and 12%
  on CH₄ — those runs were taken under different machine load. So Ne's 1.15 is a floor and CH₄'s
  1.45 a ceiling; the trustworthy pairs are Ar, C₂H₂ and O₃, where the column matches to 2%:
  1.35, 1.23, 1.24. Anything comparing methods must run them back to back.
- Grid ratios carry a few percent of noise at 10⁶ steps (visible as non-monotonic wiggles on the
  steep low-acceptance side). A parabola through `log ms_indep` over 0.4–0.95 puts the m1 optima
  at 0.69–0.79 and the m4 optima at 0.57–0.61, the fit being biased low by that steep flank.
- `vmc_method : 4` is PyCasino-only and documented nowhere: not in `docs/source/tutorial/vmc.rst`,
  not in the CHANGELOG, and with no test. Under it, backflow and geminals still take the slow
  `wfn.log_value` branch.

### The ceiling of the profile, and the pending measurement

The profile is a formula, `g(r) = 1/max(1, max_a min(Z_a², Z_a/r_a))`, i.e. a guess at
`⟨|∇ᵢlnΨ|²⟩` from the nuclear positions alone. Measured against the truth on Ne (binned by
distance to the nearest nucleus) it overshoots by 2–3× in the valence, where ~70% of the sample
lives, and is accurate to 0.96–0.99 at the cusp; the real profile is two plateaus (≈95 inside
0.25 bohr, ≈4 beyond 0.5), not a `Z/r` ramp. What no position-only profile can fix is the spread
at fixed `r` — up to 60× between a 1s and a 2p electron at the same distance.

How much a perfect profile could still buy is bounded by `corr_E`: under m4 it is 3.0–3.9 on all
eight systems and 3.41 on He, which has no inhomogeneity left to remove, so 3.41 is the
single-electron Metropolis floor. For Ne that bounds the remaining gain at 3.41/3.92 = 0.87 in
`corr_E`, ≤13% in `ms_indep` and ≤7% in the error bar, against ~15% per move for the `np.interp`
a tabulated profile costs. `examples/step_profile/tabulated.py OUTDIR STEPS [SYSTEM ...]` closes
it: it measures `⟨|∇lnΨ|²|r⟩` on a 33-point log grid, patches the table into a private copy of
`casino/` as a literal (so numba's source hash stays a function of the numbers), and re-runs the
sweep with both variants back to back.

### Series F, measured 2026-08-21: the exact profile buys 1.8%, and it is not the shape that costs

Ne, `examples/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater` under m4, 10⁶ steps a point, in
`examples/step_profile/tabulated/`:

| | best `ms_indep` | at acc |
|---|---|---|
| formula `max(1, min(Z², Z/r))` | **0.1650** | 0.565 |
| measured table `⟨\|∇lnΨ\|²\|r⟩` | 0.1760 | 0.487 |
| the same table with `np.interp` made free | 0.1621 | 0.487 |

As it stands the table **loses** 6.7%: `us_move` is 12.6% higher in every row, and that is the
whole difference. Rebuilding `cost` (pycasino.py, the `(1 + ρᵖ)/(1 − ρᵖ)` block) from the table's
`corr_E` and `move_frac` but the formula's timings isolates the shape at **1.8%**. The `corr_E`
minima are identical to three figures — 3.66 both — so `3.66` for Ne (and `3.41` for He) is a
floor neither profile controls: it is the part of the gradient that is not a function of one
electron's position at all.

Where they differ is off-optimum. At matched acceptance the formula degrades and the table does
not: `corr_E` 8.59 vs 4.78 at acc 0.33, 30.39 vs 8.08 at acc 0.21. Over acc 0.37–0.61 the table's
`ms_indep` varies by 8.5%, the formula's by 42%. **The smoking gun is the `diffusion` column**: at
acc ≈ 0.21 the formula has the *larger* mean displacement (0.0742 vs 0.0701) and 3.8× the
correlation time. It moves electrons, just not the ones E_L lives on — which is also the cleanest
statement of why CASINO's `opt_dtvmc : 1` misses by 2.7–3.8× (§4).

### Why the basin widens: a profile equalises ℓ, not the step

For a one-electron Gaussian proposal of width `√s`, the local acceptance depends only on
`ℓ(r) = s(r)·I(r)` with `I = ⟨|∇ln|Ψ|²|²⟩` (RGG). A table built as `s ∝ 1/I` makes **ℓ constant
in space**; the formula does not. Normalised to a density-weighted mean of 1 on the Ne table:

| r | share of density | ℓ_formula/ℓ_ideal | step too wide by |
|---|---|---|---|
| < 0.12 | 7% | 1.4 | 1.19 |
| 0.15–0.35 | 16% | 1.6–2.6 | 1.28–1.61 |
| 0.43–0.65 | 22% | 0.97–0.53 | 0.99–0.73 |
| 0.80–1.5 | 42% | 0.51–0.71 | 0.71–0.84 |

p5..p95: ℓ spans **5.0×**. Two consequences, and they are the same fact twice:

- **Acceptance stops being a sufficient statistic.** With ℓ constant every electron sits at the
  same point of one curve and the global acceptance fixes it. With ℓ spread, the acceptance is a
  density-weighted average over a population scattered along that curve, and two runs at equal
  acceptance are not equal runs. The apparent acceptance-dependence of the formula's efficiency is
  heterogeneity in disguise.
- **The optimum moves up in acceptance**, because it has to be held high enough that the
  over-stepped shell (r ≈ 0.15–0.35, exactly where E_L fluctuates) does not freeze while the mean
  still looks healthy. Monotone in how heterogeneous ℓ is: m1 (one global step) 0.69–0.80, m4
  formula 0.58–0.68 (Ne 0.565), m4 ideal table 0.487. It stops well above RGG's 0.234 because
  `ms_indep` pays `us_energy` per accepted move and counts E_L decorrelation, not displacement.

### A two-parameter formula for the table, to apply after testing

The measured `κ = √I` says the middle branch is wrong: `κ·r` is 1.6–2.3 across the whole
inter-shell fall on Ne (2.09, 2.25, 2.05, 1.75, 1.60, 1.60, 1.73 at r = 0.23…0.80), i.e.
**κ ∝ 1/r**, while the code assumes `κ² ∝ Z/r`, which falls half as fast. Fitting

```
I(r) = min(Z², (a·Z^⅓ / r)²) + κ_v²        a = 0.76,  κ_v = 1.18
```

jointly to Ne (weighted by the measured density) and Ar (uniform in log r — its counts were
deleted before they were used):

| | rms log(I_model/I_meas), Ne | Ar | ℓ spread on Ne |
|---|---|---|---|
| `max(1, min(Z², Z/r))` | 0.518 | 0.771 | 4.97× |
| `min(Z², (0.76·Z^⅓/r)²) + 1.18²` | 0.241 | 0.351 | **2.03×** |

Checked and rejected: the Z exponent barely matters (`a = 1.86` with no Z gives 0.261/0.362);
a cubic middle branch fits Ne to 0.128 but Ar to 0.455, since one steep branch cannot span
K→L→M; a Slater-exponent shell model, `ρ = Σ N_n ζ_n³ e^{−2ζ_n r}` with `κ = |∇lnρ|/2`, gives
0.754/0.743 — no better than the current formula, because Slater's valence exponents are off by
2–3×. What the fit still misses is the L-shell shoulder (Ne r = 0.19–0.35, 16% of the density,
under by 1.5×) and the Gaussian tail beyond r = 3.5, where the true curve turns back up
(2.34 → 4.03 → 6.02) and 1% of the density lives.

The patch, **not applied** — it invalidates the numba cache for `vmc.py`, breaks
`tabulated.py`'s `assert FORMULA in source` (its constant has to move in lockstep) and makes the
whole m4 series incomparable, so it waits on a sweep of its own:

```python
    def impl(self, r_e, e):
        gradient = 0.0
        for atom in range(self.wfn.atom_positions.shape[0]):
            charge = self.wfn.atom_charges[atom]
            r = np.sqrt(((r_e[e] - self.wfn.atom_positions[atom]) ** 2).sum())
            screened = 0.76 * charge ** (1 / 3) / r
            gradient = max(gradient, min(charge * charge, screened * screened))
        return 1 / (gradient + 1.39)
```

The floor is outside the loop so a molecule adds it once rather than per nucleus. What the sweep
has to show is not a better minimum — the exact table itself only bought 1.8% — but the wide
basin: the optimum should fall from 0.565 towards ≈0.49 and the spread over acc 0.37–0.61 from
42% towards the table's 8.5%. If it does, `ACCEPTANCE_TARGET` needs a separate value for m4.

### Does the Jastrow move the table?

It cannot move the part that matters most, and where it can, the fix is free.

- **The cusp anchor is Jastrow-proof.** At r → 0 the electron-nucleus cusp pins `|∇lnΨ| = Z`
  whatever else is in the wave function: the determinant carries the cusp and CASINO's χ term is
  constrained to `χ'(0) = 0` precisely so as not to spoil it. The measured plateaus confirm it —
  103 against Z² = 100 for Ne, 320 against 324 for Ar, on runs with no Jastrow at all. The `Z²`
  branch needs no re-measurement, ever.
- **What is exposed is `κ_v`**, the valence floor, which is the one region where a Jastrow
  gradient is comparable to the orbital one (O(1) against κ_v = 1.18). The u term also
  contributes a gradient that is not a function of `r_i` at all, so it partly lands in the
  irreducible `corr_E` floor of 3.66 rather than in any profile.
- **Tabulating per run is the wrong response** even so. The exact table beats the fitted formula
  by at most the 2.03× → 1.0× of ℓ spread, and 5.0× → 1.0× was worth 1.8%; against that,
  `np.interp` on every move costs 12.6% of `us_move`, i.e. 6.7% of `ms_indep`. The interpolation
  eats the answer several times over — which is the whole point of Series F.
- **The right response is to re-fit the two constants at run start**, not to carry a table: keep
  the closed form, spend one short walk (`TABLE_STEPS = 20000` is ~40 s and already coded in
  `tabulated.measure`) to measure `I(r)`, fit `a` and `κ_v`, and pay nothing per move. That
  absorbs a Jastrow, a backflow, a pseudopotential or a molecule in two numbers.
- **The cheap measurement that decides whether even that is needed** is one table on a
  Jastrow-optimised run against the bare-determinant one — add
  `examples/gwfn/Ne/HF/cc-pVQZ/CBCS/Slater_Jastrow` to `SYSTEMS` and compare the two `.table`
  files. If `κ_v` moves by less than the ~20% that the fit residual already absorbs, the fixed
  constants stand and nothing has to be measured per run.

## 5. CBCS vs EBES

### What the sum rule says about the EBES step

Same derivation as §3.5.1, one line changes: EBES moves **one** electron, so at fixed `R` and
chosen electron `i`, `Var(X|R,i) = 4(ts²/3)·|∇ᵢlnΨ|²`. Averaging over `i` as well as `R`,
`⟨|∇ᵢlnΨ|²⟩ = (1/N)Σᵢ⟨|∇ᵢlnΨ|²⟩ = 2⟨T⟩/N`, hence

```
Var(X) = (8/3)·ts²·⟨T⟩/N          — N times smaller than CBCS at the same ts
ts₅₀(EBES)·√(⟨T⟩/N) = 0.826·(1 + correction)
ts₅₀(EBES) = √N · ts₅₀(CBCS)
```

So the EBES step is set by the kinetic energy **per electron**, `t̄ = ⟨T⟩/N`, and three sharp
predictions follow:

1. **It is size-independent at fixed composition.** `t̄` is intensive, so C₂H₂, C₄H₄ and C₆H₆ should
   all take the *same* EBES step. Nothing like this holds for CBCS.
2. For neutral atoms `t̄ = |E|/Z ≈ 0.7687·Z^(4/3)`, so `ts₅₀ ∝ Z^(−2/3)` — **not** the
   `1/log(max Z)` the code currently returns (`approximate_step_size`, `vmc_method == 1`), which
   also divides by zero for a hydrogen-only system.
3. **The non-Gaussian correction should be far larger than for CBCS**, and this is the part that
   will not be simple. `X` is now a sum of 3 terms (one electron × 3 axes) rather than 3N, and the
   single-electron `|∇ᵢlnΨ|²` fluctuates without the averaging over N electrons that tames the
   CBCS one. If electrons contributed independently, `CV_EBES ≈ √N·CV_CBCS`, so with `CV_CBCS ≈ 0.5`
   even Ne reaches `CV ≈ 1.6`, where the second-order expansion of §3.6 is useless and the exact
   `A(ts) = ⟨2Φ(−ts√(2T_D/3))⟩` must be solved numerically. Expect a large, system-dependent excess.

### Series D, measured 2026-08-02: prediction 1 holds exactly, predictions 2–3 do not

All 31 calibration systems, 10⁵ steps per point, `examples/time_step/EBES`, on the same grid and
the same conventions as the CBCS set, so the two are directly comparable file by file. Produced by
`examples/time_step/time_step.py examples/time_step/EBES 100000`, which now reads the mode off the
name of the output directory.

Two checks that the thing measured is the thing intended. **H-pp has one electron**, for which EBES
is identically CBCS, and every column agrees: correction 1.041 vs 1.046, sum rule 0.773 vs 0.771,
kurtosis −0.34 vs −0.32, step ratio 1.00. **The gto/sto twins agree** to 0.1–1% throughout (Ar
2.358/2.350, Kr 2.611/2.610, Ne 1.860/1.855, O₃ 1.892/1.908), so what is measured is not an
artefact of how the orbitals are expanded.

**Prediction 1 is confirmed, and sharply.** At fixed composition the EBES step does not move with
the size of the system:

```
C₂H₂ (14 e⁻)  ts₅₀ = 0.6644
C₄H₄ (28 e⁻)  ts₅₀ = 0.6667      spread 0.6% over a threefold change in N
C₆H₆ (42 e⁻)  ts₅₀ = 0.6629
```

**The sum rule itself is fine; the whole defect is the non-Gaussian part.** Scaling every step by
the kinetic energy per electron, `ts₅₀·√(⟨T⟩/N)/0.826`, reproduces the measured `correction` column
to 1–3% on all 31 systems — i.e. `Var(X) = (8/3)ts²⟨T⟩/N` is right and `A = 2Φ(−σ/2)` is what fails.

**Prediction 2 fails, and prediction 3 is the reason.** `ts₅₀(EBES)/ts₅₀(CBCS)` against the `√N` the
sum rule predicts:

| | H-pp | He | Be²⁺ | Be | N | Ne | CH₄ | Ar | O₃ | C₄H₄ | Kr | C₆H₆ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 1 | 2 | 2 | 4 | 7 | 10 | 10 | 18 | 24 | 28 | 36 | 42 |
| measured | 1.00 | 1.42 | 1.42 | 3.23 | 4.74 | 5.56 | 5.83 | 9.17 | 8.94 | 9.87 | 14.35 | 12.07 |
| √N | 1.00 | 1.41 | 1.41 | 2.00 | 2.65 | 3.16 | 3.16 | 4.24 | 4.90 | 5.29 | 6.00 | 6.48 |

`√N` holds only where all electrons are equivalent — one electron, or the two-electron ions. The
excess is the `correction` column, and **it is set by composition, not by N**:

```
one occupied shell (He, Li⁺, Be²⁺)              1.06
pseudoatoms, valence shell only (B→Ne)          1.19–1.31
second period (Be 1.72, N 1.88, Ne 1.86,
              O₃ 1.89, CH₄ 1.96, H₂O 2.00)      ≈1.9
third period (Ar)                               2.36
fourth period (Kr)                              2.61
```

Two controlled comparisons make this airtight. **Be²⁺ vs Be**: same nucleus Z = 4, the only
difference is two valence electrons, and the correction goes 1.057 → 1.718. **CH₄ → C₂H₂ → B₂H₆ →
C₄H₄ → C₆H₆**: 10 to 42 electrons at fixed composition, corrections 1.96, 1.91, 1.83, 1.91, 1.91 —
flat. For CBCS the same column is 1.01–1.06 on every all-electron system.

**Mechanism.** `Var(X|R,i) = 4(ts²/3)|∇ᵢlnΨ|²` depends on *which* electron was picked. The sum rule
averages `|∇ᵢlnΨ|²` over `i` and gets `2⟨T⟩/N`, but the acceptance is a nonlinear functional of the
variance, not a function of its mean: at a step where the average electron sits at 50%, a core
electron is rejected almost always and a valence one accepted almost always, so the 50% is made up
by the valence electrons, whose `|∇lnΨ|` is far below the mean. Hence a larger step than predicted,
a correction that grows with the number of occupied shells rather than with N, and heavy tails — a
mixture of normals with different variances. The kurtosis column confirms it: −0.33 on the
one-shell systems (three uniforms, `n_eff ≈ 3`, exactly the CBCS value) against 4.87 on Ar, 4.63 on
Kr, 7.0 on F-pp, and in every case above the CBCS value for the same system (B₂H₆ 2.94 vs 1.22,
C₆H₆ 2.40 vs 0.99) because there is no averaging over N electrons to tame it.

**Status: not acted on, waiting on Drummond.** The right replacement for `√N` is not a fitted
factor but the solution of `E_i[2Φ(−σᵢ/2)] = ½` over the distribution of `|∇ᵢlnΨ|²` across
electrons, which is measurable from `drift_velocity` per electron with no free parameter. Whether
that is the accepted way to state the EBES step, and whether anyone has done it, is question 6 of
§6. Until there is an answer, `approximate_step_size` keeps `√N` and `optimize_vmc_step` corrects
it in three iterations — which it now can, since the acceptance is counted honestly.

`approximate_step_size`: CBCS the sum rule plus the `1 + 0.045/nuclei` patch, EBES the same times
`√N`. Measured CBCS optima confirm the `1/N` collapse (see §3); the EBES factor is wrong by
1.0–2.6 as above.

Lee et al. Table III, `E_EBES/E_CBCS`, Slater–Jastrow: 1.05 (N pp, 5e⁻), 1.47 (O, 8e⁻),
1.65 (NiO, 16), 1.93 (N₂H₄, 18), 3.11 (HEG, 38), 4.70 (diamond, 64). With backflow it can
**invert** on small systems: 0.90 (N), 0.83 (N₂H₄). Their recommendation is EBES nearly
always, except below ~20 electrons with backflow. Averaging energies over proposed moves
(CBCS2) is worse in every case.

So for He/Be/N the EBES gain is 1.05–1.5, not the 5–10× a naive step-size-ratio argument
suggests: CBCS with `p_opt ≈ 8` recovers most of the gap.

Both obstacles that used to sit here are gone, 2026-08-02:

- `one_electron_step` goes through `SlaterState.ratio_1e`/`accept_1e` in `casino/slater.py`: the
  basis is walked for the moved electron only, the determinant ratio is one dot product per
  determinant against the stored inverse, and an accepted move updates that inverse by the rank-one
  formula of Fahy et al., PRB 42, 3503 (1990) Eq. (26), refreshed in full every `dbar_max_age`
  moves (CASINO's `DBARRC`). Backflow and geminals fall back on the whole configuration — one
  electron moves the quasi-particle coordinates of all its neighbours inside the cutoff, so the
  block update `calc_q_bf`/`update_dbar_bf` would be needed, and it is not written.
- `acceptance_ratio` counts accepted single-electron proposals instead of reading the fraction of
  sweeps in which anything moved. The old reading saturates: at the 50% target Kr changes at least
  one of its 36 electrons with probability `1 − 2⁻³⁶`, so every sample read exactly 1, the
  inversion `1−(1−p)^(1/N)` returned 1, and `optimize_vmc_step` multiplied the step by the 10.76 its
  clipped acceptance asked for. Measured on the current code at a walker already sitting on 50%,
  the old estimator gives 1.000 for Ar, Kr and O₃ (step factor ×10.76), 0.614 for Be and 0.549 for
  Ne — i.e. **every EBES step for anything past neon had been set by a coin toss**. Small systems
  were fine: He 0.510 against a true 0.511.

What is still missing before EBES pays off in wall time: the Jastrow is evaluated in full for every
proposal, `O(N²)`, so the determinant saving is invisible on any system with a Jastrow. The
one-electron Jastrow (CASINO `oneelec_jastrow`, `jas1_diff`) is the next piece.

## 6. Open questions worth putting to Drummond

1. What is `ts0`? Equivalently, which log-ratio distribution gives
   `odds of rejection ∝ e^u − 1`? The exact constraints (`P(−x)=e^xP(x)`, `K(λ)=K(1−λ)`,
   `A = 2P(X>0)`) plus the Bhattacharyya-fidelity route in §3 look like the way in.
2. Is the two-parameter form exact for CBCS, or only excellent? It beats a Gaussian model
   4–8× at equal parameter count over the whole range p ∈ [0.02, 1] on eight systems from
   He to Kr and on two molecules, and `a/ts0 → 1` as the system grows.
3. **The `N` half of the law is closed; the `Z` half is not.** Series B (C₂H₂/C₄H₄/C₆H₆, which
   hold `z_eff` and `h` identically fixed) gives `a = C·√N·(1 + c/N)` with `C = 0.3533` against
   `0.826/√(|E|/N) = 0.3525` predicted by the kinetic-energy sum rule — 0.23%, no free
   parameter. So `α = ½` is exact, RGG is confirmed, and the `1/N` term is the CLT approach of
   `ln(Ψ′²/Ψ²)` to a Gaussian, visible independently in `ts₅₀·√|E|` = 0.8492/0.8381/0.8356 → 0.826.
   β follows from the same sum rule: all Z dependence enters through `⟨T⟩`, Thomas–Fermi makes it
   `Σ_a Z_a^(7/3)` (measured exponent 2.380 ± 0.004), and `β = ⅔` is half of 4/3. With the right
   moment `z_TF = (ΣZ^(7/3)/ΣZ)^(3/4)` the free-fit β lands at 0.6623 ± 0.0014, 3σ from ⅔, and the
   charge-spread patch `h` collapses. **So the whole scaling law is one sum rule plus Thomas–Fermi,
   with two constants left.** Remaining 1%: TF's own Scott/Weizsäcker corrections, largest on He
   (−2.4%) and Kr (−1.5%). Open: does the exact `T_s[ρ] = C_F∫ρ^(5/3) + λ/8∫|∇ρ|²/ρ + …` gradient
   expansion close that 1%, and is anyone interested in an **anisotropic** `dtvmc` optimizing the
   traceless part of `Q_αβ = ⟨Σ∇_α lnΨ ∇_β lnΨ⟩` rather than its trace?
4. CASINO's automatic `VMC_DECORR_PERIOD` is measured on 200–500 moves, where
   `σ_τ ≈ 21` on Be, and falls back to `corper = 1` on a single-precision timer
   difference. Both are fixable by measuring on the production block and inverting the
   AR(1) thinning law — the same Eq. 9 CASINO already has. Would they take it?
5. For CBCS the paper recommends maximizing `D`, but `D` counts motion along directions
   `E_L` is insensitive to (their own HEG rigid-translation example). Is there a better
   cheap proxy — e.g. the mean square change of `E_L` itself per move, which is what
   `n_corr` actually depends on?
6. **The EBES step is not `√N` times the CBCS one, and the excess is not a small correction.**
   Series D (§5, all 31 systems) confirms the sum rule for EBES exactly — scaling by `⟨T⟩/N`
   reproduces the measured step to 1–3% — and confirms the prediction it implies about size
   independence at fixed composition to 0.6% over C₂H₂/C₄H₄/C₆H₆. What it falsifies is the
   Gaussian acceptance law on top of it: the measured `ts₅₀(EBES)/ts₅₀(CBCS)` runs from `√N` on
   one-shell systems to 2.4 times `√N` on Kr, and the factor tracks the number of occupied shells
   rather than N (Be²⁺ 1.06 against Be 1.72 at the same nucleus; 1.9 flat from CH₄ to C₆H₆).
   The reason is that `A` is a nonlinear functional of `Var(X|i) ∝ |∇ᵢlnΨ|²` while the sum rule
   only fixes its mean, so at 50% the step is set by the valence electrons and the core ones are
   dead weight — visible as kurtosis 4.6–7.0 where CBCS has 1–5. The step that follows is the root
   of `E_i[2Φ(−σᵢ/2)] = ½` over the spread of `|∇ᵢlnΨ|²` across electrons, which needs no fitting,
   only the per-electron drift. **Questions: is this the accepted way to state the EBES time step,
   has the shell-structure dependence been reported, and does CASINO's own `OPT_DTVMC` for
   `VMC_METHOD : 1` do anything about it?** Note their DTVMC and ours are not comparable as
   numbers — CASINO proposes from a Gaussian, PyCasino from a rectangular distribution, and only
   the variance per component is common to both — so the comparison to make is acceptance at equal
   variance, not step against step.
   *Half-answered by Series E (§4):* the two codes do agree on DTVMC (CASINO's own optimizer lands
   on the 50% row of our grid), and `OPT_DTVMC : 1` does nothing about the mixture — it targets
   50% and pays 2.7–3.8× in its own printed efficiency against the best step on the same grid,
   which sits at 0.70–0.80. So the sharper question is not the `√N` but the target itself: **is
   there a reason the 50% rule survived in EBES, where the efficiency optimum is measurably above
   it and moves with the shell structure?** PyCasino now aims at 0.70.

## 7. Pointers

Code:
- `casino/pycasino.py`: `ACCEPTANCE_TARGET`, `approximate_step_size`, `vmc_step_graph`,
  `vmc_corr_graph`, `vmc_profile_graph`, `optimize_vmc_step`, `decorr_period`,
  `optimize_decorr_period`, `vmc_energy_accumulation`
- `casino/vmc.py`: `vmc_step_profile` (the `vmc_method : 4` step profile)
- `casino/vmc.py`: `simple_random_step` (CBCS, uniform proposal per Cartesian component,
  all electrons at once), `one_electron_step` (the EBES proposal, also what `log_ratio_walk`
  samples in EBES), `gibbs_random_step` (a sweep of it), `reset` (counters and cached state),
  `acceptance`, `vmc_random_walk`, `observable` (reuses the previous energy for unmoved
  configurations)
- `casino/slater.py`: `SlaterState` (inverse of every slater matrix, log and sign of every
  determinant, age), `Slater.state`, `Slater.orbitals_1e`, `SlaterState.ratio_1e`,
  `SlaterState.accept_1e`, `dbar_max_age`
- `casino/sem.py`: `correlated_sem` (pyblock reblocking; returns `None` if
  `find_optimal_block` fails, which the callers do not guard)

Data: `examples/time_step/{CBCS,EBES,Biased}/*.dat` plus `fit.log`, `time_step.sh`,
`single_atom.dat` (fitted `a`, `x0` vs atomic charge), `forum.txt`.
Efficiency sweeps: `examples/step_profile/` — `acceptance.py` (the campaign),
`acceptance_report.py` (reads it), `casino_scan.sh` (the CASINO control),
`step_profile.py`, `tabulated.py` (the measured-profile experiment, §4 Series F), and the results
in `acceptance_1e6/`, `casino_{ne,ch4,o3}/`, `{CBCS,EBES}/`, `tabulated/`.

CASINO source: `vmc.f90` — `equilibration` (corper calibration block),
`get_optimal_corper`, `eff_estimate`, `DEBUG_OPT_CORPER`; `numerical.f90` —
`correlation_time`, `correlation_time_alt`; `esdf_key.f90` — keyword documentation.

Papers: `pdfs/casino_recomendations.pdf` (Lee et al. 2011, the central reference),
`pdfs/casino_overview.pdf` (Needs et al., JCP 152, 154106 (2020)),
`pdfs/Alternative sampling/trail2010.pdf` and `0909.550{4,5}.pdf` (Trail optimum sampling,
the `VMC_SAMPLING` keyword — not implemented in PyCasino and silently ignored if set).

See also the `qmc` skill for the wave-function side and `profiling` for how to time any of
this.
