---
name: jastrow-forms
description: >
  Use this skill when working on the analytic (non-polynomial) Jastrow forms in PyCasino: the
  exponential correlation hole u = -gamma b exp(-r/b), the bell chi = A w(r/L) and the Gaussian
  chi = A exp(-(r/a)^2), with or without cutoff ("Functional form" 1 and 2 in correlation.data),
  their implementation in casino/jastrow.py and casino/readers/jastrow.py, the f-term forms
  (f "Functional form" 1 = product (r1-L)^C (r2-L)^C g g h, 2 = the product under a Gaussian envelope,
  3 = the CASINO polynomial under a Gaussian envelope exp(-(r1/a)^2-(r2/a)^2), all without cutoff but 1),
  the examples examples/stowfn/*/HF/QZ4P/CBCS/Jastrow_emin_uncut_polynomial, the offline varmin of
  candidate Jastrows on fixed VMC configurations (ranks of the product, density- and 1-RDM-based f), the
  no_dup restart, and the traps met on the way: stale numba cache of wfn.py, a negative hole radius,
  varmin making no step, unreweighted varmin blowing up a polynomial f outside the sampled region.
  Trigger on: functional form, exponential u, bell chi, Gaussian chi, uncut, no cutoff, hole radius,
  u_form, chi_form, f_form, UNCUT_POLYNOMIAL, Gaussian envelope, product f, Jastrow_emin_uncut_polynomial,
  no_dup, prevent duplication, 1-RDM Jastrow, jastrow_form research.
  See also the qmc skill (Jastrow in the wave function) and the numba skill.
---

# Analytic Jastrow forms (u, chi) in PyCasino

Branch `jastrow_functional_form`. The study was in `research/jastrow_form/` (`REPORT.md`, `FORMALISM.md`);
it was deleted on 2026-09-27 and survives only in git history (HEAD before that date).

## 0. Restart 2026-09-27: all earlier results are void

**Why.** Every run behind §1, §3 and §5 had "Prevent duplication of u/chi term" = 0. The polynomial f
then contains cutoff × P(r12) and cutoff × P(r_iI) terms, i.e. a full polynomial u and χ inside its
cutoff. It took over the work of the analytic u and χ, so even terms that contributed nothing gave a
near-polynomial energy, and the CASINO u and χ profiles the forms were fitted to were ambiguous.
- **He uncut with no_dup = 1** (PyCasino): −2.9016 against −2.9037, variance ×3.4. Emin crawled at the
  trust radius (χ amplitude A moved exactly `dp_max` = 0.2 per cycle, A is not log-scaled) after varmin
  threw χ into A = −2.66, a = 6.4 (≈ const + 0.065 r²).
- The 28 standard `Jastrow_emin` inputs (gwfn cc-pVQZ He, Be, N, Ne, Ar, Kr, B2H6, O3, H2O, HF, NH3, CH4;
  gwfn MP2-CASSCF(2.4) Be; stowfn QZ4P He, Be, N, Ne, Ar, Kr, O3; ppotential_HF H, B, C, N, O, F, Ne, B2H6)
  now have both flags = 1 and are rerun in CASINO; all later calculations are to be repeated on them.

**CASINO polynomial with no_dup = 1** (stowfn QZ4P, last emin cycle, against no_dup = 0 at HEAD):

| system | E no_dup 1 | E no_dup 0 | ΔE, mHa | variance 1 / 0 |
|---|---|---|---|---|
| He | −2.90369(5) | −2.90375(4) | +0.06 ± 0.06 | 0.00095 / 0.00054 |
| Be | −14.6480(9) | −14.6508(8) | +2.7 ± 1.2 | 0.081 / 0.044 |
| N | −54.5639(22) | −54.5637(19) | 0 ± 3 | 0.57 / 0.36 |
| Ne | −128.889(5) | −128.908(4) | +19 ± 6, still falling | 2.12 / 1.23 |

The polynomial loses little energy, the variance grows 1.6–1.9×.

**no_dup = 1 does not make u and χ unique.** Fit of the forms (exponential u with free constant, Gaussian χ,
HF-density weights), residual / spread of the profile:

| | channel | no_dup 0 | no_dup 1 |
|---|---|---|---|
| He u | ↑↓ | 0.27 | 0.99 |
| He χ | | 0.04 | 0.53 |
| Be u | ↑↑ / ↑↓ | 0.09 / 0.01 | 1.00 / 0.24 |
| Be χ | | 0.11 | 0.77 |
| N u | ↑↑ / ↑↓ / ↓↓ | 0.05 / 0.04 / 0.97 | 0.08 / 0.10 / 1.00 |
| N χ | ↑ / ↓ | 0.03 / 0.03 | 0.28 / 0.61 |
| Ne u | ↑↑ / ↑↓ | 0.04 / 0.07 | 0.18 / 0.22 |
| Ne χ | | 0.02 | 0.59 |

- With no_dup = 1 u is a hump (He ↑↓: 0.33 at 0, 0.51 at r = 1, then decays; Be ↑↑ likewise), χ almost
  vanishes (Be 6.9 → −0.1…−0.7, Ne χ(0) ≈ 0 with a −1.2 dip at r = 2, N halved).
- no_dup zeroes only γ_{n,0,0} and γ_{0,0,l}. The mixed terms cut(r1)cut(r2) r1ˡ r2ᵐ, summed over the second
  electron, are a one-body function again, and u summed over partners is one too.

**Where the one-body part went** (old vs new Jastrow, rms(new − old)/spread(old), HF-density weights,
independent electrons, spin densities ∝ Nσ):

| system | χ | χ + ⟨f⟩ | χ + ⟨f⟩ + ⟨u⟩ |
|---|---|---|---|
| He | 0.69 | 2.66 | 0.13 |
| Be | 0.91 | 1.06 | 0.83 |
| N ↑ / ↓ | 0.43 / 0.68 | 0.08 / 0.55 | 0.35 / 0.39 |
| Ne | 0.68 | 0.75 | 0.33 |

Pair part u + ⟨f | r12⟩: He ↑↓ 2.63 → 0.46, Be ↑↑ 5.35 → 1.12, Be ↑↓ 0.63 → 0.60, N 0.36/0.38/2.75 →
0.29/0.34/0.66, Ne 0.82/0.81 → 0.61/0.50.
- The one-body content moves between χ, the mean field of u and the mean field of f. He is a clean case
  (0.13): u ↑↓ changed sign and took over what χ carried. Be does not match (different Jastrows,
  ΔE 2.7 mHa, or the HF-density approximation fails for the ↑↑ shell correlation).
- Even the effective profiles agree only to 13–40 % (Be 83 %): Jastrows of nearly equal energy are
  noticeably different functions; the energy surface is flat in parameter space.

**Fit of the full J** (J = Σu + Σχ + Σf of `correlation.out.4`, evaluated by PyCasino on 2·10⁵ VMC
configurations of its own Slater–Jastrow, stowfn QZ4P; models as in old step 1: (U) uncut u + Gaussian χ,
(A) (U) + product f with L_f of CASINO, each with a free constant; rms residual, a.u.):

| system | run | std J | rms (U) | rms (A) | (A)/std | (A): b; χ (A, a) |
|---|---|---|---|---|---|---|
| He | no_dup 1 | 0.084 | 0.029 | 0.010 | 0.12 | b 1.00, 1.58; χ 0.55, 1.61 |
| He | no_dup 0 | 0.084 | 0.029 | 0.010 | 0.12 | b 1.00, 1.59; χ 0.57, 1.60 |
| Be, N, Ne | | pending | | | | |

- He: the bare u and χ of the two runs differ completely (u ↑↓ even changes sign), but the full J is the
  same function — same spread, same fit, same fitted parameters. The full J is the invariant.
- The He (A) residual 0.010 equals the old step 1(A) value.

**Consequences.**
- u and χ profiles separately are no target for analytic forms, with either no_dup. The only
  well-defined target is the full J = Σu + Σχ + Σf on real VMC configurations (the idea of the old step
  1 A).
- The old step 1(A) result (§5) stands: it fitted the full J, which does not depend on how the roles are
  split between u, χ and f.
- Never judge a form by its energy with no_dup = 0.
- Scripts of this check (scratchpad of the 2026-09-26 session, not in the repo): the forms fit to the
  profiles, the profile table, and the mean fields of u and f. HF densities were taken from
  `research/jastrow_form/data/densities.json` in git.

## 0b. f without cutoff (2026-09-27)

Goal: the whole Jastrow without any cutoff. u = −γ b e^{−r/b} and χ = A e^{−(r/a)²} (form 2) are kept; the
question was the f-term.

**Offline method** (`el_fit.py`, `rdm_fit.py` in the session scratchpad, not in the repo). On 20 000 fixed VMC
configurations of the CASINO no_dup = 1 Slater–Jastrow (stowfn QZ4P) PyCasino gives, per configuration, E_L,
J, ∇J, ∇²J and ∇ ln D. A model J_m changes only the kinetic energy,
E_L(m) = E_L − ½[(∇²J_m − ∇²J) + |∇ln D + ∇J_m|² − |∇ln D + ∇J|²], with ∇J_m, ∇²J_m by central differences.
Each model is fitted to J, then its **unreweighted** variance of E_L is minimized (Levenberg–Marquardt; a width
bounded by a sigmoid map, trf with bounds stops at the start). Validated: the rank-1 product after the J fit
gives var 0.0122, exactly the start of the real PyCasino run.

**He, variance of E_L after offline varmin** (CASINO polynomial on the same configurations 0.00095):

| f model | f parameters in use | var E_L | rms(J_m − J) | J far away |
|---|---|---|---|---|
| none (u + χ) | 0 | 0.0236 | 0.036 | |
| product g g h, rank 1 | 6 | 0.0092 | 0.017 | |
| product, rank 2 | 12 | 0.0058 | 0.012 | |
| **polynomial, Gaussian envelope (form 3)** | 26 γ + a | **0.00078** | 0.002 | **+88 at r1 = r2 = 6** |
| polynomial in t = (ρ̃/ρ̃(0))^{1/3}, s = r12 ρ̃(r̄)^{1/3} | 19 | 0.0037 | 0.013 | const (bounded) |

- A single physical term (hole with a position-dependent length, with or without a Gaussian of r̄ or a
  one-body sum) was worse still: 0.18–0.53 of the spread of J.
- **Products do not work:** each rank gains less (×2.6, then ×1.6). Real PyCasino emin agrees: form 2 (rank 1)
  He −2.90323, var 0.011; with order 4 the same; with the cutoff (form 1) −2.90280, var 0.0135. The envelope
  is not the problem, rank 1 is.
- **Why the polynomial wins on He:** the exact He ground state depends on (r1, r2, r12) only, and the
  Hylleraas expansion e^{−ks/2} Σ c sˡ t^{2m} uⁿ is exactly a polynomial under an exponential envelope.
- **The polynomial is not bounded:** outside the sampled region (He: 99 % of electrons within 2.8 bohr)
  unreweighted varmin makes J grow (+88 at r1 = r2 = 6 bohr).
- The density-based polynomial is bounded (t ∈ (0, 1], f → 0 as an electron leaves, no e-n cusp because
  ρ̃(r) = ρ(√(r² + ε²)), ε = 1/(2Z)) but 4–5× worse in variance: t is flat in the core.
- **Physical interpretation:** the good forms are a flexible basis; the physics is in the constraints (Kato
  cusp, no e-n cusp, decay), not in the functional form of f.

**Form 3 in PyCasino, He** (`Jastrow_emin_uncut_polynomial`, f = 0 start, varmin + 3 emin):

| cycle | E | var |
|---|---|---|
| start | −2.84301(37) | 0.140 |
| varmin | −2.90159(9) | 0.0075 |
| emin 1 / 2 / 3 | −2.90318 / −2.90366 / **−2.90356(4)** | 0.0021 / 0.00126 / **0.00113** |
| CASINO polynomial no_dup 0 / 1 | −2.90375(4) / −2.90369(5) | 0.00054 / 0.00095 |

Energy 0.13–0.19 mHa above the polynomial, variance at its level and still falling; a↑↓ = 4.01, b↑↓ = 0.67,
χ almost gone (A = 0.20), f took it (no_dup = 0).

**Form 3 fails on Be: varmin blows up.** Unreweighted varmin (cost 1.55 → 0.13 on its own configurations)
gives E = +76, variance 1755. First run: a↑↓ grew 5.66 → 7.59 and f↑↓(8, 8, 11) = 49; with a fixed during
varmin (see below) f↑↑ still reaches 31–49 at r = 8–10 bohr. A degree-9 polynomial under exp(−(r/a)²) peaks
near r ~ a√(9/2), beyond the sampled region (Be ~6–7 bohr). The CASINO cutoff avoids this: f = 0 beyond L.
Emin alone (no varmin) does not blow up but the variance does not fall (user, Be).

**Candidates in progress:**
- Bounded variables x = r/(1 + r/b): no-cusp constraints unchanged (linear terms coincide), f bounded by
  Σ|γ| bᵏ; costs a chain-rule rewrite of the f gradient and Laplacian. Not tested.
- **1-RDM variable** γ̃_ij = γ(r_i, r_j)/√(ρ(r_i)ρ(r_j)) ∈ [−1, 1], γ = Σ_occ φ_k(r_i)φ_k(r_j) (HF 2-RDM =
  ρρ − |γ|²). Bounded, knows the shells (Be: −0.18 … 1, mean 0.34; He: ≡ 1, useless — He is a
  Hylleraas problem), e-e cusp-free (γ̃ = 1 − O(r²)), e-n cusp cancels to first order for s orbitals (to be
  checked). Offline test on Be, N (`rdm_fit.py`: f = Σ c (tˡt^m + t^m tˡ) sⁿ γ̃^q, l, m ≤ 2, n, q ≤ 2, 5000
  configurations) running. In PyCasino it would couple the Jastrow to the orbitals (wfn.py rework). If γ̃
  helps, pair it with the polynomial in r (the Hylleraas part) rather than with t, s.

**Linear offline method** (`lin_fit.py`, `eval_lin.py`, scratchpad): every f model is linear in its
coefficients, f = Σ c_k B_k; B_k, ∇B_k, ∇²B_k are computed once, the Jacobian w.r.t. c is exact, only u and χ
(4–7 parameters) take a numerical one. Minutes instead of hours. Report only **reweighted** E and variance
(w = e^{2(J_m − J)}) with N_eff, evaluate on held-out configurations, probe J with all electrons at 4–15 bohr.
Unreweighted means came out below the exact energy (Be −15.008, −14.672) and were meaningless; energies are
noisy to 3–8 mHa, only the variance compares reliably.

**Be and N, offline varmin, same u and χ, variance (Be on held-out configurations):**

| f | Be features / var | N features / var | J far away |
|---|---|---|---|
| CASINO polynomial with cutoff L (fixed) | 52 / 0.038 | 78 / 0.286 | bounded |
| polynomial in r under a Gaussian, a = 0.8 L | 52 / N_eff = 1 | | grows (blows up) |
| polynomial in t, s of the density | 38 / 0.049 | | bounded |
| + 1-RDM γ̃ | 48 / 0.047 | | bounded |
| (t1 t2)^1.5 × polynomial in r, order 3 | 52 / 0.0345 | 78 / 0.329 | bounded (Be) |
| (t1 t2)^2 × polynomial in r, order 3 | 52 / 0.0376 | 78 / 0.491 | bounded (Be) |
| (t1 t2)^2 × polynomial in r, order 4 | 114 / 0.028 | not finished | bounded (Be) |
| (t1 t2)^1.5 × polynomial in r, order 4 | 114 / 0.024 | | −390 at 10 bohr |

Real CASINO varmin (polynomial u, χ, f with cutoffs, fresh samples): Be −14.6445(12), variance 0.022 — the
offline method understates every model (fixed sample, one pass), so compare models among themselves only.
- The density envelope (t1t2)^p works on Be (bounded, no free length, variance at the cutoff's level) but not
  on N (15–70 % worse, energy 22–33 mHa higher, 3–4σ); likely the spherical density switches f off too early
  in the 2p valence shell (untested). The decay must beat the polynomial: p = 1.5 fails at order 4.
- The 1-RDM variable γ̃ gave nothing measurable on Be.

## 0c. Summary (2026-09-28)

**The cutoff polynomial wins every test (He, Be, N).** Not because it is unphysical: it is the only form
that has both properties the Jastrow needs.
1. **Flexibility where the electrons are.** The variance of E_L depends on the second derivatives of J; a form
   that reproduces J itself to 1 % can have 10× the variance.
2. **No growth where no electrons are.** Varmin sees only its sample; any unbounded f is inflated outside it
   (the Gaussian-envelope polynomial on Be: E = +76).
3. The exact conditions (Kato cusp, no e-n cusp, symmetry, J → const when an electron leaves) — the
   polynomial with (r − L)^C satisfies them exactly.

| form | flexibility | bounded | result |
|---|---|---|---|
| exponential u, Gaussian χ | no | yes | 25× the variance (He) |
| product g g h, rank 1–2 | little | yes | 6–10× (He) |
| polynomial under a Gaussian | yes | **no** | He like CASINO, Be blows up |
| polynomial of density variables t, s (+ γ̃) | little | yes | 4–5× (He), 2× (Be) |
| polynomial in r × density envelope | yes | if the decay is strong enough | Be equal, N 15–70 % worse |
| polynomial with cutoff | yes | yes | best |

- **Why the "physical" forms lose:** each imposes a shape between the nucleus and the asymptotics (hole,
  bell, product, density decay) that the exact conditions do not fix; a small shape error shows in the second
  derivatives. The physics of the Jastrow is in its boundary conditions, not in the profile between them.
- **What the cutoff gives up:** the power-law tails (−A/r in u, polarization (r1·r2)/r1³ in f). Negligible for
  atoms; for extended systems CASINO adds long-range terms.
- **The real weakness of the cutoff (user's point): L is set by hand, and a bad L is bad.** The basis lives
  on [0, L] only; too small cuts correlation where electrons are, too large puts basis functions where there
  are none (the same unsampled-region problem as the Gaussian, in a milder form). The old research found the
  optimized CASINO cutoffs stay within ±10 % of the hand-chosen starts, so the optimizer does not rescue a bad
  guess, and cutoffs are fixed in varmin (`fix_cutoffs=T`). Open task: choose L from physics, e.g. from the
  radius that holds a given fraction of the HF density, or scan L and check the variance.
- **Also learned:** u, χ, f separately are not unique for either no_dup; only the full J is invariant.
  Offline comparisons need reweighting, N_eff, a held-out sample and a probe of J far away.
- **Where physics can help:** choosing L, starting values from a fit of the full J, fewer f parameters
  without losing flexibility, long-range terms for extended systems.
- **Limits:** He, Be, N only, mostly offline; forms 2 and 3 ran in PyCasino only on He and Be.

## 1. What the research found (profile level, no energies) — VOID, see §0

- **Reproducibility of the profiles.** Density-weighted, the CASINO N=8 polynomials are reproduced by
  N=2–3 to <1 %. Two emin runs of the same system (CASINO vs PyCasino) differ by 16 % (u) and 24 % (χ),
  so N=8 is far over-parameterized.
- **Backflow** η, μ and Φ/Θ profiles are not reproducible at all (noise ≥100 %). Φ/Θ holds 56 % of all
  parameters and has no simple structure.
- **u:** hole-like profiles collapse to `u = -γ b e^{-r/b}`, with γ = 1/4 (↑↑) and 1/2 (↑↓).
  - Symbolic regression (gplearn) returns exactly `-1.003·exp(-x)`.
  - b ≈ 0.8–1.4 bohr for AE atoms, independent of Z. For PP atoms b ∝ Z^-1.05.
  - Exception: **shell correlation** in parallel channels with u(0) > 0 (Be ↑↑ 1s–2s, N ↓↓). One
    exponential cannot describe them; `exp2` (hole + cusp-free tail, 3 parameters) can.
- **χ:**
  - Shape: a Gaussian bell (symbolic regression gives `exp(-0.673 x²)` with x = r/r_half).
  - Width: r_half ≈ 1.8–2.3 × the radius of the outermost shell.
  - χ ≈ α + β·V_u(r), the mean field of u over the HF density: R² 0.83–0.99 for AE atoms, i.e. χ is
    largely slaved to u.
- **f:** 94 % of its variance duplicates u and χ (`no_dup = 0` everywhere). The rest is a function of
  (r1+r2)/2 and r12 (R² 0.98).
  - Local hole radius law over all atoms: `b = 3.24 r_s/(r_s + 2.71)` (R² 0.92).
- **Cutoffs:** CASINO cutoffs stay within ±10 % of the hand-chosen starting values, so L(Z) in the data
  is not physics.

## 2. Implementation (casino/jastrow.py, casino/readers/jastrow.py)

### Forms

The form is selected per term by the constants `POLYNOMIAL = 0`, `ANALYTIC = 1`, `UNCUT = 2`:

| field | form | shape | function |
|---|---|---|---|
| `u_form` (one for u) | 1 | `(n_spin, 1)` = [b] | `-γ b e^{-r/b} w(r/L)` |
| `u_form` | 2 | `(n_spin, 1)` = [b] | `-γ b e^{-r/b}`, no cutoff |
| `chi_form[set]` | 1 | `(n_spin, 1)` = [A] | `A w(r/L)` |
| `chi_form[set]` | 2 | `(n_spin, 2)` = [A, a] | `A e^{-(r/a)²}`, no cutoff |

f forms (`f_form[set]`, constants `PRODUCT = 1`, `UNCUT = 2`, `UNCUT_POLYNOMIAL = 3`):

| form | f | parameters per spin set | cutoff |
|---|---|---|---|
| 0 | CASINO polynomial (r1−L)^C (r2−L)^C Σ γ r1ˡ r2ᵐ r12ⁿ | γ | L |
| 1 | (r1−L)^C (r2−L)^C g(r1) g(r2) h(r12) | g, h (g0 = 1, g1 = C/L, h1 = 0) | L, fixed |
| 2 | e^{−(r1/a)²} e^{−(r2/a)²} g g h | g, h, a (g1 = 0) | inf |
| 3 | e^{−(r1/a)²} e^{−(r2/a)²} Σ γ r1ˡ r2ᵐ r12ⁿ | γ, a | inf |

- `f_envelope(r, L, C, a, form)` → E, r E'/E, E''/E: (r−L)^C or the Gaussian; every f value, gradient and
  Laplacian (and the polynomial parameter-derivative loops) use it, so forms 0 and 3 share one code path.
- `f_width` reads a, stored as the last column of `f_product` (form 3: `f_product` is (spin, 1) = [a]).
- `is_product(form)` selects forms 1, 2 (the g, h parameters); the product branches test it.
- Form 3 constraints: `construct_a_matrix(0, f, −1, ...)` via `f_constraint_trunc_cutoff` — the L → ∞ limit
  of the cutoff constraints divided by −L (C terms vanish, L terms become 1): e-e
  Σ_{l+m=k}(2−δ_lm)γ_{1ml} = 0, e-n Σ_{m+n=k} γ_{nm1} = 0 (verified numerically: symmetric part ∝ ε²).
- a is optimized as ln a and only when `cutoffs_optimizable` (so `fix_cutoffs=T` in varmin fixes it).
  Its derivatives: value d1 analytic (d ln E1E2/d ln a = 2(r1² + r2²)/a²), gradient/Laplacian d1 and all d2
  by central differences (delta, delta_2). d2 is used only by emin newton.
- Reader: `Functional form` 2 or 3 sets the cutoff to (inf, False); form 3 reads `! a,<spin>,<set>` after γ,
  unset a starts at 1 bohr. no_dup flags still work for form 3.
- Fixed on the way: after the polynomial parameter-derivative loops n is set to n_start + available
  explicitly (before, a last pair outside the cutoff left n at n_start, wrong for several f sets).

- **Window:** `w(x) = (1-x)^C (1+Cx)`. Since w(0) = 1 and w'(0) = 0, the cusp is set by the function
  alone, independent of L.
- **Cusp value:** `U_CUSP[cusp_set]` (0.25, 0.5, 0.25) is taken per pair, so the Kato cusp is exact.

### Helpers (module level, njit)

- `window(r, L, C, form)` — returns (1, 0, 0) for UNCUT.
- `u_exp(r, b, γ, L, C, form)` → u, u', u''.
- `u_exp_log_b_d1(...)` → derivatives of u, u', u'' w.r.t. **ln b**.
- `u_exp_log_b_d2(...)` → second derivative w.r.t. ln b.
- `chi_value(r, params, L, C, form)` → χ, χ', χ''.
- `chi_value_d1(...)` → array(n_params, 3): derivatives w.r.t. A and ln a.
- `chi_value_d2(...)` → (A, ln a) Hessian of the value.

### Where the branches are

Every u/χ method has an `if form != POLYNOMIAL` branch:
- `*_term`, `*_term_1e`, `*_gradient(_1e)`, `*_laplacian`;
- `*_parameters_d1` (value, gradient, laplacian) and `u/chi_term_parameters_d2`;
- `fix_*`, `set_u_parameters_for_emin`, `get_parameters_scale`, `get_parameters_constraints`,
  `get/set_parameters`.

### Parameter handling

- **Log scale:** b (forms 1 and 2) and a (form 2) are stored as lengths but optimized as ln b and ln a:
  `get_parameters` returns the log, `set_parameters` exponentiates. The optimizer cannot reach b ≤ 0.
  Before this change the Be form-1 run drifted to b = −3.79, a growing exponential cut by the window.
- **No constraints:** the analytic forms have a 0-row block with the right number of columns in
  `get_parameters_constraints`. `set_parameters_projector` uses p = I when there are no constraints
  at all.
- **UNCUT cutoff:** the reader sets the cutoff to `(np.inf, False)`. All `r < L` gates pass and nothing
  is optimized; it is written back as `inf`. `max_ee_order` and `max_en_order` are at least 2, so the
  powers array always has r.

### correlation.data

An optional line in the set, placed before `Expansion order` (for χ after `Impose electron-nucleus cusp`,
which must be 0):
```
 Functional form (0=polynomial; 1=exponential hole -gamma*b*exp(-r/b)*w(r/L); 2=the same without cutoff)
   2
 Expansion order N_u
   0
```
- **Expansion order:** 0 for u; 0 for χ form 1; 1 for χ form 2 (parameters A, a).
- **Defaults:** unset b and a start from 1 bohr.
- **CASINO compatibility:** without the line the set is the CASINO polynomial, and the writer omits the
  line for polynomials, so files stay CASINO-readable.

### Tests

`casino/tests/test_jastrow.py`:
- Inputs live in `casino/tests/inputs/Jastrow/{He, He_uncut, He_product, He_uncut_product, He_uncut_polynomial}`
  (copies with a real `stowfn.data`; the example directories they came from were deleted).
- `TestJastrowUncut`, `TestJastrowProduct`, `TestJastrowUncutProduct` (form 2), `TestJastrowUncutPolynomial`
  (form 3, a fitted nonzero f so the ln a derivatives are tested; derives from `TestJastrowUncut`, no d2 test),
  tolerance rel=1e-5, abs=1e-9.
- A d2 test of form 3 failed at 1e-5 in the ln a column only: the test differentiated a finite-difference d1.
  Fixed by the analytic value d1 w.r.t. ln a, then the test was dropped (Newton is not used).
- `test_wfn_value_parameters_d1` is xfail there: with an optimized polynomial f-term the wfn-level d1
  differs by ~1e-2 from the numerical one, the same with the pure CASINO Jastrow. This is pre-existing
  and not caused by the forms.

## 3. Examples and starting values (energies here are with no_dup = 0, see §0)

**Current (2026-09-27):** `examples/stowfn/{He,Be,N,Ne,Ar,Kr,O3}/HF/QZ4P/CBCS/Jastrow_emin_uncut_polynomial` —
form 2 u, χ (from the deleted `Jastrow_emin_uncut/correlation.out.4`), form 3 f with γ = 0, a = 0.8 L of the
polynomial `Jastrow_emin` (Ar: L of its correlation.data), orders 3/3, no_dup = 0; input with
`1 method=varmin fix_cutoffs=T` + 3 emin. Compare with the CASINO polynomial with no_dup = 0.
**Deleted 2026-09-27:** all `Jastrow_emin_uncut`, `Jastrow_emin_product`, `Jastrow_emin_uncut_nodup`,
`Jastrow_emin_uncut_product` directories (git history keeps the tracked ones). The bullets below describe them.

- The form-1 examples `Jastrow_emin_analytic` are removed: form 2 superseded them.
- `.../Jastrow_emin_uncut` — form 2, written by `make_uncut_examples.py`. The user's runs of 2026-09-25
  (correlation.out.0-4, pycasino.log) started from an **empty f** (O3: f set 1 from CASINO, set 2 empty); only the
  earlier §3 Be result below used the CASINO f. Check `correlation.data` before stating what a run started from.
  These runs are the starting point of the next examples. Inputs have no `vm_filter`.
- `.../Jastrow_emin_product` — u, χ of `Jastrow_emin_uncut/correlation.out.4` and the product f fitted to its
  optimized polynomial f on cached VMC configurations (`make_product_examples.py`).
- `examples/gwfn/{He,Be,N,Ne,Ar,Kr,O3}/HF/cc-pVQZ/CBCS/Jastrow_emin_uncut` — u, χ of the stowfn
  `Jastrow_emin_uncut/correlation.out.4`, the empty polynomial f of the gwfn `Jastrow_emin`, its input without
  `vm_filter` (`make_gwfn_uncut_examples.py`). The e-n cusp comes from the cusp correction of the orbitals.
- `Jastrow_emin_uncut` with approximate starts (`make_uncut_inputs.py`), empty f, no `vm_filter`: gwfn B2H6,
  C2H2, C4H4, C6H6, CH4, H2O, HF, NH3, Li⁺, Be²⁺ and ppotential_HF B, B2H6, C, F, H, N, Ne, O. u: b = 1.0 / 0.8;
  χ per species set: (A, a) of the N and O3 runs for all-electron N and O, (1, 1.5) otherwise. The optimized f
  of these runs is the data the product (or interpolated) f is to be fitted to.
- Polynomial references `examples/gwfn/{H2O,HF,NH3,CH4}/HF/cc-pVQZ/CBCS/Jastrow_emin`: the B2H6 template (u N=8 L=6,
  χ N=8 L=5, f 3/3 L=5, spin dep 1), input identical to `Jastrow_emin_uncut`; not run yet.
- `examples/gwfn/*/HF/cc-pVQZ/CBCS/Jastrow_emin_uncut_nodup` (the 12 gwfn systems above): `Jastrow_emin_uncut` with
  both "Prevent duplication" flags = 1, f empty; not run yet.

**Result of the Jastrow_emin_uncut runs (REPORT §7.3).** ΔE vs polynomial, mHa, stowfn (4 cycles) / gwfn (8 cycles,
continuing stowfn out.4): He 0/0, Be 14.5/1.6, N 1/2, Ne 13/8, Ar 174/80, Kr 761/207 (noisy, 1e5 steps), O3 24/−18
(noisy), B2H6 −/4.
- 4 emin cycles are not enough; Ar still falls every cycle.
- **Ar, Kr: f replaces u.** b collapses to 0.1–0.4 bohr, rms f / rms u = 7–15, and f is cut at L ≈ 3–4 bohr, so
  valence pairs are left uncorrelated — the likely cause of the gap (untested: bound b from below).
- The optimized f is a local-density hole there (R²(s,r12) 0.97–1.00) — supports candidate (B).
- f mostly duplicates u and χ (R² additive 0.9–0.99); genuine three-body parts only in Be ↑↓, N ↓↓, H of B2H6.
- Product f fits f alone with R² ≥ 0.9 except Be ↑↓ (0.51) and B2H6 H ↑↓ (0.57). Fit it to the gwfn runs.
- f is not converged between runs (stowfn vs gwfn differ by 0.1–3.8 of its size).

**No-duplication flags** ("Prevent duplication of u/chi term" in the F TERM set) are fully supported:
`construct_a_matrix` equals CASINO `construct_A_sym`; they zero γ_{n,0,0} and γ_{0,0,l}, and mixed monomials can
still carry part of u and χ.

**The start matters. Fitting radial profiles one by one is not enough.**
- **f is part of the Jastrow.** The CASINO u, χ were optimized together with f, and for Be f is as
  large as u + χ (rms 1.64 vs 1.70). Profile-fitted u, χ without f make an inconsistent Jastrow, and
  varmin then flips the sign of χ (Be: A +6.9 → −10, E = −13.83).
- **Profile fits are not enough even with f:** VMC −14.467 for Be, against −14.650 for the CASINO
  polynomial.
- **What works:** `make_uncut_examples.py` samples configurations by VMC from the CASINO Slater–Jastrow,
  then fits Σu + Σχ + const to the same sum of the CASINO polynomial on them. Use Levenberg–Marquardt:
  the default trf stops after 2 evaluations on these badly scaled parameters. Refit with bounds if a
  hole radius leaves [0.1, 20] bohr (Kr ↑↑ runs to b → 0 and ends at the bound 0.1).
  - Be start from this fit: VMC −14.570.

**Be result, form 2 + CASINO f, example input (varmin + 3 emin):**
- Energy −14.6463 against −14.6504 for the CASINO polynomial; variance 0.079 against 0.042.
- Varmin makes no step (the first Gauss–Newton step is below ftol = 2/√(N−1)); emin does the work.
- b↑↑ keeps growing (3.9 → 7.1): the ↑↑ shell correlation is the remaining 4 mHa. The next step is
  `exp2` for parallel channels.

**User runs with form 1 (earlier, possibly on a stale cache, see below):**
- He equals the polynomial.
- N is 3 mHa worse (its ↓↓ channel is merged into ↑↑).
- Ne and Ar are within the polynomial's own cycle-to-cycle spread.

## 4. Traps

- **Stale numba cache.** Functions of `casino/wfn.py` are cached together with the inlined Jastrow code,
  and the cache is invalidated only by a change of `wfn.py`. After editing `jastrow.py`, run
  `find casino -name '*.nbi' -delete; find casino -name '*.nbc' -delete`. Otherwise `Wfn` keeps
  running the old Jastrow: the tests and runs look fine but are not your code. It showed up here as a
  NaN projector, and as a NaN energy with the uncut f (old code: (r − inf)^C × 0 = NaN). Compare the
  times of `casino/__pycache__/wfn*.nbi` with `jastrow.py` before debugging.
- **Unreweighted varmin on fixed configurations exploits the unsampled region.** Any unbounded f
  (polynomial under a Gaussian) can grow where no configuration lies; the energy on the next VMC is
  positive. Seen offline (He, +88 at r = 6 bohr → PyCasino E = +69) and in PyCasino varmin (Be, E = +76).
  Probe J along r1 = r2 → large r after any fit.
- **pycasino writes its log and correlation.out.* into the current directory**, not the example
  directory, when started from `examples/`.
- **Out-of-bounds write.** `get_parameters_constraints` used to build `u_matrix[0, 1]` before the form
  branch. With one parameter per channel that write is out of bounds, and numba has no bounds check.
  Fixed: the matrix is now built only in the polynomial branch. Look for similar polynomial-only index
  1 accesses when adding forms.
- **Swapped f orders in the reader.** `f_parameters_independent` passed `f_parameters.T` to `construct_a_matrix`,
  swapping N_f_eN and N_f_ee (wrong mask of independent parameters when they differ). Fixed 2026-09-26.
- **Pre-existing issues, not fixed:**
  - `fix_optimizable` tests `u_parameters.shape[1] == 3` instead of `shape[0]` for spin_dep 2, so
    absent spin channels (e.g. N) are not excluded.
  - The `Jastrow.en_powers` proxy calls `ee_powers`.
- **Running PyCasino in the cloud container:**
  - `pip install mpich` and set `LD_LIBRARY_PATH=/usr/local/lib`.
  - `pip install -e . --no-deps` (entry point metadata), then run `pycasino <dir>`. `python -m
    casino.pycasino` does nothing: there is no `__main__` guard.
  - The log goes to `pycasino.log`; the first run compiles for ~20 min.
- **The Be varmin/emin cutoff drift** (L_u −0.2 per emin cycle with form 1) disappears with form 2:
  no cutoff is optimized.

## 5. Plan — VOID except the step 1(A) result (full-J fit, valid), see §0

Order: **cheap offline filter first, implementation second**. The f-term is the target: it holds 1483 of
2149 Jastrow parameters (69 %). `exp2` is a small, local fix and waits.

### Step 1. Offline filter: fit candidates to the full CASINO Jastrow (research/, no change in casino/)

1. Extend `research/jastrow_form/make_uncut_examples.py` into a reusable fitter:
   - sample VMC configurations of the CASINO Slater–Jastrow, as now;
   - the target is the **full** CASINO J = Σu + Σχ + Σf on those configurations, plus a free constant;
   - cache the configurations per system (npz in the scratchpad) so the models can be refitted without
     re-running VMC.
2. Fit three candidates on He, Be, N, Ne, Ar, Kr, O3:
   - **(A) product f:** uncut u and χ plus f = κ(r1, r2)·g(r1)g(r2)h(r12), with κ the CASINO cutoff factor
     and g, h cubic, i.e. 10 coefficients per spin channel. Rank 1 had R² 0.98 against the CASINO f.
   - **(B) local-density hole:** u(r12; r1, r2) = −γ b̄ e^{−r12/b̄} with b̄ = B r_s/(r_s + c),
     r_s = (3/(4π ρ̄(r̄)))^{1/3} and r̄ = √((r1² + r2²)/2), where ρ̄ is the spherical HF density around
     the nucleus (`research/jastrow_form/data/densities.json`). Plus χ (Gaussian or β·V_u). No f. The
     parameters are B and c per spin channel (start 3.24, 2.71), plus χ.
   - **(C) reference:** uncut u and χ with the CASINO f fixed (the current examples).
3. Criterion: the rms of (fit − target) against the spread of the target, per system.
   - Reference points for Be: (C) gave 0.26 of 1.40 before fitting f, and it reaches −14.646 after emin.
   - A candidate passes if its residual is ≤ the (C) residual on most systems.
4. Record the results in `research/jastrow_form/REPORT.md` (a new section) and in this skill.

### Result of step 1 (A) (`research/jastrow_form/product_f.py`, REPORT §7.2)

rms residual of the fit of the full CASINO J on 50 000 VMC configurations:

| system | spread J | (C) | (A) | f params CASINO / product |
|---|---|---|---|---|
| He | 0.084 | 0.020 | 0.010 | 58 / 10 |
| Be | 0.154 | 0.229 | 0.037 | 116 / 10 |
| N | 0.142 | 0.041 | 0.029 | 174 / 15 |
| Ne | 0.162 | 0.060 | 0.039 | 116 / 10 |
| Ar | 0.194 | 0.248 | 0.088 | 116 / 10 |
| Kr | 0.191 | 0.330 | 0.120 | 116 / 10 |
| O3 | 0.341 | 0.147 | 0.086 | 232 / 20 |

- **(A) passes everywhere**, with 5 parameters per spin channel.
- **(C) is worse than the spread of J for Be, Ar, Kr:** the CASINO f fixed next to uncut u and χ is a bad start.
- **(B) is not tested yet.**
- **Kr:** b_par runs to 0 and needs a bound.
- **Cost:** the fit is slow for Kr and O3 (hours), because the Jacobian is numerical and the pair loops are
  Python loops. The configurations are cached in `$TMPDIR/jastrow_form_configs`.

### Step 2. Decision

- **(B) passes:** go to the universal term, step 3B.
- **(B) fails and (A) passes:** implement the product f, step 3A.
- **Both fail:** keep the polynomial f. Only then consider an f of lower polynomial order.

### Step 3A. Product f in casino/jastrow.py (`f_form = 1`)

1. Reader: a `Functional form` line in the F TERM set. Parameters are g (4) and h (4) per spin channel,
   stored as the (spin, 2, 4) layout; document it.
2. Value, `_1e`, gradient, Laplacian. Each is a product of 1D polynomials times the cutoff factor.
3. The e-e and e-n no-cusp conditions:
   - h'(0) = 0 fixes one h coefficient;
   - the e-n condition on g together with the cutoff factor fixes one g coefficient.
   - Derive both in FORMALISM.md and eliminate the fixed coefficients exactly, as done for u and χ.
4. Parameter d1 and d2: the form is nonlinear (a product), so the full d2 is needed for emin.
5. Tests: a new class on the He example.
6. Examples `Jastrow_emin_product` with a configuration-fit start. Run the Be emin and compare with
   −14.6504.

### Step 3B. Universal local-density term

1. The density inside the Jastrow: the spherical HF density ρ̄_I(r) of each nucleus on a radial grid.
   - Add it to the config: computed once from the orbitals, as in `density.py`.
   - Use a cubic spline so that ρ̄, ρ̄' and ρ̄'' are available for the gradient and the Laplacian.
   - Molecules: start with ρ̄ = Σ_I ρ̄_I(|r − R_I|) (promolecule).
2. The term u(r12; r1I, r2I) replaces u + f. Its derivatives w.r.t. r1, r2 go through b̄(r̄) by the chain
   rule. The e-n no-cusp condition holds automatically thanks to the rms mean r̄.
3. χ: the Gaussian, or β[V_u(r) − V_u(L)] with V_u precomputed on the same grid.
4. Parameters: B and c per spin channel (log scale), χ parameters. Analytic d1 and d2.
5. Tests, examples `Jastrow_emin_universal`, then Be, N, Ne emin against the polynomial.

### Step 4. exp2 u (independent, low priority)

`Functional form = 3` for u: −γ b1 e^{−r/b1} − A2 (1 + r/b2) e^{−r/b2}, 3 parameters per channel, with
b1 and b2 on a log scale. It targets the shell-correlated parallel channels (Be ↑↑, N ↓↓; restore spin
dep 2 for N). Expected gain: the ~4 mHa of Be and ~3 mHa of N.

### Step 5. Backflow (after the Jastrow)

Φ/Θ holds 56 % of all parameters and its profiles are not reproducible. Test by VMC first whether it
can be reduced or dropped, before looking for forms.

### Always

- Clear the numba cache after editing `casino/jastrow.py` (section 4).
- Check derivatives with the `TestJastrow*` numerical tests before any VMC run.
- Judge only by energy and variance after emin against the polynomial; a profile match is not enough.
