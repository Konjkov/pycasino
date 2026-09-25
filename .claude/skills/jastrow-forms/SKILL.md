---
name: jastrow-forms
description: >
  Use this skill when working on the analytic (non-polynomial) Jastrow forms in PyCasino: the
  exponential correlation hole u = -gamma b exp(-r/b), the bell chi = A w(r/L) and the Gaussian
  chi = A exp(-(r/a)^2), with or without cutoff ("Functional form" 1 and 2 in correlation.data),
  their implementation in casino/jastrow.py and casino/readers/jastrow.py, the examples
  examples/stowfn/*/HF/QZ4P/CBCS/Jastrow_emin_analytic and Jastrow_emin_uncut, how their starting
  values are made, and the research behind them in research/jastrow_form (profile fits of the
  CASINO u, chi, f, eta, mu, Phi, symbolic regression, local-density hole, mean-field chi).
  Also covers the traps met on the way: stale numba cache of wfn.py, a negative hole radius,
  varmin making no step, the f-term being as large as u + chi for Be. Trigger on: functional
  form, exponential u, bell chi, Gaussian chi, uncut, no cutoff, hole radius, u_form, chi_form,
  make_uncut_examples, Jastrow_emin_analytic, Jastrow_emin_uncut, jastrow_form research.
  See also the qmc skill (Jastrow in the wave function) and the numba skill.
---

# Analytic Jastrow forms (u, chi) in PyCasino

Branch `jastrow_functional_form`. The study is in `research/jastrow_form/`: `REPORT.md` has the results,
and `FORMALISM.md` has the mathematics (window, cusp conditions, weights, mean-field χ, local-density
hole).

## 1. What the research found (profile level, no energies)

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
- `TestJastrowAnalytic` and `TestJastrowUncut` run on the He examples, with tolerance rel=1e-5, abs=1e-9.
- `test_wfn_value_parameters_d1` is xfail there: with an optimized polynomial f-term the wfn-level d1
  differs by ~1e-2 from the numerical one, the same with the pure CASINO Jastrow. This is pre-existing
  and not caused by the forms.

## 3. Examples and starting values

- `examples/stowfn/{He,Be,N,Ne,Ar,Kr,O3}/HF/QZ4P/CBCS/Jastrow_emin_analytic` — form 1.
  - Written by `research/jastrow_form/make_examples.py` (N..O3) or by hand (He, Be).
  - The CASINO f-term was inserted later. The user may have edited these files (e.g. zeroed f for Be).
- `.../Jastrow_emin_uncut` — form 2 + the CASINO f-term, written by `make_uncut_examples.py`.

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
  NaN projector.
- **Out-of-bounds write.** `get_parameters_constraints` used to build `u_matrix[0, 1]` before the form
  branch. With one parameter per channel that write is out of bounds, and numba has no bounds check.
  Fixed: the matrix is now built only in the polynomial branch. Look for similar polynomial-only index
  1 accesses when adding forms.
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

## 5. Open directions

- **`exp2` u** for shell-correlated parallel channels: 3 parameters, cusp-free tail/bump.
- **Local-density hole** replacing u + f: u(r12; r1, r2) = −γ b̄ e^{−r12/b̄} with b̄ = b(r_s(ρ(r̄))) and
  r̄ = √((r1² + r2²)/2). The rms mean keeps the e-n no-cusp condition. Two universal parameters;
  FORMALISM.md §7.
- **Density-preserving χ** = β[V_u(r) − V_u(L)] w: one parameter per species.
- **Backflow:** decide by VMC whether Φ/Θ can be reduced or dropped.
