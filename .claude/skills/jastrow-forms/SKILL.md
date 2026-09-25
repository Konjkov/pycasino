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

## 5. Plan

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
