---
name: omega-backflow
description: >
  Use this skill when implementing or working on the Omega (Ω) term of the backflow
  transformation in PyCasino — the electron-electron-electron (e-e-e) three-body backflow
  displacement. Covers the ωijk polynomial (manual Eq. 253), the triplet displacement
  (Eq. 247), the cutoff, the no-cusp + particle-symmetry constraints, spin triplets, the
  correlation.data format, and the mapping of CASINO's pbackflow.f90 onto
  casino/backflow.py + casino/readers/backflow.py. The reader is implemented; the
  evaluation (value/gradient/laplacian) is NOT yet in casino/backflow.py.
  See also the backflow skill (eta/mu/Phi/Theta) and the qmc skill.
---

# Ω (Omega) backflow term — e-e-e correlation

The Ω term adds **electron–electron–electron** correlation to the backflow displacement.
Unlike η/μ/Φ/Θ it does **not** depend on any nucleus — it is a pure three-electron term.

**Status (2026-07):**
- CASINO Fortran: `~/bin/CASINO/src/pbackflow.f90` (~724 refs; zero in `backflow.f90`/`gbackflow.f90`).
- PyCasino **reader**: implemented in `casino/readers/backflow.py` (see below).
- PyCasino **evaluation** (`omega_term` value/gradient/laplacian in `casino/backflow.py`):
  NOT implemented yet — this is the next big step.
- Manual: `~/bin/CASINO/manual/casino_manual.pdf` §23.1.4 (the Ω function) and Eq. 247/253 —
  the PDF text extraction of Eq. 247 is garbled; the authoritative form is the Fortran below.

---

## The ω function (manual Eq. 253)

For a triplet of electrons with pairwise distances `r_ij, r_jk, r_ki`:

```
ω_ijk = f(r_jk; L)·f(r_ki; L)·f(r_ij; L) · Σ_{l,m,n} K_lmn · r_jk^l · r_ki^m · r_ij^n
f(r; L) = (1 - r/L)^C · Θ(L - r)          # single cutoff L_omega, C = trunc
```

- One cutoff `L_omega`, one truncation order `C = self.trunc` (shared with the rest of BF).
- Parameters `K_lmn` of shape `(Nω+1, Nω+1, Nω+1)` per spin triplet.
  In CASINO `omega_param(l,m,n,s)`: index `l ↔ r_jk`, `m ↔ r_ki`, `n ↔ r_ij`.
- Reuses the **e-e powers** only (no e-n powers): `r_ab^k = e_powers[a,b,k]`. Requires
  `max_ee_order ≥ Nω + 1`.

Reference routines in `pbackflow.f90`:
- `bf_omega` (:7920 region, the pure value) — the triple loop above.
- `omega_grad` (:8031) — ω and ∂ω/∂(r_jk, r_ki, r_ij).
- `omega_grad_hessian` (:8086) — Hessian w.r.t. the three magnitudes.
- `bf_omega_derivs` (:7920) — assembles Cartesian first/second derivatives via chain rule.
- `omega_eevec_grad_lap` — grad/lap of the pair magnitudes w.r.t. positions.

---

## The displacement (manual Eq. 247, reconstructed from `pbackflow.f90:5086-5091`)

The PDF rendering of Eq. 247 is corrupted. The code is unambiguous. With CASINO vectors
`ijvec = eevecs(:,jp,ip) = r_i − r_j`, `jkvec = r_j − r_k`, `kivec = r_k − r_i` (ordered
`ip<jp<kp`):

```
ξ_ip += ω · (ijvec − kivec) = ω · (2 r_i − r_j − r_k)
ξ_jp += ω · (jkvec − ijvec) = ω · (2 r_j − r_i − r_k)
ξ_kp += ω · (kivec − jkvec) = ω · (2 r_k − r_i − r_j)
```

Fully symmetric over the triplet (the `i<j<k` ordering only selects the spin-triplet
parameter set + index remap; for `spin_dep=0` it is irrelevant).

In PyCasino conventions (`e_vectors[a,b] = r_a − r_b`) this is simply:

```
ξ_i += ω · (e_vectors[i,j] + e_vectors[i,k])
```

Key contrast with η: η displaces along `r_ij` with `η(r_ij)`; Ω displaces along the sum of
two e-e vectors, and ω depends on **all three** pair distances at once.

**AE multiplier:** Ω is an e-e-e term with no nucleus centred on it, so — exactly like η —
it always lands in the "AE cutoff maybe applied" branch (`ae_cutoff_condition = 1`, the
size-2 axis index 1; see the `backflow` skill / memory `backflow-ae-multiplier-axis`).

---

## Spin triplets

`spin_dep_omega ∈ {0, 1}` for a collinear (2-spin) system — `levels_striplets = 1` because
`split_by_triplets` has a single level (`monte_carlo.f90:3325`). Noncollinear spin forces
`spin_dep = 0`.

`no_striplets(spin_dep)` = number of parameter sets:
- **spin_dep = 0 → 1 set** (all triplets share one `K_lmn`).
- **spin_dep = 1 → 4 sets** for nspin=2. The enumeration is `do j; do k≥j; do l≥k`
  (`monte_carlo.f90:3362`), i.e. multisets of size 3 from the 2 species:
  `(1,1,1), (1,1,2), (1,2,2), (2,2,2)` → sets 1..4 = `(uuu),(uud),(udd),(ddd)`.
  Per-set symmetry `omega_symm = [3, 1, 1, 3]` (via `eq_triplet`: all-equal→7→symm 3,
  two-equal→symm 1). So the two mixed-spin sets have **more** free params than the
  fully-symmetric same-spin sets.

CASINO maps `(ispin,jspin,kspin) → s = which_striplet(...)` and stores a permutation
`omega_index(:,ispin,jspin,kspin)` (:1990-2009) so the polynomial always reads params in a
canonical `i'<j'<k'` order; `eq_triplet` decides `omega_symm(s)`.

### `striplet_exists` — which sets are actually optimized

**A set's params (and its cutoff `L_s`) are only made optimizable if that spin triplet
actually occurs in the system.** The gate is `striplet_exists(s, spin_dep)` in the optable
loop (`pbackflow.f90:2119`, `if(.not.striplet_exists(s,spin_dep_omega))cycle`). This array is
built in `assign_spin_deps` (`monte_carlo.f90:3588`) with an **asymmetric** enumeration:

- same-spin triplet (uuu/ddd) needs `nele(ispin) > 2`;
- the "2-of-`ispin` + 1-of-`jspin`" triplet is registered **only for `jspin > ispin`**, i.e.
  the doubled spin is always the *lower* index;
- 3-distinct-spin triplets need `nspin ≥ 3`.

So for **Be (2 ↑, 2 ↓)** only set 2 (uud) exists:

| set | triplet | needs | Be (2↑,2↓) |
|-----|---------|-------|-----------|
| 1 | uuu | `nele(↑) > 2` | no |
| 2 | uud | ispin=↑ (≥2), jspin=↓ | **yes** |
| 3 | udd | ispin=↓ (≥2) + jspin=↑, but jspin>ispin impossible | **no** |
| 4 | ddd | `nele(↓) > 2` | no |

Consequence in optimization: `{u,d,d}` triplets physically exist but set 3 is *not* marked,
so its `K` stay frozen at 0 and `L_3` never moves. Only set 2 has optimizable params, so
**only `L_2` changes** during varmin (verified in `Backflow_omega_varmin/1_2`: all written
params are `K_...,2`, and `L_2` 4.0→2.64 while `L_1,3,4` stay 4.0). The `out` file's
"No. of free params: 10" is the *constraint-only* count from `find_determined_omega` (before
the `striplet_exists` filter); the actually-optimizable count is 4 (set 2 only).

**Port implication:** the PyCasino optimization mask for omega params/cutoffs must replicate
`striplet_exists` (per-set electron-count gate with the `jspin>ispin` rule), otherwise
PyCasino will optimize sets CASINO keeps frozen.

---

## Constraints (`construct_omega_constraints`, `pbackflow.f90:4711`)

Needed only when `K_lmn` are optimizable. Built as a matrix put into reduced echelon form;
the null space gives free params (same pattern as `construct_c_matrix` for Φ/Θ).

**1. No-cusp conditions** — as each pair distance → 0, tie the linear coefficient to the
constant one through the cutoff derivative. `3·(2·Nω + 1)` rows.

**2. Particle-symmetry constraints** — driven by `omega_symm(s)`:
- `omega_symm = 1` (two spins equal): first two indices interchangeable, `K_lmn = K_mln`.
- `omega_symm = 3` (all three equal): fully symmetric in `(l,m,n)` — generators
  `K_lmn = K_mln` and `K_lmn = K_lnm`.
- `omega_symm = 0`: none.

Related: `find_determined_omega` (:4799) counts free params; `impose_cusp_omega` (:4842)
applies the pivots; `omega_param_indx` maps `(l,m,n)→p`.

**Empirical check (Backflow_omega_dmc, spin_dep=0, Nω=4):** the written independent set is
exactly `{l ≥ m ≥ n}` = 35 params = `C(Nω+3, 3)` multisets. So for spin_dep=0 the no-cusp
rows are redundant with full symmetry and the free set is just the sorted-index reps. This
is what the current reader mask assumes; **the no-cusp reduction and the per-set symmetry
for spin_dep=1 are NOT yet handled** (see reader TODOs).

---

## Config format (`correlation.data`)

Section order in real files: **ETA, MU, PHI, OMEGA, AE CUTOFFS** (Ω is *after* Φ, before AE
cutoffs), but the reader is **order-independent** (keyed on START/END markers).

Real block (`examples/stowfn/Be/HF/QZ4P/CBCS/Backflow_omega_dmc`, spin_dep=0, Nω=4):

```
 START OMEGA TERM
 Expansion order
   4
 Spin dep
   0
 Cut-off radius ;     Optimizable (0=NO; 1=YES)
   4.4903128499482028                1       ! L_1
 Parameter ;          Optimizable (0=NO; 1=YES)
   1.0154879885304502E-002           1       ! K_000,1
   ...                                                ! K_lmn,s
 END OMEGA TERM
```

Format facts (all verified against real files):
- The trailing **comment column `! K_lmn,s` is optional** and may be absent → the reader
  reads parameters **sequentially by an independent-parameter mask**, not by comment index.
- **Parameter loop order in the file:** `s` outer, then `n`, then `m`, then `l` innermost
  (`l` = first digit of `K_lmn`). Only independent params are written.
- **Empty `Parameter` section** (header immediately followed by `END OMEGA TERM`) means
  **all K = 0** — the varmin starting point. Handled via the `try/except ValueError` fallback
  (same as eta/mu/phi).
- **Cutoff lines:** one per set (`1` for spin_dep=0, `4` for spin_dep=1), each `L value` +
  optimizable, unless the first optimizable flag is `2` (shared cutoff, single line — same
  "YES BUT NO SPIN-DEP" convention as eta).

Reference reader in Fortran: `pbackflow.f90:1935-2020` (INIT_PBACKFLOW). Writer/format:
`write_pbackflow` (~:2847), comment `K_(p3-1)(p2-1)(p1-1),s`.

---

## Reader implementation (done) — `casino/readers/backflow.py`

- `__init__`: `omega_parameters` `(0,0,0,0)`, `omega_parameters_optimizable`,
  `omega_spin_dep`, `omega_cutoff` structured `[('value',float),('optimizable',bool)]`.
- `read()`: `START/END OMEGA TERM` handling (calls `fix_omega_parameters`), and an
  `elif omega_term:` branch reading Expansion order → Spin dep → Cut-off radius(es) →
  Parameter (sequential, masked, `try/except ValueError` for empty). Array layout
  `omega_parameters[s, l, m, n]`; `number_of_sets = 1 if spin_dep==0 else 4`.
- `omega_parameters_independent(parameters)` — **symmetry-only mask `l ≥ m ≥ n`**, applied
  to every set. TODO: no-cusp rows + per-set symmetry (spin_dep=1 sets 2,3 have symm=1 so
  should have MORE free params than this mask gives).
- `fix_omega_parameters()` — symmetrizes `K_lmn` from `l≥m≥n` reps, but **only for a single
  set** (spin_dep=0). TODO: per-set symmetry + no-cusp for spin_dep=1.
- Write side: `omega_term_template` / `omega_set_template`; `write()` emits the block when
  `omega_cutoff['value'].any()`, and `terms = eta + mu + phi + omega + ae`.

Verified: reads `Backflow_omega_dmc` (35 params, symmetry `K_234==K_324==K_432`), round-trips
identically; reads the empty varmin templates as all-zero with correct shapes/cutoff counts.

**Kappa note:** `casino/backflow.py` is on the `kappa_term` WIP branch and referenced a
non-existent `config.backflow.kappa_parameters`. All kappa references there are commented
out (`# TODO: kappa term not implemented yet`) — class spec, `phi_term` (dead `kappa_poly`),
`__init__` signature/assignments, and the call site. Backflow tests pass (8) after this.

**Example dirs:** `examples/stowfn/Be/HF/QZ4P/CBCS/Backflow_omega_varmin/{0,1}_{2..6}/` —
10 varmin starting points (spin_dep × order) with empty omega sections, `input` copied from
`Backflow_varmin`, `stowfn.data` symlinked `../../../stowfn.data`.

---

## Implementation plan — remaining work in PyCasino

Python has no per-move `r2x`/`bf_rmap` machinery — `value/gradient/laplacian` are computed
in full — so Ω maps cleanly onto the `eta_term` pattern with an added `i>j>k` triple loop
(O(N³)).

1. ~~`casino/readers/backflow.py` parser~~ — **done** (see above).
2. ~~`omega_term` (value)~~ — **done**: triple loop `e1>e2>e3` (canonical `(i,j,k)=(e3,e2,e1)`),
   guard all three `r < L` with per-set `L = omega_cutoff[set % len]`,
   `set = (#down in triplet) % shape[0]`, `res[1, i] += w·(e_vectors[i,j]+e_vectors[i,k])`
   and cyclic. Early return when `omega_cutoff` is all zero (no omega block in file).
3. ~~`omega_term_gradient` / `omega_term_laplacian`~~ — **done, NUMERICAL for now**: central
   differences of `omega_term` over `e_vectors` (AbstractBackflow `numerical_gradient` /
   `numerical_laplacian` pattern, steps `delta` / `delta_2` from `casino/__init__.py`),
   recomputing `ee_powers` per shift. Analytic versions (chain rule via `omega_grad` /
   `omega_grad_hessian` / `bf_omega_derivs`, Jacobian pieces `pbackflow.f90:5328-5346`) are
   a future optimization. Verified vs full-value numerical diff on Be omega_dmc
   (gradient ~4e-11, laplacian ~1e-7) and no-omega He unaffected.
4. ~~Wire into `value`/`gradient`/`laplacian`~~ — **done**: `omega_term(_gradient/_laplacian)`
   added to `ae_value`/`ae_gradient`/`ae_laplacian`. ~~Structref + init~~ — **done**:
   `omega_parameters` / `_optimizable` / `_available` + `omega_cutoff` / `_optimizable` are in
   `Backflow_t`, filled in `backflow_init` (eta-style, single array not a list), included in
   `max_ee_order`; `fix_optimizable` applies the `striplet_exists` availability rule
   (set 3 udd always unavailable for 2-spin; verified init on He, Be omega_dmc, Be 1_2).
5. **Optimization** — extend `get/set_parameters`, `_mask`, `_scale`, `_constraints`,
   `_projector`; `omega_term_d1` may be numeric (like `phi_term_d1:1710`).
6. **Constraints** — finish `omega_parameters_independent` / `fix_omega_parameters`
   (no-cusp + per-set symmetry) and, for optimization, port `construct_omega_constraints`.

Test each derivative against a finite-difference of the value (see the `profiling` skill's
numerical-derivative tests).

---

## File map (CASINO Fortran)

All Ω logic is in `~/bin/CASINO/src/pbackflow.f90`:
- reader: `INIT_PBACKFLOW`, `START OMEGA TERM` branch (~:1935-2020); writer ~:2847
- displacement/derivs in `add_contrib_r2x` / `backflow_r2x` (~:5064, 5157, 5296, 5473)
- value/grad/hessian: `bf_omega`, `omega_grad`, `omega_grad_hessian`, `bf_omega_derivs`,
  `omega_eevec_grad_lap` (~:7920-8185)
- constraints: `construct_omega_constraints` (:4711), `omega_param_indx`,
  `find_determined_omega` (:4799), `impose_cusp_omega` (:4842)
- spin triplets: `monte_carlo.f90` `assign_spin_deps` (~:3313 `split_by_triplets`, ~:3362
  triplet enumeration, `no_striplets` / `which_striplet` / `eq_triplet`)
