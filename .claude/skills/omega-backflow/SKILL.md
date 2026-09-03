---
name: omega-backflow
description: >
  Use this skill when implementing or working on the Omega (Ω) term of the backflow
  transformation in PyCasino — the electron-electron-electron (e-e-e) three-body backflow
  displacement. Covers the ωijk polynomial (manual Eq. 253), the triplet displacement
  (Eq. 247), the cutoff, the no-cusp + particle-symmetry constraints, spin triplets, the
  correlation.data format, and the mapping of CASINO's pbackflow.f90 onto
  casino/backflow.py + casino/readers/backflow.py. Also covers the cost scaling of all
  backflow terms with electron count and expansion order. The term is fully implemented:
  reader, constraints, analytic value/gradient/laplacian and parameter derivatives,
  varmin/emin hooks.
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

**2. Particle-symmetry constraints** — driven by `omega_symm(s)`, `omega_symm = [3]` for
spin_dep=0 and `[3,1,1,3]` for spin_dep=1 (uuu,uud,udd,ddd):
- `omega_symm = 1` (two spins equal): one generator, first two indices.
- `omega_symm = 3` (all three equal): two generators — first two, and last two indices.
- `omega_symm = 0`: none (needs ≥3 spin species, impossible for collinear spin).

### ⚠ CASINO writes the symmetry rows as `+1 / +1`, making ω ANTIsymmetric

`construct_omega_constraints` (:4754-4759) sets **both** entries to `+1`:

```fortran
q=q+1 ; p=param_indx(m,l,n)
cmat(q,p0)=1._dp ; cmat(q,p)=1._dp        ! K_lmn + K_mln = 0  (!)
```

whereas `construct_H` in `pjastrow.f90:8777` — the routine this one's own comment says it
follows — uses `H(q,p0)=H(q,p0)+1 ; H(q,p)=H(q,p)-1` (a *difference*, i.e. true symmetry).

Consequences of the `+1/+1` form:
- `K_lmn = −K_mln` and (for symm=3) `K_lmn = −K_lnm` → **K is fully antisymmetric**;
- any `K` with two equal indices is **zero** (for `l=m` the row degenerates to `K_lln = 0`);
- free params for spin_dep=0 = `C(Nω+1, 3)` (the antisymmetric-tensor dimension), NOT the
  `C(Nω+3, 3)` symmetric multiset count;
- no-cusp rows are **redundant** with antisymmetry when symm=3, but do add constraints for
  symm=1.

This looks like a genuine CASINO bug: under j↔k electron exchange the pair distances
`(r_jk, r_ki, r_ij)` map as `m ↔ n`, so ω → −ω while `(r_ij + r_ik)` is unchanged — the
displacement ξ_i flips sign, so the backflow is not exchange-equivariant. **PyCasino
replicates it anyway**, because every reference run was produced with it. `+=` vs `=`
assignment is irrelevant (homogeneous system → row scaling doesn't change the null space).

**Verified exactly** against all 13 Be reference runs (`Backflow_omega_varmin/{0_2..0_9,
1_2..1_6}`): free-param count *and* the exact `K_lmn,s` label sequence match CASINO's
`correlation.out.2`, and the reconstructed full tensor satisfies `c @ K = 0` to ~1e-17.

| Nω | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----|---|---|---|---|---|---|---|---|
| spin_dep=0 free (symm=3) | 1 | 4 | 10 | 20 | 35 | 56 | 84 | 120 |
| spin_dep=1 free (set 2, symm=1) | 4 | 17 | 41 | 79 | 134 | — | — | — |

Related: `find_determined_omega` (:4799) counts free params (its count is *before* the
`striplet_exists` filter); `impose_cusp_omega` (:4842) applies the pivots; `omega_param_indx`
maps `(l,m,n)→p` with **l fastest**: `p = l + m·nd + n·nd²`.

### Version timeline — the sign was introduced at v2.13.1272

Ω is a **new, actively developed** term (first released Nov 2025). From CASINO's DIARY
(https://casinoqmc.net/DIARY_cb.txt) plus the `Omega term:` block of real run outputs:

| version | date | what |
|---------|------|------|
| v2.13.1225 | 2025-11-19 | "Implemented three-electron backflow term omega" (Clio) |
| v2.13.1241 | 2026-01-13 | reports **35** free params at Nω=4 → true symmetry, no no-cusp rows |
| v2.13.1248 | 2026-01-20 | "Ensured omega_param symmetries are maintained without no-cusp conditions" |
| v2.13.1258 | 2026-02-16 | no-cusp conditions imposed on eta+omega (Drummond) |
| **v2.13.1272** | **2026-04-09** | **symmetry constraints rewritten → `+1/+1` introduced here** |
| v2.13.1291 / 1309 / 1313 | 2026-06→07 | reports **10** free params at Nω=4 → antisymmetric |

The v2.13.1272 entry states the intent outright — *"This approach is fully analogous to the one
done in the H term. Proper symmetry constraints on the omega parameter array are now applied"* —
so `construct_H` was the model and the two `+1`s contradict the commit's own stated goal. That
plus the code-vs-comment mismatch makes this an unintentional regression, not a design choice.
No DIARY entry after v2.13.1272 touches the omega constraints.

Numerically demonstrated (independent numpy script, Be `0_2` K): exchange equivariance
`ξ_a(P_ab r) = ξ_b(r)` is violated by 2.26e-06 against max|ξ| = 1.77e-05 (~13%), while any
symmetric K satisfies it to 2.7e-19.

### PyCasino deliberately DIVERGES here (decided 2026-07-25)

`construct_omega_matrix` uses the **corrected** `+1/−1` form, i.e. true symmetry, matching
`construct_H`. Consequences, all accepted knowingly:

* **Ω parameter files are not interchangeable with CASINO** until upstream fixes the sign. Only
  the varmin *starting* templates (empty Parameter section, all K = 0) are convention-neutral,
  and that is enough: the workflow is to optimize **independently in both codes from the same
  zero start and compare final energies** — no cross-reading of results, so no guard against a
  convention mismatch was added on purpose (it would protect an impossible scenario).
* PyCasino has **more** variational freedom than CASINO, so it should win the comparison:

| Nω | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|----|---|---|---|---|---|---|---|---|
| spin_dep=0, CASINO `+1/+1` | 1 | 4 | 10 | 20 | 35 | 56 | 84 | 120 |
| spin_dep=0, PyCasino `+1/−1` | 5 | 13 | 26 | 45 | 71 | 105 | 148 | 201 |
| spin_dep=1 set 2 (symm=1), PyCasino | 8 | 26 | 57 | 104 | 170 | 258 | 371 | 512 |

Verified end-to-end after the flip (random free params → `fix_omega_parameters`): K exactly
symmetric under all index permutations (0.0), `c @ K = 0` to 2.1e-17, no-cusp residual 1.4e-17,
exchange equivariance 4.4e-16. **The no-cusp condition is a SUM over `m + n = a`** for each of
the three pair distances (as in eta/phi), *not* a per-`(m,n)` relation — easy to get wrong when
checking.

`Backflow_omega_dmc` was the only pre-1272 artifact (and the only Ω *DMC* run); deleted
2026-07-25 as unusable — pre-1272 K cannot be converted, since a symmetric tensor's
antisymmetric projection is zero (ω ≡ 0). Regenerate any Ω DMC reference from a
`Backflow_omega_varmin/{sd}_{N}` result. Reported in the forum thread
https://www.vallico.net/casino-forum/viewtopic.php?f=4&t=246.

---

## Config format (`correlation.data`)

Section order in real files: **ETA, MU, PHI, OMEGA, AE CUTOFFS** (Ω is *after* Φ, before AE
cutoffs), but the reader is **order-independent** (keyed on START/END markers).

Real block (`Backflow_omega_varmin/0_4/correlation.out.2`, spin_dep=0, Nω=4 → 10 params):

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
- `omega_parameters_independent(parameters)` — non-pivot columns of `construct_omega_matrix`
  via `rref`, per set, skipping sets `striplet_exists` rejects. Enumeration order is
  `n → m → l` (l fastest), matching `omega_param_indx`.
- `fix_omega_parameters()` — solves for the pivot (dependent) K from the free ones, same
  pattern as `fix_phi_parameters` (`rref` → `np.linalg.solve(c[:, pivots], b)`).
- `striplet_exists(spin_dep, number_of_sets)` — per-set electron-count gate with the
  `jspin>ispin` rule; needs `neu`/`ned`, hence the reader is now `Backflow(neu, ned)`
  (constructed with `self.input.neu, self.input.ned` in `readers/__init__.py`).
- Write side: `omega_term_template` / `omega_set_template`; `write()` emits the block when
  `omega_cutoff['value'].any()`, and `terms = eta + mu + phi + omega + ae`.

The port was first written in the CASINO-compatible `+1/+1` form and validated against all 13 Be
reference runs (mask reproduced CASINO's `K_lmn,s` label sequence exactly, independent values
preserved bit-for-bit, `c @ K = 0` to ~1e-17) — that is what establishes the machinery is right.
The sign was then deliberately corrected, see above. Empty varmin templates read as all-zero
with correct shapes/cutoff counts either way.

**Known gap (accepted):** the reader does not check that the number of parameter lines matches
the mask, so a CASINO-written Ω block is read silently and wrongly (it consumes only the first
`mask.sum()` lines; the rest fall through the line loop unmatched). Left in on purpose because
results are never cross-read — but it is the thing to add first if that ever changes.

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
   a future optimization. Verified vs full-value numerical diff on a Be config with real
   nonzero K (gradient ~4e-11, laplacian ~1e-7) and no-omega He unaffected. NOTE: that check
   used the since-deleted `Backflow_omega_dmc`; redo it on a `Backflow_omega_varmin/0_N`
   result, whose K obey the current convention.
4. ~~Wire into `value`/`gradient`/`laplacian`~~ — **done**: `omega_term(_gradient/_laplacian)`
   added to `ae_value`/`ae_gradient`/`ae_laplacian`. ~~Structref + init~~ — **done**:
   `omega_parameters` / `_optimizable` / `_available` + `omega_cutoff` / `_optimizable` are in
   `Backflow_t`, filled in `backflow_init` (eta-style, single array not a list), included in
   `max_ee_order`; `fix_optimizable` applies the `striplet_exists` availability rule
   (set 3 udd always unavailable for 2-spin; verified init on He and Be).
5. ~~**Constraints**~~ — **done**: `construct_omega_matrix(trunc, omega_parameters,
   omega_cutoff, spin_dep)` in `casino/backflow.py` (next to `construct_c_matrix`, returns
   `(c, cutoff_constraints)`); reader `omega_parameters_independent` / `fix_omega_parameters`
   rewritten onto it in the phi/theta style (rref → pivots = dependent, solve for them), gated
   by a new `striplet_exists(spin_dep, number_of_sets)`. Reader now takes `Backflow(neu, ned)`
   (call site in `readers/__init__.py`). Verified against all 13 Be reference runs.
6. ~~**Optimization hooks**~~ — **done** (2026-07-25), 8 places in `casino/backflow.py`:
   njit `fix_omega_parameters` overload (gated on `omega_parameters_available[s].any()`);
   omega branches in `get_parameters_mask` / `get_parameters_scale` (`6 / L**(l+m+n) / ne**3`,
   the triplet analogue of phi's `2/…/ne**3`) / `get_parameters` / `set_parameters` (which calls
   `fix_omega_parameters`) / `get_parameters_constraints` (per-existing-set block plus ∂/∂L
   columns — each set's rows depend only on its own cutoff); new `omega_term_d1` (analytic in K,
   numerical in L like phi), `omega_term_gradient_d1` / `omega_term_laplacian_d1`; omega added to
   `ae_value`/`ae_gradient`/`ae_laplacian` and to the concatenate/reduce in all three
   `*_parameters_d1` (offset after phi). Sizes verified consistent on Be `0_2`:
   `mask.size = 317 = ` constraint columns `= get_parameters(True)`, and
   `mask.sum() = 150 = get_parameters(False) = scale`.
   `value_parameters_d2` for backflow is dead code (`wfn.py` raises `NotImplementedError`) — untouched.
7. ~~**Analytic position derivatives**~~ — **done** (2026-07-25). All four are now analytic:
   `omega_term_gradient`, `omega_term_laplacian`, `omega_term_gradient_d1`,
   `omega_term_laplacian_d1`. Only the **cutoff** rows stay as finite differences of the (now
   analytic) term, exactly as eta/mu/phi do — 2 evaluations per cutoff, not per parameter.
   `delta_2` is no longer used in `casino/backflow.py`.

   Derivation (all of it follows from ω depending on positions only through the three pair
   distances, and `u_p` being linear in them):

   * `∂ξ_p,α/∂r_q,β = u_p[α]·(∇_q ω)[β] + ω·(3δ_pq − 1)·δ_αβ` — fill all 9 (p,q) blocks of the
     triplet: `outer(u_p, ∇_q ω)` plus `2ω·I` on the diagonal, `−ω·I` off it.
   * `∇_i ω = −ω_b·B/b + ω_c·C/c` and cyclically, with `A = r_j−r_k`, `B = r_k−r_i`, `C = r_i−r_j`.
   * Each electron moves only two of the three distances, so e.g.
     `∇²_i ω = ω_bb + ω_cc − 2ω_bc·(B·C)/(bc) + 2ω_b/b + 2ω_c/c`.
   * **Key simplification:** translational invariance gives `Σ_q ∇_q ω = 0`, so the whole
     first-derivative part of the laplacian collapses to
     `Σ_q ∇²_q(ω·u_p) = lap(ω)·u_p + 6·∇_p ω`.
   * With `F = f(a)f(b)f(c)`, `g_x = −C/(L−x)`, `h_x = C(C−1)/(L−x)²`:
     `ω = F·P`, `ω_a = F(P_a + g_a P)`, `ω_aa = F(P_aa + 2g_a P_a + h_a P)`,
     `ω_ab = F(P_ab + g_a P_b + g_b P_a + g_a g_b P)`.
   * The `_d1` versions are the same formulas with the polynomial replaced by the single monomial
     `M = a^l b^m c^n` of each parameter (ω is linear in K), i.e. `M_a = M·l/a`,
     `M_ab = M·l·m/(ab)`. Cosines between the pair vectors are hoisted out of the parameter loop.

   Verified against finite differences on Be `0_2` with random K at an `initial_position()`-style
   point: `bf.gradient` 5.7e-11, `bf.laplacian` 3.0e-08 (an error in the formulas would show as
   O(1), not 1e-8). Knock-on accuracy: kinetic energy 5.4e-06 → **3.0e-07** (floor without Ω is
   5.3e-08); `wfn.energy_parameters_d1` 1.9e-05 → 4.2e-05 → **2.0e-06** — the intermediate rise
   was not a regression but the loss of error cancellation once the reference became exact.
   The residual 2.0e-06 is **not** Ω: freezing the Ω cutoff changes neither the value nor the
   worst row, and the worst row (19) lies in the eta block (rows 0–19 are 1 eta cutoff + 19 free
   eta parameters). So `test_wfn_energy_parameters_d1` sitting near its default 1e-6 tolerance is
   a pre-existing backflow property.

   Reasons it was needed:
   * **accuracy** — measured on Be at a *physical* configuration (electrons on atoms), Ω vs a
     no-Ω control on the same system: `wfn.energy_parameters_d1` 2.7e-07 → **1.9e-05** (~70x),
     kinetic energy 8.6e-08 → 5.4e-06 (~60x). So `test_wfn_energy_parameters_d1` (default
     rel 1e-6) fails on an Ω config, and `test_wfn_laplacian` only just passes (4.8e-07
     relative). Backflow-level checks are fine: `value_parameters_d1` 2.2e-15,
     `gradient_parameters_d1` 2.3e-10, `laplacian_parameters_d1` 2.6e-07.
     **Pitfall:** an `uniform(-1.5,1.5)` configuration gives rel 1.7e-02 *with or without* Ω —
     finite differences of the local energy are meaningless there. Always compare at an
     `initial_position()`-style point, and keep a no-Ω control before blaming Ω.
   * **cost** — numerical derivatives multiply Ω by 6N term evaluations. Rough per-configuration
     estimate: Be/Nω=2 ~1e4 ops (irrelevant), **Ne/Nω=9 ~3e7 ops** (Ω dominates everything else
     by orders of magnitude, varmin becomes impractical). So analytic derivatives are what makes
     the Ne comparison feasible at all, more than an accuracy matter.
   Port from `pbackflow.f90`: `omega_grad` (:8031), `omega_grad_hessian` (:8086),
   `bf_omega_derivs`, `omega_eevec_grad_lap` — chain rule through the three pair magnitudes;
   Jacobian pieces at :5328-5346.

8. ~~**Fold the particle symmetry into the enumeration**~~ — **done** (2026-07-26), this removed
   the high-order cost wall.

   Cost of the *old* `fix_omega_parameters` (njit, called on **every** `set_parameters`):
   Nω=4 → 6 ms, Nω=6 → 102 ms, **Nω=9 → 2590 ms** (≈43 min per 1000 calls), of which **96% is
   `rref`** on a 3(2Nω+1) + 2·nd³ by nd³ matrix (2057×1000 at Nω=9). `construct_omega_matrix`
   0.5%, building `b` 1.6%, `np.linalg.solve` 0.7%. So the practical ceiling was Nω≈6.
   `fix_phi_parameters` is 2.8 ms — never the problem.
   **After the fold: Nω=9 → 0.30 ms/call, i.e. ~8600× faster** (more than the 745× matrix-size
   ratio, because `rref` is superlinear). Order is no longer a cost consideration at all.

   The symmetry rows are ±1 integers and 97% of the matrix; they encode a *relabelling*, not a
   linear system. The Jastrow f-term already does the right thing —
   `construct_a_matrix` enumerates only `l ≥ m` (`parameters_size =
   (f_en_order+1)(f_en_order+2)(f_ee_order+1)//2`), has **no** symmetry rows, and weights a
   representative by the number of entries it stands for (the `1` vs `2` at jastrow.py:41-45).
   So this is consistency with the project, not invention. CASINO does *not* do it for omega
   (`impose_cusp_omega` rebuilds and re-echelons everything on every `put_pbf_params`), though
   its own H term caches (`Hpivot` once, `Hmatrix` only when `L_h` moves).

   Implemented as `construct_omega_folded_matrix` in `casino/backflow.py`, used by the njit
   `fix_omega_parameters` overload and by the reader's `fix_omega_parameters` /
   `omega_parameters_independent`. `construct_omega_matrix` is **kept** — `get_parameters_constraints`
   still needs the full parameter space for the projector, and it runs once per
   `set_parameters_projector`.

   Algorithm:
   * **Representative** of an orbit = the entry the current code leaves free = the one with the
     largest `p = l + m·nd + n·nd²`, i.e. **ascending** indices: `sorted((l,m,n))` for symm=3,
     `(min(l,m), max(l,m), n)` for symm=1. (Confirmed empirically: PyCasino writes `K_002`, not
     `K_200`.)
   * Enumerate representatives in increasing `p` (loops `n` outer, `m`, `l` inner, keeping those
     equal to their own representative). Counts: symm=3 → `C(nd+2,3)` (220 at Nω=9);
     symm=1 → `nd·nd(nd+1)/2` (550).
   * Build **only** the 3(2Nω+1) no-cusp rows, *accumulating* each tensor entry's coefficient
     onto its representative's column — accumulation makes the 1/3/6 multiplicities appear by
     themselves, which removes the one real risk of this approach. Matrix becomes 57×220 / 57×550
     at Nω=9, i.e. **745× / 119× less work**. (The three row families coincide under full
     symmetry; let `rref` zero the duplicates rather than reasoning about it.)
   * `rref`, then **no solve**: the form is *reduced*, so the pivot submatrix is exactly the
     identity (verified: deviation 0.0e+00 for omega and phi alike). Dependent representatives
     are `x[pivot_of_row] = −Σ_{q free} a[row,q]·x[q]` — CASINO's own formula.
   * Broadcast: `K[l,m,n] = x[column_of_representative(l,m,n)]`.

   **Validated against the previous (full-matrix) implementation**, with repo code: identical
   free-parameter masks for Nω=2..9 × symm∈{3,1}, 16/16 (free counts 5, 13, 26, 45, 71, 105, 148,
   201 for symm=3); fixed tensors identical (worst 3.6e-15) and satisfying the **full** constraint
   matrix, symmetry rows included, to 3.6e-15. End-to-end on Be `0_{2,4,6,8,9}`: `get/set_parameters`
   round-trip exact, `|c@K|` ≤ 3.6e-15, correlation.data write→read exact with the same mask.

   Note the pitfall in the back-substitution loop: `x[p] -= c[row, q]·x[q]` over *all* q silently
   zeroes the accumulated value when `q == p` (since `c[row, p] = 1`). Accumulate into a local and
   skip `q == p`.

   Still outside Ω: dropping the redundant `np.linalg.solve` from `fix_phi_parameters` is the same
   free win — ask before touching it.

Test each derivative against a finite-difference of the value (see the `profiling` skill's
numerical-derivative tests). **Clear the numba cache first** — see
[[numba-cache-stale-across-modules]]: editing an overload here does not invalidate cached njit
functions in `abstract.py`, and a stale parameter count shows up as a bogus shape mismatch.

---

## Cost scaling of the backflow terms

Read off the loop structure in `casino/backflow.py`; `value`, `gradient` and `laplacian` share it,
only the per-element work differs.

| term | enumerated | count | per element | total |
|------|-----------|-------|-------------|-------|
| η | unordered pairs | `N(N−1)/2` | `O(N_η)` | `O(N²·N_η)` |
| μ | electron × nucleus | `N·N_I` | `O(N_μ)` | `O(N·N_I·N_μ)` |
| Φ/Θ | **ordered** pairs × nucleus | `N(N−1)·N_I` | `O(N_eN²·N_ee)` | `O(N²·N_I·N_eN²·N_ee)` |
| Ω | unordered triplets | `N(N−1)(N−2)/6` | `O(Nω³)` | `O(N³·Nω³)` |

Φ/Θ runs over ordered pairs (both directions) because `Φ r_ij + Θ r_iI` is not symmetric — a
factor 2 over η.

Ω is the only term that lifts backflow's own scaling from `O(N²)` to `O(N³)`. The determinant
algebra and the Jacobian contraction are already `O(N³)`, so Ω does not change the **order** of a
QMC step — only the prefactor, and the prefactor is the problem: per-element work grows as the
**cube** of the expansion order.

| system | pairs | triplets | η monomials (N_η=9) | Ω monomials (Nω=9) | Ω/η |
|--------|-------|----------|---------------------|---------------------|-----|
| Be (N=4) | 6 | 4 | 54 | 4 000 | 74× |
| Ne (N=10) | 45 | 120 | 405 | 120 000 | 300× |
| Ar (N=18) | 153 | 816 | 1 377 | 816 000 | 590× |

So at high Nω the Ω term dominates the whole backflow long before `N³` itself matters; dropping
Nω 9→4 is an immediate ×8 on this factor.

**Cutoffs do not save an atom.** Every term tests `r < L` *before* the polynomial, so for an
extended system the actually-working pairs/triplets are `O(N)`. But the enumeration is still
complete — the `e1>e2>e3` loops pay `O(N³)` in bare tests, there are no neighbour lists. For an
isolated atom, where all distances are below `L`, there is no saving at all.

**Parameter derivatives.** Same order in `N`; the monomial loop *is* the parameter loop, so no
extra factor in Nω either, just a bigger constant (Ω writes 9 3×3 blocks per parameter per
triplet). The real cost is memory: `*_gradient_d1` returns `(n_params, 2, N, 3, N, 3)`, i.e.
`O(n_params·N²)`, and for Ω `n_params ~ Nω³/6` → `O(Nω³·N²)` (~6 MB per call for Ne at Nω=9).

Ω parameter count: `C(Nω+3, 3)` orbit representatives (220 at Nω=9, 201 of them free for
spin_dep=0) — cubic in the order, against linear for η.

Note the symmetry fold (step 8) only removed the `fix_omega_parameters` cost. It does **not**
touch the `O(N³·Nω³)` per-configuration cost of evaluating the term.

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
