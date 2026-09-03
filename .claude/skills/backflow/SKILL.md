---
name: backflow
description: >
  Use this skill when working on the backflow transformation in casino/backflow.py or
  casino/readers/backflow.py: the eta/mu/Phi/Theta displacement terms, the cutoff and
  truncation order C, AE-cutoff multiplier g(r), Kato cusp constraints, irrotational
  backflow, construct_c_matrix, or the backflow Jacobian used in the kinetic energy.
  This is the physics/math reference (López Ríos et al. 2006) mapped to the code.
  See also the qmc skill for how backflow enters the wave function and local energy.
---

# Backflow transformation (`casino/backflow.py`)

Backflow (BF) replaces the electron coordinates fed into the **Slater determinant** with
collective quasiparticle coordinates, improving the nodal surface (and thus the fixed-node
DMC energy). The Jastrow factor is left untouched.

```
Ψ_BF(R) = exp(J(R)) · Φ(X),   x_i = r_i + ξ_i(R)
```

`ξ_i` is the backflow displacement of electron `i`, depending on the whole configuration.
Primary reference: **`pdfs/backflow/backflow.pdf`** (López Ríos, Ma, Drummond, Towler,
Needs, *Inhomogeneous backflow transformations in QMC*, PRE **74**, 066701 (2006)). The
equation numbers below refer to that paper. The constraint code is a direct port of
CASINO's `pbackflow.f90` (`pdfs/backflow/pbackflow.f90`).

---

## The three displacement terms

The total displacement is a sum of e-e, e-n and e-e-n contributions (Eqs. 4–6):

```
ξ_i = ξ_i^ee + ξ_i^en + ξ_i^een
```

| Term | Eq. | Function of | Code (`impl`) |
|------|-----|-------------|---------------|
| `η` e-e   | 4  | `η(r_ij)·r_ij`                         | `eta_term` |
| `μ` e-n   | 5  | `μ(r_iI)·r_iI`                         | `mu_term` |
| `Φ,Θ` e-e-n | 6 | `Φ^{jI}·r_ij + Θ^{jI}·r_iI`           | `phi_term` |

`Φ^{jI} r_ij + Θ^{jI} r_iI` spans the plane of `r_i, r_j, r_I`, so no separate
perpendicular component is needed. `Θ` is often slightly more important than `Φ`.

CASINO also has a three-electron **Ω** (e-e-e) term and a **Π** (periodic) term. Ω is being
ported — its reader lives in `casino/readers/backflow.py`; the evaluation is not in
`casino/backflow.py` yet. See the **`omega-backflow`** skill. Note: the `kappa_term` WIP
branch left `kappa_parameters` references in `casino/backflow.py` commented out (no reader
support yet).

### Polynomial + cutoff form (Eqs. 7–11)

Each radial function is a **natural power expansion** multiplied by a smooth cutoff:

```
f(r; L) = (1 - r/L)^C · H(L - r)          # Eq. 7, truncation order C = self.trunc
η_ij   = f(r_ij; Lη)  · Σ_k c_k r_ij^k                          # Eq. 8
μ_iI   = f(r_iI; Lμ)  · Σ_k d_k r_iI^k                          # Eq. 9
Φ_i^jI = f(r_iI;LΦ)f(r_jI;LΦ) · Σ_klm φ_klm r_iI^k r_jI^l r_ij^m   # Eq. 10
Θ_i^jI = f(r_iI;LΦ)f(r_jI;LΦ) · Σ_klm θ_klm r_iI^k r_jI^l r_ij^m   # Eq. 11
```

- `C = self.trunc` (only `C = 2` and `C = 3` are used; the C-th derivative of `f` is
  discontinuous at `r = L`, so the Laplacian in the kinetic energy is discontinuous for `C < 3`).
- Cutoffs: `eta_cutoff` (Lη), `mu_cutoff` (Lμ), `phi_cutoff` (LΦ). Optimizable via the
  `*_cutoff_optimizable` flags.
- Parameters are **spin-pair dependent**: `eta_set / mu_set / phi_set` index into the
  parameter array by counting how many of the pair are down-spin
  (`int(e1>=neu)+int(e2>=neu)`), mod the number of spin channels.
- Powers `r^k` are precomputed once per configuration: `ee_powers` → `e_powers`,
  `en_powers` → `n_powers`, then `poly = parameters @ powers[:k_max]`.

**Code pattern** (`eta_term`, all terms follow it):
```python
poly = parameters[eta_set] @ e_powers[e1, e2, :k_max]
bf = (1 - r / L) ** C * poly * r_vec
res[ae_cutoff_condition, e1] += bf
res[ae_cutoff_condition, e2] -= bf   # η is antisymmetric in the pair
```

---

## AE-cutoff multiplier `g(r)` and the size-2 axis

Near an **all-electron (AE) nucleus** the e-n Kato cusp cannot be satisfied unless the
*total* displacement vanishes at the nucleus. So every contribution that is **not**
already centered on that nucleus is multiplied by a smooth cutoff `g(r_iI)` (Eq. B1):

```
g(r) = (r/Lg)² · [6 - 8(r/Lg) + 3(r/Lg)²],   r < Lg;   g = 1 otherwise   # Lg = ae_cutoff
```

`g`, `g'` are continuous at `r = Lg`, and `g(0) = g'(0) = 0`. Implemented in
`ae_multiplier` / `ae_multiplier_gradient` / `ae_multiplier_laplacian`.

**Critical convention** — the leading size-2 axis of the term arrays (`res[2, nelec, 3]`)
is **NOT spin**. It is the AE-cutoff branch:
- index `0` — "AE cutoff exactly **not** applied" (contribution centered on this nucleus,
  or `r > ae_cutoff`);
- index `1` — "AE cutoff **maybe** applied" (multiply by `g(r_iI)`).

`ae_cutoff_condition = int(r > self.ae_cutoff[label])`. The `value`/`gradient`/`laplacian`
methods combine the two branches with the `ae_multiplier`. For PP atoms `ae_cutoff = 0`,
so everything lands in branch 1 with `g = 1` (no-op). See memory
`backflow-ae-multiplier-axis`.

---

## Cusp & irrotational constraints (`construct_c_matrix`)

The BF parameters are **constrained** so the transformation does not spoil the Kato cusp
conditions already enforced by the Slater/Jastrow parts (Appendix A). `construct_c_matrix`
builds the homogeneous constraint matrix `C·p = cutoff_constraints`; the null space
(`rref` / `parameters_projector`) gives the free parameters. This is a line-by-line port of
`SUBROUTINE construct_C` in `pbackflow.f90`.

Two boolean flags per Φ/Θ channel drive the constraint count:

- **`phi_cusp`** — AE atom (`True`) vs PP atom (`False`).
  - AE: `phi_constraints = 6·(en+ee)-2`-type rows enforce that `Φ,Θ` and their gradients
    vanish appropriately at the nucleus (Eqs. A1, A4, A5).
  - PP: fewer rows, and the cutoff enters via `-C/phi_cutoff` terms (`cutoff_constraints`).
- **`phi_irrotational`** — impose `∇×ξ = 0` so `ξ = ∇Y` derives from a scalar backflow
  potential (Appendix A2, Eqs. A6–A8). Adds `((eN+3)(ee+2)-4)(eN+1)` rows (minus a block
  when `trunc == 0`), roughly halving the free Φ/Θ parameters. Gives poor results for
  atoms (e.g. carbon) — use with care.

Cusp-related building blocks in the matrix:
- e-e cusp (EKCC, Eqs. A2/A3): rows active for like-spin / `spin_dep in (0,2)` channels,
  keyed on `m == 1` (the `r_ij^1` term).
- e-n cusp (NKCC): rows keyed on `k == 0/1` (Φ) and `l == 0/1` (Θ), i.e. the low e-n
  power terms, with `-trunc/phi_cutoff` factors for PP.

When editing `construct_c_matrix`, keep the offset bookkeeping (`ee_constrains`,
`en_constrains`, `offset`) in exact correspondence with `pbackflow.f90` — the row indices
are hand-computed and easy to break silently.

---

## Value, gradient, Laplacian and the kinetic energy

- `value(e_vectors, n_vectors)` → total displacement `(nelec, 3)` added to `n_vectors`
  before the Slater determinant (`wfn.py`).
- `gradient` → the **Jacobian** `∂x/∂r` of shape `(nelec·3, nelec·3)` (`b_g`).
- `laplacian` → returns `(b_l, b_g, b_v)`: backflow Laplacian, Jacobian, displacement.

Because each quasiparticle depends on **all** electrons, the kinetic energy needs the
Slater **Hessian** and the chain rule (see the qmc skill):
```python
b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
s_h, s_g = self.slater.hessian(b_v + n_vectors)
s_l = np.sum(b_g * (s_h @ b_g)) + s_g @ b_l
```
This also breaks the EBEA fast determinant updates: with BF every electron move
recomputes the determinants via LU (CBCA-style), which is the main cost of BF.

Parameter optimization (varmin/emin) goes through `get_parameters` / `set_parameters` /
`get_parameters_mask` / `set_parameters_projector`; the projector keeps moves inside the
constraint null space. The `_d1` methods (`eta_term_d1`, …) are parameter derivatives used
by the linear/emin method.

---

## Reference PDFs (`pdfs/backflow/`)

**Primary — matches the code directly:**
- `backflow.pdf` / `lopezrios_backflow.pdf` — López Ríos et al., PRE **74**, 066701 (2006).
  Eqs. 4–11 = the displacement terms; Appendix A = the constraints in `construct_c_matrix`;
  Appendix B (Eq. B1) = `ae_multiplier`. **Start here.**
- `pbackflow.f90` — CASINO Fortran source that `construct_c_matrix` and the term
  evaluators were ported from. The authoritative index bookkeeping.

**Background / origins:**
- `pandharipande1973.pdf` — Pandharipande & Itoh, PRA **8**, 2564 (1973): effective mass of
  ³He, the quantum backflow idea (Feynman–Cohen lineage).
- `pandharipande1986.pdf` — Pandharipande, Pieper, Wiringa, PRB **34**, 4571 (1986): VMC of
  ⁴He/³He drops with Feynman–Cohen backflow.
- `schmidt1990.pdf` — Schmidt & Moskowitz, JCP **93**, 4172 (1990): correlated wave
  functions for atoms He–Ne (e-e-n correlation forms).
- `holzmann2003.pdf` — Holzmann, Ceperley, Pierleoni, Esler, PRE **68**, 046707 (2003):
  analytic backflow + three-body for the electron gas / metallic hydrogen.

**Extensions / theses:**
- `single_vector.pdf` — P. López Ríos PhD thesis, *Backflow and pairing wave function for
  QMC* (Cambridge, 2006). Full derivations behind the paper.
- `backflow_wfn.pdf` — P. Seth PhD thesis, *Improved wave functions for QMC* (Cambridge,
  2012).
- `orbital_backflow.pdf` — Holzmann & Moroni, *Orbital-dependent backflow* (arXiv:1910.07167).
- `lopezrios_nov12.pdf` — Seth/López Ríos/Needs slides on orbital-dependent backflow (2012).
