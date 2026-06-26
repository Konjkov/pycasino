---
name: qmc
description: >
  Use this skill when working with any file in pycasino/casino/: wfn.py, slater.py,
  jastrow.py, backflow.py, dmc.py, vmc.py, cusp.py, harmonics.py, ppotential.py.
  Trigger on: local energy, kinetic energy, Slater determinant, Jastrow factor,
  backflow transformation, drift velocity, branching, trial wave function, VMC,
  DMC, wavefunction optimization, varmin, emin, linear method, stochastic
  reconfiguration, cusp condition, pseudopotential, T-move.
---

# QMC: pycasino

Quantum Monte Carlo (QMC) in pycasino computes properties of many-electron systems
stochastically by sampling electron configurations from the probability density `|Ψ|²`.

---

## Trial wave function

The wave function (`wfn.py`) has the Slater-Jastrow-backflow form:

```
Ψ(r) = exp(J(r)) · Φ(r')
```

where:
- `exp(J)` — Jastrow factor (`jastrow.py`): explicit e-e and e-n correlation
- `Φ(r')` — Slater determinant (`slater.py`): fermionic antisymmetry
- `r'` — backflow-transformed coordinates (`backflow.py`), optional

**Code (`wfn.py`):**
```python
def impl(self, r_e):
    e_vectors, n_vectors = self._relative_coordinates(r_e)
    res = np.exp(self.jastrow.value(e_vectors, n_vectors))   # exp(J)
    if self.backflow is not None:
        n_vectors += self.backflow.value(e_vectors, n_vectors)  # r → r'
    res *= self.slater.value(n_vectors)                       # Φ(r')
    return res
```

**Coordinate conventions:**
- `r_e` — electron positions `(nelec, 3)`, atomic units (bohr)
- `e_vectors[i,j]` = `r_e[i] - r_e[j]` — e-e vectors `(nelec, nelec, 3)`
- `n_vectors[I,i]` = `r_e[i] - R_I` — e-nucleus vectors `(natom, nelec, 3)`

---

## Local energy

The central quantity in QMC is the **local energy**:

```
E_L(r) = Ψ⁻¹ Ĥ Ψ = T_L(r) + V(r)
```

**`wfn.energy(r_e)`** (`wfn.py`):
```python
return self.kinetic_energy(r_e) + self.coulomb(r_e) + self.nonlocal_potential(r_e)
```

### Kinetic energy

Expressed through logarithmic derivatives (`∇ ln Ψ = ∇Ψ/Ψ`):

```
T_L = -½ ∇²Ψ/Ψ = -½ (∇²ln Ψ + |∇ln Ψ|²) + ...
```

pycasino uses the F and T form to minimise variance:

```python
F = s_g + j_g          # ∇ln Ψ = ∇ln Φ + ∇J
T = s_g @ s_g - s_l    # |∇ln Φ|² - ∇²ln Φ
return (T - F @ F) / 2
```

where `s_g` = `∇ln Φ` (Slater gradient), `s_l` = `∇²ln Φ` (Slater laplacian),
`j_g` = `∇J`, `j_l` = `∇²J`.

### Coulomb energy

`wfn.coulomb(r_e)` = e-e + e-n + n-n interactions + local pseudopotential channel.

```
V = Σ_{i<j} 1/|r_i - r_j|  -  Σ_{I,i} Z_I/|r_i - R_I|  +  Σ_{I<J} Z_I Z_J/|R_I - R_J|
```

---

## Slater determinant (`slater.py`)

Φ is expanded in a **multi-determinant expansion**:

```
Φ = Σ_k c_k · det[M↑_k] · det[M↓_k]
```

where `M_k[i,j]` = value of the j-th orbital at the position of the i-th electron.

All derivatives are expressed through the inverse matrix `A⁻¹`:

```
∇ln det(A) = tr(A⁻¹ · ∇A)                        → slater.gradient()
∇²ln det(A) = tr(A⁻¹ · ∇²A) - tr((A⁻¹·∇A)²)     → slater.laplacian()
```

For the multi-determinant case, a weighted sum:

```python
for i in range(self.det_coeff.size):
    c = det_coeff[i] * det(wfn_u[perm_u[i]]) * det(wfn_d[perm_d[i]])
    grad += c * tr_grad   # tr(A⁻¹ · ∂A/∂r)
return grad / val
```

**Derivative hierarchy** (each builds on the previous):

| Method | Returns | Used in |
|---|---|---|
| `value_matrix` | orbital matrix `(nelec, nelec)` | all derivatives |
| `gradient_matrix` | `∂φ_j/∂r_i^a` | `gradient`, `hessian` |
| `laplacian_matrix` | `∇²φ_j` at electron i | `laplacian` |
| `hessian_matrix` | `∂²φ_j/∂r_i^a ∂r_i^b` | `hessian`, `tressian` |
| `tressian_matrix` | `∂³φ_j/∂r_i^a ∂r_i^b ∂r_i^c` | `tressian` |
| `gradient` | `∇ln Φ` vector `(nelec·3,)` | kinetic energy, drift |
| `laplacian` | `∇²ln Φ` scalar | kinetic energy |
| `hessian` | `(H, g)`: second-derivative matrix, `∇ln Φ` | backflow |
| `tressian` | `(T, H, g)`: third, second, first derivatives | backflow + opt |

---

## Jastrow factor (`jastrow.py`)

```
J = u(e-e) + χ(e-n) + f(e-e-n)
```

- **u-term**: depends only on |r_i - r_j| — Pade-type form with cusp condition
- **χ-term**: depends only on |r_i - R_I| — e-n correlation
- **f-term**: depends on |r_i-r_j|, |r_i-R_I|, |r_j-R_I| — three-body e-e-n term

Each term is parameterised by polynomials with a cutoff radius `L`:

```
u(r) = Σ_k α_k * r^k * (r - L)²   for r < L, else 0
```

**Cusp condition for the u-term**: as r→0,
`∂u/∂r|_{r=0} = 1/4` (parallel spins) or `1/2` (antiparallel spins).
The first coefficient `α_1` is fixed by this condition and is not optimised.

---

## Backflow transformation (`backflow.py`)

Backflow replaces electron coordinates with quasiparticle coordinates:

```
x_i = r_i + ξ_i(r)
```

where `ξ_i` is a collective displacement depending on all electrons.

Kinetic energy with backflow via chain rule:

```python
b_l, b_g, b_v = self.backflow.laplacian(e_vectors, n_vectors)
s_h, s_g = self.slater.hessian(b_v + n_vectors)  # Hessian required for chain rule
s_l = np.sum(b_g * (s_h @ b_g)) + s_g @ b_l
```

`b_g` — Jacobian `∂x/∂r` of shape `(nelec·3, nelec·3)`, `b_l` — backflow Laplacian.

---

## VMC — variational Monte Carlo (`vmc.py`)

Metropolis-Hastings sampling of `|Ψ(r)|²`:

**CBCS (method=3)** — configuration-by-configuration (all electrons at once):
```python
next_r_e = r_e + step_size * uniform(-1, 1, (nelec, 3))
accept = |Ψ(r')|² / |Ψ(r)|² > random()
```

**EBES (method=1)** — electron-by-electron (Gibbs) sampling.

**Step size** `step_size` is adjusted automatically (`OPT_DTVMC=T`) to achieve ~50%
acceptance rate.

**Decorrelation**: `decorr_period` steps between recorded configurations reduces
Markov chain autocorrelation. `VMC_DECORR_PERIOD=0` triggers automatic selection.

**Observable accumulation**:
```python
# VMC.observable calls energy on each stored configuration
energy[i] = wfn.energy(position[i])   # sequential, not batched
```

---

## DMC — diffusion Monte Carlo (`dmc.py`)

DMC projects onto the exact ground state by applying `exp(-τĤ)` to an ensemble
of walkers.

### Drift-diffusion step

Each walker evolves by the Langevin equation:

```
r(t+τ) = r(t) + τ·v(r) + √τ·η
```

where:
- `v(r) = ∇ln Ψ` — drift velocity (quantum force)
- `η` — Gaussian noise (diffusion)
- `τ` — DMC time step (`DTDMC`)

### Drift velocity limiting

The drift is limited near wave function nodes (`alimit`):

```python
# dmc_alimit_vector: Umrigar limiting scheme
v_eff = v * (-1 + sqrt(1 + 2·alimit·|v|²·τ)) / (alimit·|v|²·τ)
```

Parameter `alimit` (`ALIMIT`): smaller values impose stronger limiting.

### Branching

Walkers are duplicated or killed with weight:

```
W = exp(-τ · (E_L(r) - E_T))
```

where `E_T` is the trial energy, updated to keep the walker population stable.

**Branching energy** is averaged between the current and next step to reduce
time-step error.

### T-move (pseudopotential)

With nonlocal pseudopotentials, T-move (`USE_TMOVE=T`) transfers part of the
nonlocal contribution as a new walker position with probability
`-step_size * V_NL * Ψ(r')/Ψ(r)`.

---

## Parameter optimisation (`pycasino.py`)

### Varmin — variance minimisation

Minimises `Var[E_L] = <E_L²> - <E_L>²`.

The variance gradient uses `value_parameters_d1` (= `∂ln Ψ/∂α`) and
`energy_parameters_d1` (= `∂E_L/∂α`):

```python
# σ² = <(E_L - E)²>
# ∂σ²/∂α = 2·<(E_L - E)·∂E_L/∂α> - 2·<(E_L - E)>·<∂E_L/∂α> + ...
```

### Emin — energy minimisation

**Linear method** (recommended):

Generalised eigenvalue problem:

```
H·c = ε·S·c
```

Matrices are accumulated from VMC samples:

```python
# overlap_matrix S[i,j] = <O_i O_j> - <O_i><O_j>
# hamiltonian_matrix H[i,j] = <O_i H O_j>
# O_i = ∂ln Ψ/∂α_i = value_parameters_d1[i]
S = extended_wfn_gradient.T @ extended_wfn_gradient / N
H = extended_wfn_gradient.T @ (E * extended_wfn_gradient + energy_gradient) / N
```

**Stochastic reconfiguration** (method `reconf`):
```
Δα = S⁻¹ · g
```
where `g` is the energy gradient and `S` is the overlap matrix.

---

## Units

All quantities in **atomic units**:
- Length: bohr (a₀ = 0.529177 Å)
- Energy: hartree (1 Eh = 27.2114 eV)
- Time (DMC step): hartree⁻¹

---

## Common pitfalls

**Sign of the wave function**: `slater.value()` can be negative — this is correct.
`|Ψ|²` is always positive. Never apply `abs` before squaring; that destroys the
nodal structure.

**n_vectors indexing**: `n_vectors[atom, electron, xyz]`, NOT `n_vectors[electron, atom, xyz]`.
Wrong indexing in e-n computations is a frequent source of bugs.

**∇ln Ψ vs ∇Ψ**: all methods `gradient`, `laplacian`, `hessian`, `tressian` return
logarithmic derivatives (`∇ln Ψ`, `∇²ln Ψ`), not `∇Ψ`. Confusing the two is a
serious error.

**Backflow Jacobian**: `b_g` has shape `(nelec·3, nelec·3)` — the full Jacobian of
all quasiparticle coordinates with respect to all real coordinates. With backflow,
`slater.hessian` is required (not `laplacian`) because the chain rule needs second
orbital derivatives.

**Cusp correction**: applied only when `atom_basis_type=gaussian`. Do not apply to
STO orbitals — they satisfy the cusp condition exactly.

**Multi-determinant expansion**: `det_coeff[0]=1` and `permutation_up[0]`/`permutation_down[0]`
define the leading determinant. Only `det_coeff[1:]` are optimised; the leading
coefficient is fixed.

---

## Angular part of orbitals: solid harmonics (`harmonics.py`)

The angular part of orbitals uses **real solid harmonics** (not spherical harmonics):

```
S_l^m(x,y,z) = r^l · Y_l^m(θ,φ)   with CASINO normalisation
```

Advantage over spherical harmonics: computed without trigonometry, pure algebra
over `x, y, z`.

`harmonics.value` global index: `l² + m + l` under the CASINO convention
(m = -l ... +l, ordered as in `harmonics_get_value`):

```
[0]=1, [1]=x, [2]=y, [3]=z,                          # l=0,1
[4]=(3z²-r²)/2, [5]=xz, [6]=yz, [7]=x²-y², [8]=xy,  # l=2
...
```

Two backends:
- `harmonics.py` — pure Numba, works inside `@njit`, supports up to l=4
- `sphericart.py` — C++ library `sphericart`, arbitrary l, Python wrapper only

`Slater` uses `Harmonics` (`harmonics.py`) inside JIT kernels. `sphericart.py` is
used only outside JIT (e.g., for debugging or verification).

---

## Cusp correction (`cusp.py`)

Gaussian orbitals do not satisfy the Kato cusp condition at nuclei. The correction:

```
φ̃(r) = C + sign · exp(p(r))     for r < rc
φ̃(r) = φ(r)                      for r ≥ rc
```

where `p(r) = α₀ + α₁r + α₂r² + α₃r³ + α₄r⁴` is a quartic polynomial.

**`Cusp_t` fields:**

| Field | Shape | Contents |
|---|---|---|
| `alpha` | `(5, natom, norb)` | polynomial coefficients of `p(r)` |
| `rc` | `(natom, norb)` | correction radius per orbital per atom |
| `shift` | `(natom, norb)` | constant shift `C` |
| `orbital_sign` | `(natom, norb)` | sign of the corrected orbital |

**Logic in `cusp_value`**: inside `rc`, returns `exp(p(r)) + shift` and subtracts
the uncorrected Gaussian s-part to avoid double-counting.

Cusp correction applies **only to s-orbitals** (l=0): only s-orbitals have a
nonzero value at the nucleus. p, d, f orbitals vanish as r→0.

---

## Pseudopotential (`ppotential.py`)

Replaces core electrons with an effective core potential (ECP):

```
V_PP = V_local(r) + Σ_l V_l(r) |l><l|
```

- **Local channel** `V_local`: folded into `wfn.coulomb()` as a correction to `-Z/r`
- **Nonlocal channels** `V_l`: require angular quadrature integration

**Quadrature grid** for the nonlocal integral (`ppotential_integration_grid`):
randomly oriented grid points on a sphere of radius `|r_iI|` around atom I.
Random orientation removes systematic quadrature bias.

**Nonlocal potential** (`wfn.nonlocal_potential`):
```
V_NL = Σ_l Σ_q w_q · V_l(|r_iI|) · P_l(cos θ_q) · Ψ(r_q) / Ψ(r)
```

`potential[atom][electron, l]` — value of `V_l(r)` for each atom/electron/angular momentum.

---

## Data flow between modules

```
r_e (nelec, 3)
    ↓ _relative_coordinates
e_vectors (nelec, nelec, 3)    n_vectors (natom, nelec, 3)
    ↓ jastrow                       ↓ backflow (if present)
J (scalar)                    Δn_vectors → n_vectors' (natom, nelec, 3)
    ↓                               ↓ slater
exp(J)                        value_matrix → wfn_u (norb_u, neu), wfn_d (norb_d, ned)
                                   ↓ det + det_coeff
                              Φ (scalar)
    ↓ ×
Ψ = exp(J) · Φ
```

---

## Optimisation parameters and projectors

Optimisable parameters are concatenated into a single vector via `wfn.get_parameters()`:

```
[Jastrow params | Backflow params | det_coeff[1:]]
```

**Projector** (`parameters_projector`) — matrix mapping independent parameters to
the full set, accounting for symmetry constraints (e.g., e-e Jastrow: ↑↑ and ↓↓
pairs may share parameters). This allows optimising fewer degrees of freedom without
breaking symmetry.

After each optimisation cycle, `wfn.set_parameters_projector()` recomputes the
projector for the updated cutoff lengths.

---

## Derivative call chain

```
wfn.kinetic_energy
├── without backflow:
│   ├── slater.gradient(n_vectors)    → s_g = ∇ln Φ
│   └── slater.laplacian(n_vectors)   → s_l = ∇²ln Φ
└── with backflow:
    ├── backflow.laplacian(e,n) → b_l, b_g, b_v
    ├── slater.hessian(n+b_v)   → s_h (matrix), s_g
    └── s_l = sum(b_g * (s_h @ b_g)) + s_g @ b_l

wfn.kinetic_energy_parameters_d1   (optimisation)
├── jastrow: j_g_d1, j_l_d1 → ∂T/∂α_J
├── backflow: b_l_d1, b_g_d1, b_v_d1
│   └── slater.tressian(n+b_v) → T, H, g  (third derivatives required)
└── det_coeff: slater.gradient_parameters_d1, slater.hessian_parameters_d1
```

`tressian` is called **only during backflow optimisation** — it is the most
expensive operation in the project. Returns the triple `(T, H, g)`:
- `T[a,b,c]` — third derivatives `∂³ln Φ / ∂r^a ∂r^b ∂r^c`
- `H[a,b]` — second derivatives (Hessian)
- `g[a]` — first derivatives (gradient)

---

## Tressian bottleneck: never materialise the `N³` tensor

`N = (neu+ned)·3`. The dense tressian `T` has `N³` elements: ~3 MB for O3
(`N=72`), ~10 MB for Kr (`N=108`). It no longer fits in L3, so building it and
then reading it back thrashes the cache — this is why `tressian` time grows
super-cubically (`time.dat`: 24→36 electrons is +1.5× size but +5.5× time).

**The dense tensor is never needed.** `T` has exactly one consumer
(`wfn.kinetic_energy_parameters_d1`, the backflow-parameter branch). Tracing it:

```
s_t, s_h, s_g = slater.tressian(b_v + n_vectors)         # s_t: (N,N,N)
s_h_coordinates_d1 = s_t - s_g[:,None,None]*s_h           # (N,N,N)
s_h_d1[p]          = b_v_d1 @ s_h_coordinates_d1          # (P,N,N), P = #backflow params
bf_d1[p] += sum(s_h_d1[p] * BB) / 2,   BB = b_g @ b_g.T   # (N,N), symmetric
```

`s_t` enters **only** through `Σ_bc s_h_d1[p,b,c]·BB[b,c]`. Pull the sums in:

```
Σ_bc s_h_d1[p,b,c]·BB[b,c] = Σ_a b_v_d1[p,a] · ( TBB[a] − s_g[a]·⟨s_h,BB⟩ )
where  TBB[a] = Σ_bc T[a,b,c]·BB[b,c]        # vector, size N
```

So the object actually required is **`TBB`, a length-`N` vector**, not the `N³`
tensor (nor the `P·N²` array `s_h_d1` — that contracts along axis 0 with
`b_v_d1` and is *larger*; the user's worry about "an even bigger tensor" comes
from contracting the wrong axis). Contract with `BB` over the **last two** axes,
then with `b_v_d1`.

**`TBB` is computable without ever forming `T`**, term by term from the same
decomposition `slater_tressian` already builds (`g`=tr_grad, `PH`=partial_hess):

- Outer/rank part `g[c]·PH[a,b] + g[b]·PH[a,c] + g[a]·PH[b,c]` collapses to
  `2·(PH @ (BB @ g)) + g·⟨PH,BB⟩` — pure `O(N²)`.
- Block part (`tr_tress`, `matrix_hess`, `matrix_grad` per spin) keeps `O(N³)`
  *flops* (the triple `matrix_grad` product is genuinely dense over electron
  triples) but writes only the `N`-vector — `O(N²)` memory, cache-resident.

The win is bandwidth, not flops: O3/Kr stop spilling a 3–10 MB tensor to memory.

**Implementation note:** this needs `BB` (i.e. `b_g`) inside the Slater routine,
so add a method like `slater.tressian_dot(n_vectors, b_g)` → `(TBB, s_h, s_g)`
and adapt the `wfn.py` call site. Keep the old `tressian` for the numerical
cross-check in `test_slater.py` (`tressian_v2`).

**What the existing attempts do — and why they only give 20–30 %** (branches
`improve_tressian` / `optimized_tressian`):
- replacing `harmonics` method calls with free functions — removes object
  overhead, small constant win;
- vectorising the 6-nested Python loops into `expand_dims`/broadcast — still
  allocates the full `N³` `tress` *plus* an `N³` per-spin intermediate
  (`res_u` shaped `(neu,3,neu,3,neu,3)`), i.e. *more* memory traffic, so the
  cache cliff is untouched;
- `tressian_v2` — finite-difference of `hessian` (`N` Hessian calls), also
  materialises `N³`; reference for tests only, not a perf path.

None remove the `N³` allocation, so none can beat the ~20–30 % ceiling. Only the
`TBB` contraction above changes the asymptotic memory footprint.
