---
name: nodal-surface
description: >
  Use this skill when working on the nodal surface of the trial wave function in PyCasino —
  the `nodal_domains` branch, `wfn.second_fundamental_form`, projection of a sample point onto
  the node, principal curvatures, the co-area formula and nodal hypersurface integrals, the
  two-nodal-cell property, weighted nodal domain averages (Mitas & Annaberdiyev), and the
  relation between node curvature and fixed-node error. Covers the literature in
  pdfs/nodal_surface, the analytically known nodes used as test cases, the defects found in the
  branch, and the staged work plan. Trigger on: nodal surface, node, nodal domain, nodal cell,
  fixed-node error, second fundamental form, principal curvature, co-area, nodal hypersurface
  integral.
  See also the qmc skill for the wave function and local energy, and the backflow skill for how
  backflow moves the node.
---

# Nodal surface of the trial wave function

The fixed-node approximation makes DMC exact up to one thing: the node of the trial function,
the `3N-1` dimensional hypersurface `Ψ(R) = 0`. Everything in this skill is about measuring the
geometry of that surface and turning the measurement into something useful.

**Status (2026-08):**
- Branch `nodal_domains`, last commit 2026-06-06. Not merged; 7 files, ~130 lines added.
- `wfn.second_fundamental_form` — implemented, **unvalidated**, several defects (below).
- `Casino.nodal_domains()` — a stub that prints averages and `return 2`.
- Working notes: `nodal_surface.txt` at repo root (analytic nodes; folded into this skill).
- Literature: `pdfs/nodal_surface/` (index at the end).

---

## The settled question: do not spend time counting nodal cells

The number of nodal cells of a fermionic ground state is **two**. This is not open:

- Ceperley, *Fermion Nodes*, J. Stat. Phys. 63, 1237 (1991) — `ceperley1991.pdf`. States the
  conjecture and the tiling/exchange criterion used to prove it.
- Mitas, *Structure of Fermion Nodes and Nodal Cells*, PRL 96, 240402 (2006) — `mitas2006.pdf`,
  preprint `0601485.pdf`. **Proves** two nodal cells for spin-polarised noninteracting fermions
  in a harmonic well at arbitrary system size, and extends it to mean-field models in other
  geometries and to Hartree–Fock atomic states.
- Mitas, *Fermion nodes and nodal cells of noninteracting and interacting fermions* —
  `0605550.pdf`. Extends to the interacting case.
- Bressanini, *Implications of the two nodal domains conjecture*, PRB 86, 115120 (2012) —
  `Be_nodes.pdf`.

So `nodal_domains()` returning the literal constant `2` is the correct answer, not a stub to be
replaced by a counting algorithm. **Do not build a domain counter.** If a request sounds like
"count the nodal domains", the answer is a citation.

---

## What is actually live: weighted nodal domain averages

The open direction is **integrals over the nodal hypersurface**, not the count of cells.

**Mitas & Annaberdiyev, *Weighted nodal domain averages of eigenstates for quantum Monte Carlo
and beyond*, arXiv:2109.01734 (2021)** — `2109.01734.pdf`. This is the roadmap paper. From the
abstract:

> …we introduce weighted nodal domain averages that provide a new probe of nodal surfaces beyond
> the usual expectations. Particular choices for the weight function reveal, for example, that
> the difference between two arbitrary fermionic eigenvalues is given by the nodal hypersurface
> integrals normalized by overlaps with the bosonic ground state… Variational formulations that
> employ different weights are proposed for prospective improvement of nodes in variational and
> fixed-node diffusion Monte Carlo calculations.

Their central result is Eq. (20): with the weight function set to the **bosonic ground state**
`Φ⁰_B` of the same Hamiltonian, the energy of any eigenstate — fermionic or excited bosonic — is

```
E = [ ∫_∂Ω Φ⁰_B(R) |∇_R Ψ(R)| dS ] / [ ∫ Φ⁰_B(R) |Ψ(R)| dR ] + E_0B
```

Read the integrand carefully, because it determines the whole design:

- **No curvature appears.** The surface integral needs only the bosonic weight and `|∇Ψ|` at the
  node. No second fundamental form, no principal curvatures, no normal-map Jacobian. The
  `second_fundamental_form` machinery is **not** on the path to this result.
- **The denominator carries `|Ψ|`, not `Ψ²`.** The natural measure of the formalism is `Φ_B·|Ψ|`,
  which vanishes only *linearly* at the node — or `Φ_B` itself, which is **nodeless**. The paper
  makes the point explicitly: "Φ⁰_B is nodeless. Therefore it is possible to obtain samples of
  the exact solution and its energy without the fundamental problem of fermion signs."

Their illustration system is the **Be atom**, both noninteracting and fully interacting, and
PyCasino already has Be examples in `examples/`. That is a ready-made point of comparison.

The companion question — *does node curvature predict fixed-node error?* — is
**Rasch, Hu & Mitas, *Fixed-node errors in QMC: interplay of electron density and node
nonlinearities*, arXiv:1310.2311** (`1310.2311.pdf`), with the lithium follow-up
`1502.07248.pdf`. The `HdF = H·|∇ψ|` quantity in the branch is an attempt at such a measure.

For *acting* on the measurement rather than just reporting it:
**Lüchow, Petz & Scott, *Direct optimization of nodal hypersurfaces in approximate wave
functions*, JCP 126, 144110 (2007)** — `10.1063@1.2716640.pdf`. Same group as the
`full_wfn_optimization.pdf` thesis referenced in the `qmc` skill.

---

## Current branch state — code map

`git diff main...nodal_domains`

### `casino/wfn.py` — `wfn_second_fundamental_form(self, r_e)`

Per sample point:

1. **Reject far points.** `dist = 1/|∇ψ/ψ|` is the approximate normal distance to the node.
   If `dist >= epsilon` return zeros — the point is outside the tube.
2. **Project onto the node.** Newton iteration `r_e -= grad/|grad|²`, i.e. `Δr = -ψ∇ψ/|∇ψ|²`,
   up to 40 steps until `|ψ| < 1e-10`.
3. **Tangent projector** `P = I - ĝĝᵀ` with `ĝ = ∇lnψ / |∇lnψ|`.
4. **Second fundamental form** `II_full = -(P·H·P)/|∇lnψ|`, where `H` is `slater.hessian`.
   The overall sign does not depend on the sign of ψ, so it is well defined on the surface.
5. **Tangent basis** by SVD of `P`, dropping the last right-singular vector (the null
   direction). `II = Eᵀ·II_full·E` is `(3N-1)×(3N-1)`.
6. **Principal curvatures** `kappas = eigvalsh(II)`.
7. **Co-area Jacobian** `J = Π 1/(1 - κᵢ·t_n)` and surface measure `dS = J/(2ε·ψ²)`.
   The `1/ψ²` undoes the VMC sampling weight so the estimator is uniform on the tube.
8. Returns `[dS, H·dS, HdF·dS, K·dS, min|κ|, J, dist, iterations]` with `H = mean(κ)`,
   `K = Π κ`, `HdF = H·|∇ψ|`.

### `casino/pycasino.py` — `Casino.nodal_domains(position)`

Runs the observable over the walk, keeps rows with a positive last component, prints ratios of
column sums, returns `2`. Wired into the `vmc` runtype after `vmc_energy_accumulation`.

---

## Defects found (fix before trusting any number)

1. **The estimator mutates the VMC walk.** `vmc.observable` passes `position[i]` — a *view*
   into the stored walk (`vmc.py:178`) — and the Newton loop does `r_e -= step` in place. Every
   projected configuration is overwritten in the sample array. Harmless today only because the
   `vmc` runtype does not touch `position` afterwards; in `vmc_opt` it would silently corrupt
   the sample. **Copy `r_e` before projecting.**
2. **Gaussian curvature `K = Π κᵢ` is meaningless here.** There are `3N-1` curvatures — 11 for
   Be, 29 for Ne. A product of that many curvature values spans far more than double precision
   and carries no usable information. Mean curvature `H` is the robust invariant and is what the
   Mitas measures use. Drop `K` or replace it with a low-order symmetric function.
3. **`epsilon = 0.4` is hard-coded** while the docstring explains at length that it should be
   adaptive, `ε ≈ R_cut/3` with `R_cut = 1/max(κ)` on the concave side. `min_abs_kappa` is
   computed and never used. Either implement the adaptive rule or delete the paragraph.
4. **`print()` calls inside the njit kernel** — two of them, debug leftovers. They serialise
   execution and destroy the timing.
5. **The node measured is the Slater node without backflow.** `slater.value(n_vectors)` takes
   the raw electron–nucleus vectors. With backflow the node moves, so as written the branch
   cannot answer "does backflow improve the node" — the most interesting question available.
6. **SVD per sample point is the wrong cost.** `np.linalg.svd(P)` on a `3N×3N` matrix is
   `O(N³)` per point purely to get a basis orthogonal to one vector. A Householder reflection
   that maps `ĝ` onto `e₁` gives the same basis in `O(N²)`.

Untested but worth thinking about: the variance of `dS = J/(2ε·ψ²)`. The `1/ψ²` weight and the
`ψ²` sampling density cancel in expectation, but the finite-sample behaviour near the node,
where the density vanishes as `t_n²`, has not been examined. Report an error bar, not a mean.

---

## The observed failure: focal points, and why ε cannot be tuned out of it

**Observed when running the branch (2026-08):** most sampled points sit far from the node, the
projection distance is large, and a point can land *near the centre of a sphere tangent to the
nodal surface* — at which stage there is no well-defined place to project it to.

This is not a solver-quality problem and no amount of Newton polish will fix it.

**What it is.** "Near the centre of the tangent sphere" means `t_n ≈ 1/κ` — the centre of the
osculating sphere, i.e. a **focal point**. The set of such points is the focal set / medial axis
of the surface. There the nearest-point projection is not inaccurate, it is *undefined*: a
continuum of surface points is equidistant. The code shows it directly — `J = Π 1/(1 - κᵢ·t_n)`
blows up and flips sign exactly at `t_n = 1/κᵢ`. That is what `R_cut` in the docstring is about,
but `epsilon = 0.4` is a fixed absolute length while `R_cut` varies point to point, so wherever
the node is strongly curved the tube already extends past the focal set.

**Why there are so few near-node points, and why shrinking ε makes it worse.** Sampling from
`Ψ²` gives a density that vanishes as `t_n²` at the node. The mass inside a tube of thickness ε
goes as `∫₀^ε t² dt ∝ ε³`. Halving ε costs a factor of 8 in statistics. The method is trying to
measure a surface with a measure purpose-built to avoid it; tuning ε cannot repair that.

**The structural fix: smear the delta in the VALUE of Ψ, not in distance.** The co-area formula
turns the surface integral into a volume integral with no projection at all:

```
∫_S Φ_B |∇Ψ| dS  =  ∫ Φ_B |∇Ψ|² δ(Ψ(R)) dR
```

With `δ_ε` narrow in `Ψ` rather than in `t_n`, the geometry is handled exactly by the co-area
factor. No normal map is built, so there are no focal points, no `J`, and no projection to
condition. Sampling from the nodeless `Φ_B` then gives well-behaved estimators on both sides:

```
numerator   ≈ ⟨ |∇Ψ|² · δ_ε(Ψ) ⟩_{Φ_B}
denominator ≈ ⟨ |Ψ| ⟩_{Φ_B}
```

neither of which degenerates at the node.

**Where projection is still unavoidable** — curvature, i.e. the diagnostics and the backflow
question — the only correct response to a focal point is to *detect and discard*, not to iterate
harder:

1. **Filter on `J`, which is already computed.** If any factor `(1 - κᵢ·t_n) ≤ 0`, the point is
   past the focal set: discard. If `|J - 1| > 0.3`, the tube is no longer a diffeomorphism to
   the required accuracy: discard. No extra theory needed.
2. **Adaptive ε from the *signed* curvature.** Only the `κᵢ` with `κᵢ·t_n > 0` are dangerous.
   Accept when `maxᵢ(κᵢ·t_n) < 1/3`. Note this needs `max`, whereas the code computes
   `min_abs_kappa`.
3. **Verify the projection landed nearby.** Compare `‖r_projected − r_original‖` against `t_n`;
   a large mismatch means Newton walked onto a different sheet of the node. Cap the iterations
   at ~10 — the present limit of 40 is itself a symptom, a well-conditioned projection converges
   in 3–5.
4. **The pre-filter is built on the same broken quantity.** `if dist < epsilon` uses
   `dist = 1/|∇lnψ|`, a *first-order* estimate of the distance to the node. Near focal points,
   and wherever `∇ψ` is small, it is unreliable — so the test deciding whether to attempt the
   projection is itself untrustworthy there.

---

## Analytically known nodes (test cases)

From the working notes in `nodal_surface.txt`, which reproduce the Mitas 2006 analysis. These
are the systems where the node is known in closed form, so curvature can be checked against an
exact answer rather than against another numerical result.

Same radial part for all orbitals — the radial factor cancels out of the determinant:

| state | nodal condition | geometry |
|---|---|---|
| s | none | no node |
| p(z) | `ez·r₁ = 0` | plane through the origin |
| p²(x,y) | `V(r₁, r₂, ez) = 0` | `r₁`, `r₂`, `ez` coplanar |
| p³ | `V(r₁, r₂, r₃) = 0` | **three electrons in a plane through the origin** |
| 2s2p | `ez·r₂₁ = 0` | plane |
| 2s2p² | `V(r₂₁, r₃₁, ez) = 0` | |
| 2s2p³ | `V(r₂₁, r₃₁, r₄₁) = 0` | **four electrons coplanar** |
| d² | factorises as `V(ex,ey,r₁)·V(ex,ey,r₂)·V(r₁,r₂,ez)` | product of planes |

Different radial parts:

| state | nodal condition | geometry |
|---|---|---|
| 1s2s | `(r₁-r₂)·(r₁+r₂) = 0` | **hypercone** |
| `(s - c·p)²` | `1 - c·r₁₂·r₃₄ = 0` | |

The `p³` plane and the `1s2s` hypercone are the two best first tests: a plane has all `κᵢ = 0`,
so `II` must come out numerically zero and `J = 1` exactly; a cone has one known nonzero
principal curvature and the rest zero. If `second_fundamental_form` does not reproduce those,
the projector, the tangent basis or the sign of `II` is wrong.

Two things in the original `nodal_surface.txt` were deliberately **not** carried over. The
determinant factorisations for `d³`, `d⁴`, `d⁵` and `f⁷` are unfinished there (marked `=?`) and
lead nowhere useful: higher-`l` noninteracting states add nothing as test cases once the plane
and the cone are checked. And the count of connected sign regions on a hypersphere
(`N = l+1` for `m=0`, else `2|m|(l-|m|+1)`) is superseded by the two-nodal-cell result above.

Further exact nodes are available in the Loos–Bressanini quasi-exactly-solvable models —
`Nodes.pdf` (JCP 142, 214112 (2015), *Nodal surfaces and interdimensional degeneracies*) and
`Nodal_surfaces.pdf`.

**Caveat worth keeping in view:** Krüger & Zaanen, *Fermionic quantum criticality and the fractal
nodal surface*, PRB 78, 035104 (2008) — `fractal.pdf`. In some regimes the nodal surface is not
smooth, and a curvature-based description has nothing to describe. This does not apply to small
atoms but bounds the generality of the whole approach.

---

## Work plan

Revised 2026-08 after the branch was run and hit the focal-point failure. The ordering matters:
the original plan put the curvature machinery first, which is backwards — the result worth having
does not need it, and the curvature path is the one blocked by an ill-posed projection.

### Track A — the result worth having. No projection, no curvature.

**A1. Estimate the surface integral by co-area in `Ψ`.** Implement Eq. (20) directly, with
`δ_ε` narrow in the *value* of `Ψ`:

```
numerator   ≈ ⟨ |∇Ψ|² · δ_ε(Ψ) ⟩_p / (weight of p)
denominator ≈ ⟨ |Ψ| ⟩_p
```

No `second_fundamental_form`, no Newton projection, no tangent basis, no `J`. Everything needed —
`Ψ` and `∇Ψ` — the code already computes. This is a small function, not a port.

**A2. Sample from something nodeless.** `Ψ²` is the wrong measure for this and is the root of the
statistics problem (mass within ε goes as ε³). Use `Φ_B`, or `Φ_B·|Ψ|` which vanishes only
linearly. For an atom a crude `Φ_B` — a product of one-particle densities, cf. Eq. (22) — is
enough to start; it need not be the exact bosonic ground state to test the machinery, only to get
the final number right. Decide and document which density the walk uses; this is a new sampling
mode, not the existing VMC walk.

**A3. Convergence in ε.** The estimator is biased at finite ε and noisy at small ε. Plot the
ratio against ε and show a plateau. Without this plot the number means nothing.

**A4. Reproduce Mitas & Annaberdiyev on Be.** `2109.01734.pdf` uses Be, noninteracting and fully
interacting; PyCasino has Be examples. Check the eigenvalue identity of Eq. (20). **This is the
first result that would be worth showing anyone**, and it is reachable without touching any of
the geometry code.

### Track B — curvature. Diagnostics and the backflow question.

Only worth starting once Track A works, or if the backflow question becomes urgent.

**B1. Make the existing kernel trustworthy.** Defects 1–4 and 6: copy `r_e` before projecting;
drop or replace `K`; remove the `print()`s; Householder tangent basis instead of SVD. Report an
error bar with every average. Rename `nodal_domains()` — it does not compute nodal domains.

**B2. Reject points where the projection is ill-posed.** The four filters in the focal-point
section above: sign check on every `(1 - κᵢ·t_n)`; `|J - 1| < 0.3`; adaptive acceptance
`maxᵢ(κᵢ·t_n) < 1/3` using *signed* max, not `min_abs_kappa`; verify the projected point landed
within a sane distance and converged in ≤ 10 iterations. Report the **acceptance fraction** — if
it is tiny, the tube approach is not viable for that system and that is itself the finding.

**B3. Validate against the analytic nodes.** `p³` (plane: every `κᵢ = 0`, `J = 1` at any `t_n`)
and `1s2s` (hypercone: one known nonzero principal curvature, the rest zero). If these do not
come out, the projector, the tangent basis or the sign of `II` is wrong and every production
number so far was noise. Put it in `casino/tests/` next to the numerical-derivative tests.

**B4. The node with backflow.** Lift defect 5: evaluate the node of the full antisymmetric part,
i.e. `slater` at the backflow-displaced coordinates. Then "does backflow flatten the node, and
does that track the DMC energy" becomes answerable, against Rasch–Hu–Mitas (`1310.2311.pdf`).

### Track C — optional, only if A produces something

Variational node improvement: the weighted formulations at the end of `2109.01734.pdf`, and
Lüchow's direct nodal-hypersurface optimisation (`10.1063@1.2716640.pdf`). A research project,
not a feature.

**Priority:** A1–A4. B is a separate, harder project with a real chance of being infeasible for
the systems we care about — the acceptance fraction in B2 decides that, and it is cheap to
measure. Do not treat B1 as progress; it is repair work on a path that may not be taken.

---

## Literature index — `pdfs/nodal_surface/`

**Core / two-cell property**
- `ceperley1991.pdf` — Ceperley, *Fermion Nodes*, J. Stat. Phys. 63, 1237 (1991). The tiling criterion.
- `mitas2006.pdf` / `0601485.pdf` — Mitas, *Structure of Fermion Nodes and Nodal Cells*, PRL 96, 240402 (2006).
- `0605550.pdf` — Mitas, *Fermion nodes and nodal cells of noninteracting and interacting fermions*.
- `Be_nodes.pdf` — Bressanini, *Implications of the two nodal domains conjecture*, PRB 86, 115120 (2012).

**The live direction**
- `2109.01734.pdf` — **Mitas & Annaberdiyev, *Weighted nodal domain averages*** (2021). The roadmap.
- `1310.2311.pdf` — Rasch, Hu, Mitas, *Fixed-node errors: electron density and node nonlinearities*.
- `1502.07248.pdf` — Rasch & Mitas, *Fixed-Node DMC of Lithium Systems*.
- `10.1063@1.2716640.pdf` — Lüchow et al., *Direct optimization of nodal hypersurfaces*, JCP 126, 144110 (2007).

**Exact / model nodes**
- `Nodes.pdf` — Loos & Bressanini, *Nodal surfaces and interdimensional degeneracies*, JCP 142, 214112 (2015).
- `Nodal_surfaces.pdf` — Loos, Gill, Bressanini, *Nodal Surfaces in Quasi-Exactly Solvable Models*.
  `o17.pdf` is a byte-identical duplicate of this file.
- `0409406.pdf` — Bajdich, Mitas et al., *Approximate and exact nodes: coordinate transformations and topologies*.

**Structure / surveys / adjacent**
- `what_we_knows_about_nodes.pdf` — Bressanini, Ceperley, Reynolds. The survey to read first.
- `JChemPhys_123_204109.pdf` — Bressanini, *Investigation of nodal structures and construction of trial wave functions*, JCP 123, 204109 (2005).
- `Atoms_nodal.pdf` — Bressanini, *Nodal structure of single-particle approximation atomic wave functions*, JCP 129, 054103 (2008).
- `glauser-jcp92.pdf` — Glauser et al., random-walk mapping of nodal regions, HF ground states Li–C.
- `etd.pdf` — Bajdich thesis, *Generalized Pairing Wave Functions and Nodal Properties*. Overlaps the `geminal` and pfaffian work.
- `lubos_mitas_tti_2006.pdf` — Bajdich et al. talk, *fermion nodes and pfaffian pairing wavefunctions*.
- `Schmidt.pdf` — Schmidt, *Pairing wave functions for QMC*.
- `fractal.pdf` — Krüger & Zaanen, *fractal nodal surface* (2008). The limit of the smooth picture.
- `s41467-023-37609-3.pdf` — Ren et al., *Towards the ground state of molecules via DMC on neural networks*, Nat. Commun. (2023).
- `cereply172.pdf` — Holzmann, Ceperley et al., *Backflow correlations for the electron gas and metallic hydrogen*, PRE 68, 046707 (2003).
- `ceperley175.pdf`, `ceperley1997.pdf` — Ceperley lecture notes / QMC for the electron gas.
- `kulahlioglu2014.pdf`, `chuvylkin2012.pdf`, `chuvylkin2013.pdf`, `smolenskii1996.pdf`, `1307.5567.pdf`, `варченко1990.pdf` — peripheral.
