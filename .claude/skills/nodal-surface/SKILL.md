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

**Status (2026-09-05):** branch `nodal_domains`, merged up to `main`, not merged into it.
- **Track A1 done.** `casino/nodal.py` — `nodal_domain_sums()`, the co-area estimator;
  `wfn.nodal_surface_integrand()` — `log|Ψ|`, `|∇Ψ/Ψ|²`, `V` of one configuration;
  `Casino.nodal_domain_accumulation()` — the `|Ψ|` walk, MPI reduction and the table over ε.
  Validated against the analytic 2p node in `casino/tests/test_nodal.py`.
- **The delta is smeared in `σ = Ψ/|∇Ψ|`, a length in bohr, not in the value of Ψ.** The
  value-smeared form of A1 was written first and does not work for an atom — see the correction
  in the co-area section below.
- **Track A2 done**: `vmc.power` selects the sampled power of `|Ψ|`, `pycasino --nodal` runs the
  averages after a `vmc` run.
- **A3–A4 done. A5 done, and its answer is negative — but read A6 below before acting on it: a
  `Ψ`-independent weight overturns it.** The
  estimator reproduces the paper's *analytic* Table 1 on the four-electron noninteracting Be,
  standalone and through the production path; its Table 2 on the interacting Be is not reproducible
  and the evidence says the fault is not ours. Neither `E^nda - ⟨H⟩` nor `E_kin^nda` is
  Jastrow-invariant, so neither is a measure of the node; weighting by `Φ = 1/J`, i.e. running with
  `use_jastrow F`, removes the Jastrow exactly and leaves a quantity that is still not a measure of
  the node, because it tracks the *amplitude* of `D` instead. Stage 1 over Be, N and Ne says so at
  8.7 σ — see the stage-1 section, which is the thing to read before spending another hour here.
- **A6 built and measured 2026-09-06, and it reverses the A5 verdict on eight of nine points.**
  Both A5 failures came from a `Φ` built out of the `Ψ` being measured. The fix is the paper's own
  §4.1, `Φ = Π η(r_i)` with `η = e^{-ζr}` and `V₀` obtained by inversion — Eq. (18), everything
  analytic, `Φ` fixed once per Hamiltonian. On the `c₂` scan at ζ ≥ 0.25 the descriptor's minimum is
  at C = 0.15, next to the true 0.164, the rise at C = 0.20 is there at 4.7 σ, and `r` with `ΔE_FN`
  goes from -0.09 at ζ = 0 to +0.98. **A chain that walks `Φ|Ψ|` itself (`vmc.zeta`, `-w`) then put
  all nine points on one measure at ζ = 1**, with no exclusion left: minimum at C = 0.15, the rise
  at C = 0.20 at 7.5σ, `r = +0.836` over nine and +0.955 over the eight without C = 0.20, which is
  now the one point out of place. **It ranks and it does not measure** — 129 mHa of descriptor per
  mHa of fixed-node error, and `E^nda` itself scatters over an au around the true energy.
  **The exact bosonic ground state is still not needed and is still a trap**: approximate it
  and Eq. (20)'s collapse into a constant is gone. A6 in the work plan carries the numbers and what
  comes next.
- **`casino/nodal_descriptor.py <run dir> ...`** is the tool to reach for: it measures the
  descriptor of any system, forcing `use_jastrow F` and choosing the ε grid from the wave
  function's own median σ (`median/100` to `median/8`), and prints a comparison table over several
  runs, one block per ζ. Executable with a shebang, like `casino/plot/plot.py`, tracked since
  2026-09-06; `examples/casino.sh` carries the invocation. Beside this file are two narrower scripts:
  `small_eps.py`, the bare measurement on a hand-chosen grid, and `noninteracting_be.py`, the
  analytic Table 1 check.
- **The live direction is counting the nodal domains of the trial function**, which this file used
  to forbid on a misreading — see the section right below, which also carries the Be ladder
  HF 10.2 → backflow 3.7 → geminal 0.22 mHa that makes the case. Nothing has been built yet.
- **The `c₂` scan on Be says topology is necessary and not sufficient** (2026-09-05, section after
  the ladder). Moving one 4-determinant family through the topology change: no plateau over the
  topologically-correct region — the bottom is quadratic, `p = 2.23 ± 0.09`, a flat bottom excluded
  at 20 σ — and **no step at `c₂ = 0`**, the HF point sits on the extrapolated curve to 1.0 σ. The
  optimum reaches `-14.667379(22)`, an exact node. Qualifies the ladder's "nothing but the domain
  count explains a jump like that".
- **Track B code deleted** (2026-09-03), `wfn.second_fundamental_form` and
  `Casino.nodal_domains` with it. Recover from `git show 9d0622c5:casino/wfn.py` if B is ever
  taken up; the defect list below is what to fix in it first. `nodal_domains` is at
  `git show 2c84cad1:casino/pycasino.py` — right target, wrong instrument, rewrite as a
  connectivity test rather than restore.
- Working notes: `nodal_surface.txt` at repo root (analytic nodes; folded into this skill).
- Literature: `pdfs/nodal_surface/` (index at the end).

---

## Counting nodal domains — the entry that used to say "don't", and why it was wrong

**This section previously read "do not build a domain counter, the answer is always two". That was
an error, it killed the branch's original direction, and it is the single most expensive mistake in
this file's history. The two-domain result is about the EXACT ground state. A trial function has as
many domains as its functional form allows, and a single determinant has four.** Counting the
domains *of the trial function* is the measurement that actually discriminates.

The literature, read correctly:

- Ceperley, *Fermion Nodes*, J. Stat. Phys. 63, 1237 (1991) — `ceperley1991.pdf`. The tiling
  theorem: all nodal domains of a ground state are equivalent under permutations. It does **not**
  fix their number.
- Mitas, PRL 96, 240402 (2006) — `mitas2006.pdf` / `0601485.pdf`, and `0605550.pdf`. Two domains
  for spin-**polarised** systems, where the determinant does not factorize.
- Bressanini, *Implications of the two nodal domains conjecture*, PRB 86, 115120 (2012) —
  `Be_nodes.pdf`. **The one to read.** For a spin-unpolarised system the spatial function factors
  into α and β determinants, `Ψ_HF = |1s2s|_α · |1s2s|_β`, whose node is
  `(r₁−r₂)(r₃−r₄) = 0` — two independent hypersurfaces crossing at π/2 in a `3N−2` set where both
  `Ψ` and `∇Ψ` vanish, cutting space into **four** domains: two positive, `(+,+)` and `(−,−)`, and
  two negative. Same-sign domains are not adjacent; they meet only on that degenerate crossing.
  Adding `1s²2p²` with any `c ≠ 0` turns the crossing into an *avoided* crossing — a channel opens,
  the two same-sign domains merge, and the count drops to two, the topology of the exact function
  (proved in the paper's appendix by a π rotation about `R₁ × R₃`).

**Why the count is worth so much energy.** `E_FN` is the lowest Dirichlet eigenvalue in one domain.
Opening the channel does not rearrange the surface, it *adds a passage* between two previously
disconnected same-sign pieces, so the new domain contains the old one, and by domain monotonicity
`λ₁` can only fall. (This is the mechanism, not a theorem about a specific HF/CI pair — real CI also
deforms the rest of the surface. And do not reach for Faber–Krahn or "half the space versus a
quarter": these domains are unbounded, they have no volume, and set inclusion is the whole argument.)

**Measured on Be, all three at QZ4P, `dtdmc = 0.02083`, 200000 stats steps, same code:**

| node | run | `E_FN` | FN error |
|---|---|---|---|
| HF | `stowfn/Be/HF/QZ4P/CBCS/Jastrow_dmc` | -14.657179(57) | 10.18 mHa |
| backflow | `stowfn/Be/HF/QZ4P/CBCS/Backflow_dmc` | -14.663645(42) | 3.71 mHa |
| geminal | `geminal/Be/HF/QZ4P/EBES/Jastrow_dmc` | **-14.667135(20)** | **0.22 mHa** |

exact non-relativistic Be = -14.667356. **This is Bressanini's ladder measured in our own examples.**
Backflow is a continuous coordinate transformation, `Ψ_BF = |X(R)|_α |X(R)|_β` is still a product of
two determinants, so §III C of the paper says it can *never* change the topology however flexible it
is — and it stalls at 3.7 mHa. The geminal is not such a product, clears the barrier, and drops 17×
to a node that is exact to 0.2 mHa. Nothing but the domain count explains a jump like that at
otherwise identical settings.

**So: build the counter.** The original `Casino.nodal_domains` (`2c84cad1`, removed in `caa4a45a`)
had the right target and the wrong instrument — it clustered same-sign VMC samples with HDBSCAN,
but density clustering looks for gaps in density, and two same-sign domains are separated by a
codimension-2 set that the sample approaches continuously from both sides. The question is
**connectivity, not density**: can a path from `R` to `P₁₂P₃₄R` be found along which `Ψ` never
changes sign? `P₁₂P₃₄` is even, so both endpoints have the same sign and the question is well posed.
Implementation is a walk with sign-rejection — reject any move that flips `sign(Ψ)` — started at `R`
and asked whether it reaches a neighbourhood of `P₁₂P₃₄R`. For a four-electron singlet there is an
even cheaper form: under HF the walk's invariants are `sign(r₁−r₂)` and `sign(r₃−r₄)`, which can
never flip with four domains and do flip through the channel with two, so **count the flips**.

---

## The c₂ scan on Be: topology is necessary and not sufficient — measured 2026-09-05

The ladder above compares three *different* wave functions. The scan below moves the node of **one**
family continuously through the topology change, which is the sharper experiment, and it answers two
questions the ladder cannot.

`/mnt/sdb1/quantum_chemistry/!PROJECT/ORCA/MP2-CASSCF(2.4)/ano-pVDZ/Be/VMC_DMC_BF/emin/<C>/tmax_1_1024_1`,
Fortran CASINO, `vmc_dmc`, `dtdmc = 0.020833`, 50000 stats steps, 1024 configs. `C = |c₂|`, the
coefficient of the `1s²2p²` CSF in a 4-determinant MDET (`c₁ = 0.95006`, the three `2p_x,y,z` dets
tied to a common `c₂`); `C = 0.00` has **no MDET block at all**, i.e. the bare HF node. Jastrow is
`gjastrow` (`parameters.casl`), backflow is the **Φ term only** — no η, no μ — truncation order 3,
`N_eN = 5`, `N_ee = 2`, cutoff 8 a.u., 122–124 free parameters. **Every point is an independent emin
optimization**: all nine `correlation.data` backflow blocks and all nine `parameters.casl` have
distinct md5. Results in `Be_4det.dat`, columns `C  E_DMC  σ  N_corr  σ`.

| C | 0.00 | 0.01 | 0.02 | 0.05 | 0.10 | 0.12 | **0.15** | 0.20 | 0.25 |
|---|---|---|---|---|---|---|---|---|---|
| ΔE_FN, mHa | 2.379(54) | 2.068(52) | 1.781(44) | 1.087(38) | 0.235(29) | 0.077(23) | **−0.022(22)** | 0.065(25) | 0.537(36) |

against exact non-relativistic -14.6673564. At the optimum the node is exact to the error bar:
**-14.667379(22)**, better than the geminal's 0.22 mHa in the ladder above.

**1. No plateau. The bottom is quadratic.** The hypothesis under test was that backflow can realize
the ideal node starting from *any* node with the right topology, so that every `C ≠ 0` point should
be equal within error bars. Instead the spread over `C ∈ [0.01, 0.25]` is **2.09 mHa** against error
bars of 20–50 µHa. Fitting `a|C−x₀|^p + b` with `p` free, weighted by σ:

| range | `p` | `x₀` | `E_min` | χ²/ndf |
|---|---|---|---|---|
| all 9 | **2.23 ± 0.09** | 0.1635(13) | -14.667394 | 0.63 (5) |
| `C > 0`, topology correct | **2.24 ± 0.10** | 0.1635(13) | -14.667392 | 0.75 (4) |
| `C ≥ 0.05` | 2.50 ± 0.21 | 0.1632(13) | -14.667376 | 0.39 (2) |

`p = 4` — a flat bottom, which is what a plateau would look like and what the `a(x−x₀)⁴+b` in the
directory's `energy.sh` encodes — is excluded at **20 σ**. So the geometry of the starting node
survives the backflow optimization: backflow lowers the whole curve but does not wash out the `c₂`
dependence. Topology is necessary and not sufficient.

**2. The topology change leaves no step in `E_FN`.** A parabola fitted to the topologically-correct
region alone (`C ∈ [0.01, 0.25]`, `a = 0.0887(21)` Ha, `x₀ = 0.1641`) extrapolates to
`E(0) = -14.665042(36)`; measured `-14.664977(54)`. The step is `+65 ± 65 µHa`, **1.0 σ**. The HF
point sits on the same smooth curve as the rest.

This does not contradict the domain argument, it locates it. The channel opens with *zero width* at
`c = 0⁺`, so domain monotonicity gives a continuous fall of `λ₁`, not a jump — there is no reason to
expect a discontinuity at `c = 0` and there is none. What the topology buys is not a step at the
origin but the existence of the family at all: no member of the four-domain one-parameter set
(`C = 0` is its only member) gets below 2.38 mHa, while the two-domain set reaches 0.02 mHa.
**Amend the ladder's claim above accordingly** — "nothing but the domain count explains a jump like
that" is right about HF vs geminal at fixed everything else, but the jump is not observable as a
discontinuity in `c₂`, and 2.36 of the 2.38 mHa the scan recovers is smooth deformation *after* the
topology is already correct.

**Caveat on the strength of the backflow.** Φ-only, so this bounds *this* backflow, not backflow in
principle. It is not a weak one — on the HF node it gives 2.38 mHa where the QZ4P `Backflow_dmc`
example gives 3.71 — but adding η/μ or the Ω term could in principle flatten the curve. Closing the
1.09 mHa still open at `C = 0.05`, where the topology is already correct and 122 backflow parameters
have already been optimized by emin, would take a qualitatively different flexibility, not a larger
expansion.

**Trap in the directory's own analysis.** `energy.sh` and `n_corr.sh` fit with gnuplot *without
weights* ("residuals are weighted equally"). Adding the three new small-`C` points, whose deviations
are ~2 mHa, then swamped the sum of squares and threw the quartic's `x₀` from 0.160 to 0.214 ± 0.044.
Weighted, the same quartic gives `x₀ = 0.1608(19)`, unchanged. Use `using 1:($2-E):3 yerrors` in any
further fit there. `N_corr` (column 4) tracks the energy — minimal at the minimum, parabola `x₀ ≈
0.13–0.14` — but with ~10% error bars and a 2 σ offset it is a weaker locator than the energy.

---

## The other live direction: weighted nodal domain averages

Integrals over the nodal hypersurface. Tried in full (track A) and it **failed** — see the campaign
below; the section is kept because the estimator is built and validated and the failure is
instructive, not because a number is expected from it.

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
- **`Φ⁰_B` belongs to the Hamiltonian, not to `Ψ`.** Easy to skim past, and A5 stage 1 was lost on
  it: the exact result holds because the weight is *the same function* for every `Ψ` of that
  Hamiltonian, which is what makes two eigenstates comparable through it. Every weight we can
  actually afford — `Φ = 1`, `Φ = 1/J` — is built from the wave function being measured and so
  changes with it, and then the ratio stops being a property of the node. Any future attempt has
  to start from a `Ψ`-independent weight — which is the next section, not stage 2 of the campaign
  below (stage 2 is a *different* and disqualified idea, the ρ-weighted node area).

### §4.1 of the paper: the one-particle product weight — the affordable `Ψ`-independent `Φ`

**The exact bosonic ground state is not needed, and chasing it is the wrong move.** §4.1 is skimmed
past easily and it is the practical entry point; the section above used to leave the impression that
`Φ⁰_B` was the only `Ψ`-independent option.

Take the weight to be a product of one-particle functions, Eq. (16), where `η` solves a one-particle
Schrödinger equation with *some* effective potential `V₀`, Eq. (17):

```
Φ(R) = Π_i η(r_i) ,   [-½∇² + V₀(r) - e₀] η(r) = 0
```

Then `T_kin Φ = (e_Φ - V_Φ)Φ` with `e_Φ = N e₀` and `V_Φ = Σ_i V₀(r_i)` **analytically**, so the
third term of Eq. (15) is in closed form and Eq. (18) reads

```
E = ∫_∂Ω Φ|∇Ψ| dS / ∫ Φ|Ψ| dR  +  e_Φ  +  ⟨V - V_Φ⟩_{Φ|Ψ|}
```

**The trick the paper does not spell out: Eq. (17) never has to be solved.** Choose any nodeless `η`
and *invert* for the potential, `V₀ = e₀ + (½∇²η)/η`; Eq. (18) then holds identically. For a Slater
`η = e^{-ζr}` with `e₀ = 0`:

```
V₀(r) = ζ²/2 - ζ/r ,   V_Φ = Nζ²/2 - ζ Σ_i 1/r_i ,   e_Φ = 0
V - V_Φ = -(Z-ζ) Σ_i 1/r_i + Σ_{i<j} 1/r_ij - Nζ²/2
```

everything analytic, `Φ` fixed once per Hamiltonian, no bosonic DMC anywhere. `ζ` is the only knob
and it is a **variance** knob, not a bias one: Eq. (18) is exact for any `ζ`, and the good `ζ` is the
one that makes `V - V_Φ` closest to constant. That is what the paper's closing remark of §4.1 —
"tempting to think about the possibilities that would lead to vanishing last term by an appropriate
tuning of `V₀`" — is about.

**It reproduces their own toy model, §4.3.** One electron, `Ψ = 2p`, `ζ = Z`: then `V - V₀ = -Z²/2`
identically and `E^nda = 3Z²/8 - Z²/2 = -Z²/8`, their Eq. (23). At that `ζ` the weight `e^{-Zr}` *is*
the exact bosonic ground state, so Eq. (18) degenerates into Eq. (20) — which is the cleanest way to
see that §4.1 is the affordable version of §4.2. `test_nodal.py` already gets the `3Z²/8` surface
term; **`-Z²/8` is the unit test to add**, and it catches every sign and normalization error in the
weight path at once.

**Why `Φ⁰_B` itself is a trap, not a goal.** It is nodeless, so DMC is sign-problem-free for it —
but DMC yields an energy and a mixed distribution, not a pointwise function, and a Metropolis weight
needs `Φ_B(R)` pointwise. Getting that means optimizing an explicit bosonic trial function
(Eq. (22)'s envelope `e^{J_B}·Π ρ(r_i)`, i.e. our Slater-Jastrow with the determinant replaced by a
product of one orbital) — and the moment `Φ_B` is approximate, the collapse of both volume terms
into the constant `E_0B` is lost and the full Eq. (15) comes back, with `T_kin Φ_B` now needing the
trial function's own local kinetic energy. **The entire advantage of Eq. (20) evaporates exactly
when it becomes computable.** Note also that `E_0B` is a pure additive constant, so for ranking
nodes on one Hamiltonian its *value* is never needed — only the *function* is.

**What a fixed `Φ` does and does not fix — do not repeat the A5 mistake in reverse.** It restores a
common measure, so two trial functions become comparable. It does **not** make the number a property
of the node alone: the overlap `∫Φ|Ψ|dR` depends on the whole shape of `Ψ`, and Eq. (20) is an
identity only for an *eigenstate*. For a trial function Eqs. (24)–(25) say outright that the
estimator is non-variational and deviates to first order in either direction. So the A5 diagnosis
("the descriptor tracks the amplitude of `D`") is not automatically cured — it merely stops being
confounded by a `Ψ`-dependent measure.

**Where the identity *is* exact: Eq. (29), written for `Ψ = Ψ_FN`.** The fixed-node solution is an
eigenstate of `H_FN` in its domain, so with the exact `Φ_B` the surface integral gives `E_FN` with no
approximation at all — the node functional the whole programme wants. It is not reachable from a DMC
run: the importance-sampled density `f = Ψ_T Ψ_FN` vanishes at the node, which is precisely where
`|∇Ψ_FN|` is wanted. **Record it, do not build it.**

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

### `casino/nodal.py` — `nodal_domain_sums(integrand, epsilon, log_weight=None)`

The Track-A estimator. Takes the per-configuration `[log|Ψ|, |∇Ψ/Ψ|², V]` produced by
`wfn.nodal_surface_integrand`, returns the sums of the surface integral (one row per ε: sum,
sum of squares, configurations inside the tube) and of the overlap (count, weight, `V`, `V²`).

- The delta is smeared in `σ = Ψ/|∇Ψ|` — the first-order distance to the node, **a length in
  bohr**. `σ` is read straight off the integrand as `1/√(|∇Ψ/Ψ|²)`; nothing is projected, no
  tangent basis and no `J`, so the focal-point problem of Track B does not arise.
- Kernel `δ_ε(σ) = |σ|/ε²` for `|σ| < ε`. Its factor `|σ|` cancels the `|∇Ψ|/|Ψ|` that the change
  of variable brings, so with configurations distributed as `Φ|Ψ|` the whole surface integral is
  **the occupancy of the tube divided by ε²** — bounded, and Poisson in the count inside.
- `|∇σ| = 1` holds on the node and is not evaluated off it, which leaves an `O(ε)` bias on top of
  the kernel's `O(ε²)`. Both show up in the ε scan as the departure from the plateau.
- There is no arbitrary scale left: ε is a physical length, and no `MPI.MAX` normalization of Ψ
  is needed.
- `log_weight` is `log(Φ/p)` up to a constant. The default assumes the sample is already
  distributed as `Φ|Ψ|`, which makes every overlap weight exactly 1.
- Only a **constant Φ** gives `E_nda = E_kin + E_pot` outright; a general Φ has the third term
  `⟨|Ψ|T_kin Φ⟩/⟨Φ|Ψ|⟩` of Eq. (15), which is not implemented. For the one-particle product weight
  of A6 that term is closed-form, `e_Φ - ⟨V_Φ⟩_{Φ|Ψ|}`, so it costs one more accumulated sum, not a
  new estimator.

**Measured behaviour** (`casino/tests/test_nodal.py`, hydrogenic 2p, `Φ = exp(-Zr)`, points drawn
from `Φ|Ψ|` directly): at 10⁷ points a flat plateau at the exact `3Z²/8 = 1.500` over
`ε ∈ [0.004, 0.05]` bohr. The number of configurations inside the tube falls off as **ε²** —
better than the `ε³` of a `Ψ²`-sampled distance tube, but still the thing that sets the smallest
usable ε for a given sample (10⁶ points run out below `ε ≈ 0.01`).

### Track B, deleted — `wfn_second_fundamental_form(self, r_e)`

Removed 2026-09-03; `git show 9d0622c5:casino/wfn.py` has it. What it did, per sample point:

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

### `casino/vmc.py` — `vmc.power`

The power of `|Ψ|` the Metropolis walk is distributed as: `2` for VMC, `1` for the nodal
averages. A `float64` field of `VMC_class_t` with a property setter, used in the three acceptance
exponents (`simple_random_step`, `one_electron_step`, `log_ratio_walk`). Everything else about
the walk — CBCS/EBES, step optimization, the decorrelation period — is unchanged.

### `casino/pycasino.py` — `Casino.nodal_domain_accumulation(epsilon=None)`

Run by `pycasino --nodal` after `vmc_energy_accumulation` in the `vmc` runtype. Sets
`vmc.power = 1`, re-equilibrates and re-optimizes the step (the `|Ψ|` distribution is much more
diffuse: `⟨r⟩` on Be goes 1.53 → 2.63), walks in chunks, feeds each chunk to `nodal_domain_sums`,
restores `power = 2`, reduces across processes and logs `E_kin^nda ± sem` and `E^nda` for every ε
plus the ε-independent potential term. Only the sums survive a chunk, so memory is flat in
`vmc_nstep`. Default grid `geomspace(0.005, 0.32, 12)` bohr.

The deleted `Casino.nodal_domains(position)` was the Track-B entry point: it ran the observable,
printed ratios of column sums and returned `2`, from the `vmc` runtype after
`vmc_energy_accumulation` — where `position` is the DMC configuration tail and so is **empty**
unless `vmc_nconfig_write > 0`. Do not re-wire A2 through that return value.

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

**Correction (2026-09-04): value-smearing was implemented, run, and does not work for an atom.**
`|Ψ|` is small in the whole exponential tail of configuration space and not only at the node, so
the tube `|Ψ| < ε` fills with configurations that are merely far away. Measured on Be: the median
`V` of the configurations inside rose from `-9.9` to `-5.8` as ε shrank — the tube was migrating
outward, not inward — while their median `σ` stayed at `≈0.198`, and `E_kin` went as `√ε` with no
plateau over five decades of ε. Choosing the reference scale better does not help (the largest
`|Ψ|` of the sample is a rare peak: `-3.119` against a median `-11.759`); the defect is that
`|Ψ|` is not a distance.

**What works is smearing in `σ = Ψ/|∇Ψ|`**, the same co-area identity written with the
first-order distance instead:

```
∫_S Φ |∇Ψ| dS  =  ∫ Φ |∇Ψ| δ_ε(σ) |∇σ| dR ,   |∇σ| = 1 on the node
```

`σ` stays of order one out in the tail, `|Ψ|` and `|∇Ψ|` vanishing together, so the tube holds
the node alone. ε is then a length in bohr and needs no normalization of Ψ at all. This is what
`casino/nodal.py` implements; it reproduces `3Z²/8` exactly on the 2p toy.

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

**A1. Estimate the surface integral by co-area. — DONE 2026-09-03, corrected 2026-09-04**, see
the code map above. No projection, no tangent basis, no `J`; `casino/nodal.py` plus one njit
observable. The delta is smeared in `σ = Ψ/|∇Ψ|`, not in the value of `Ψ` — the value-smeared
version was written first and fails on an atom, see the correction above.

**A2. Sample from something nodeless. — DONE 2026-09-03.** `vmc.power` is the power of `|Ψ|` the
walk is distributed as: 2 for VMC, 1 for the nodal averages. Why the walk had to change at all,
and why no reweighting of the VMC one will do: **any** weight `Φ` that is nonzero at the node
makes the importance weight of a `Ψ²` walk go as `1/|Ψ|`, whose overlap estimator
`⟨Ψ⁻²⟩_{Ψ²} = ∫dR` has infinite variance; a `Φ` that vanishes at the node instead kills the
surface term outright (`Φ = |Ψ|` is the paper's own remark — the node integral drops out and the
ordinary expectation comes back).

`Φ = 1` is what is implemented, so the sampling density is `|Ψ|` and `E_pot^nda = ⟨V⟩_{|Ψ|}` is a
plain sample mean. Table 2 of the paper is `Φ = 1` on Be, which is the row to compare against. A
product-of-densities `Φ_B` (Eq. 22) is the refinement after that, and needs the third term of
Eq. (15), which `nodal_domain_sums` does not have.

**A3. Convergence in ε. — in progress.** `Casino.nodal_domain_accumulation` prints the whole ε
table with error bars. Expect the count inside the tube to fall as ε²: the plateau is bounded
from below by statistics and from above by the `O(ε)` + `O(ε²)` bias, and if the two windows do
not overlap the sample is too small.

**Measured on Be** (`stowfn Be HF QZ4P`, Slater only, CBCS, `vmc_nstep = 10⁷`, `--nodal`,
2026-09-04): a plateau `E_kin^nda = 0.325–0.35` over `ε ∈ [0.005, 0.048]` bohr, flattest at
`0.3249(79)` at `ε = 0.023`; `E_pot^nda = -14.3042(32)`, so `E^nda ≈ -13.98`. The tube count runs
85 → 7896 across that window, i.e. the ε² law, and the run costs about as much as the VMC that
precedes it. Above `ε ≈ 0.2` the table explodes (`E_kin = 20.8`): the tube has swallowed the
whole sample, because ε has passed the median `σ ≈ 0.195` — the cusp scale again, not the node.

**A small σ does mean "near the node" — checked, 2026-09-04.** The worry was that `σ = 1/|∇lnΨ|`
could be small merely because an electron sits on a nucleus, where the cusp makes the gradient
large. It cannot: the cusp gives `|∇lnΨ| = Z`, *bounded*, so near a nucleus σ tends to `≈1/Z`, a
floor rather than a route to zero. On Be the whole σ distribution is a narrow spike — 5% quantile
`0.181`, median `0.195`, 95% `0.201` — set by the typical total log-gradient of four electrons,
and the near-node region is a tail of order 0.02% of the sample.

Measured directly (`scratchpad/tube_vs_nucleus.py`, 4·10⁶ configurations, Be), the smallest
electron-nucleus distance of the configurations *inside* the tube against the whole sample:

| | inside | 5% | 25% | median | 75% | 95% |
|---|---|---|---|---|---|---|
| sample | 4000000 | 0.170 | 0.347 | 0.523 | 0.744 | 1.143 |
| ε = 0.023 | 670 | 0.233 | 0.467 | 0.720 | 1.007 | 1.482 |
| ε = 0.048 | 3101 | 0.224 | 0.456 | 0.692 | 0.967 | 1.448 |
| ε = 0.103 | 18139 | 0.218 | 0.447 | 0.670 | 0.929 | 1.385 |

Every quantile moves *away* from the nuclei, monotonically as the tube narrows: the node of Be
lies where the electrons are spread out, not where one of them is on the nucleus. **Do not filter
on the nucleus distance** — there is nothing to remove, and cutting a region out of the node would
no longer be the paper's integral over the whole hypersurface.

What the cusp scale does explain is the top of the ε table: once ε enters the σ spike, the tube
swallows everything at once (`3995163` of `4·10⁶` at ε = 0.219, with the r-quantiles back to the
sample's own to three digits), which is the `E_kin = 20.8` row.

Mike Towler's `VMC_NODE_PROBE` (CASINO `vmc.f90`, 2026-08-30) writes the same pairing per
configuration — `1/|v|` for the whole configuration and per particle, the sign of Ψ, the distance
and direction to the nearest bare nucleus — so it is the way to repeat this check inside CASINO.
It cannot give the nda energy itself, because it samples `Ψ²`, where the tube occupancy goes
as ε³.

**A4. Reproduce Mitas & Annaberdiyev on Be. — Table 1 reproduced exactly 2026-09-04; Table 2
still not, and the evidence now says the fault is not ours.**

**Table 1 is the test that matters, because it is analytic.** For the *noninteracting* Be —
hydrogenic 1s and 2s at Z = 4, `H = Σ(-½∇² - Z/r)` — the wave function is an exact eigenstate, so
the third term of Eq. (15) vanishes and Eq. (6) must hold identically. The paper gives closed
forms for `1S(1s²2s²)`: `E_kin^nda = 320/221 = 1.447964`, `E_pot^nda = -4740/221 = -21.447964`,
summing to exactly -20. Two independent runs reproduce them:

| | `E_pot^nda` | `E_kin^nda` |
|---|---|---|
| exact, Table 1 | -21.447964 | 1.447964 |
| standalone, 10⁷ (`noninteracting_be.py`, next to this file) | -21.4567(38) | 1.49–1.52 over ε ∈ [0.004, 0.025] |
| production path, 10⁷ (hand-built `stowfn.data`) | — | 1.42–1.49 over the same range |

The standalone script is its own Metropolis walk with analytic gradients and no PyCasino code at
all; the production run goes through the stowfn reader, `slater.gradient`, the `|Ψ|` walk,
`nodal_surface_integrand` and `nodal_domain_sums`. **So the estimator is right on a
twelve-dimensional determinant node**, not only on the one-electron plane of `test_nodal.py`. The
residue above 1.448 is the known `O(ε)` bias — the two tightest tubes give 1.30 and 1.33.

Building the hydrogenic `stowfn.data` is worth knowing: three s shells, `ζ = 4` of radial order 0
for the 1s, and `ζ = 2` of orders 0 and 1 for the 2s, whose coefficients `1/N` and `-2/N` rebuild
`(1 - Zr/2)e^{-Zr/2}` after `Stowfn.normalize_orbitals` applies its `N`. `E_pot^nda` from the
production path is *not* comparable — PyCasino always adds the e-e repulsion the noninteracting
problem does not have (the difference, 3.82, is `⟨Σ1/r_ij⟩` over `|Ψ|`) — but `E_kin^nda` involves
no potential at all and can be compared directly. That trick makes any exact-eigenstate node a
test of the production estimator.

**Table 2 remains irreproducible, and the scaling favours our numbers.** Everything on our side is
now verified: the sampler against a He quadrature, the estimator against the analytic 2p and
against Table 1 in twelve dimensions, `drift_velocity` confirmed to be a plain `∇lnΨ` with no
Umrigar-style velocity cutoff hiding in it. A physical argument now points the same way. Between
the hydrogenic Be and the real HF Be, the 2s *node* barely moves — `r = 0.500` to `0.590` bohr —
while the orbital's extent grows by 1.77, `⟨r⟩ = 1.500` to `2.651`. `E_kin^nda` is a surface
integral over a nearly fixed node divided by a volume integral that grows with the diffuseness, so
it must fall steeply. Our `1.448 → 0.322` is a factor `1.77^2.55`, close to the dimensional `²` an
energy demands; their `1.448 → 1.126` is `1.77^0.44`, which would need the interacting Be 2s to be
nearly unscreened.

The origin of the Table 2 numbers is ref. [17], Hu, Rasch & Mitas, *Many-Body Nodal Hypersurface
and Domain Averages for Correlated Wave Functions*, ACS Symp. Ser. 1094 (2012) 77–87 — **not in
`pdfs/nodal_surface/`**. Getting it is the remaining move, along with asking the authors.

**Open discrepancy.** Table 2 of the paper (HF Be, `Φ = 1`, no Jastrow) is
`E_kin^nda = 1.126(6)`, `E_pot^nda = -15.248(6)`, `E^nda = -14.1214(4)`. Our numbers are
`E_kin^nda ≈ 0.30` and `E_pot^nda ≈ -14.32`. The sampler is not the cause — `⟨V⟩` at `power = 2`
gives `-29.18` (CBCS) and `-29.22` (EBES) against the virial `2E = -29.13`, and at `power = 1`
the two independent algorithms agree with each other (`-14.317` / `-14.335`). The estimator is
not the cause either — it is exact on the 2p toy. The trial function is not the cause: their
`⟨H⟩ = -14.5731(2)` is our `-14.573160(534)` to every digit they print, and Eqs. (12)–(13) with
`Φ = 1` are literally what the code computes, `E_pot^nda = ⟨V⟩_{|Ψ|}` over the whole space.

Both of our numbers are off in the *same* direction, which is one cause, not two: our sampled
`|Ψ|` is more diffuse than theirs (a diffuse density both softens `⟨V⟩` and down-weights the node,
where the density is large). `⟨V⟩_{|Ψ|}/⟨V⟩_{Ψ²}` is `0.491` for us and `0.523` for them, against
the one-electron hydrogenic value of exactly `0.5`.

**The `power = 1` walk is now validated against an exact number, and it is right.** A closed-shell
`1s²` determinant factorizes, `|Ψ| = |φ(r₁)||φ(r₂)|`, so `⟨V⟩_{|Ψ|} = 2⟨-Z/r⟩_{|φ|} + ⟨1/r₁₂⟩` is
a radial quadrature over the stowfn orbital (the angular average of `1/r₁₂` over two isotropic
densities is `1/r_>`). On `stowfn He HF QZ4P`:

| | `⟨V⟩_{\|Ψ\|}` |
|---|---|
| quadrature | `-2.618639` |
| `--nodal` walk, 10⁷ steps | `-2.619358(692)` |

one standard error apart, `3·10⁻⁴` relative. So `E_pot^nda = -14.304(3)` on Be is our answer and
the `-15.248(6)` of Table 2 is not a sampling artefact of ours. What remains to explain is on
their side or in the wave function they used, `⟨H⟩` agreeing notwithstanding.

**The clue from the null test:** the disagreement is an almost constant *additive* offset that
survives a change of trial function, `+0.80` on `E_kin^nda` and `-0.94` on `E_pot^nda` without a
Jastrow, `+0.82` and `-0.84` with one (against their 2-conf Slater-Jastrow row), and the two
nearly cancel in the total (`E^nda` differs by 0.14 and 0.02). A constant offset in the components
that cancels in the sum is the signature of a definition or normalization difference — note that
Eq. (10) is derived by integrating over `Ω+` alone while Eqs. (12)–(13) do not say what the
volume integrals run over — not of statistics or of a different wave function.

**Watch out for spurious basis-set nodes.** The He HF orbital of that file changes sign at
`r = 6.973` bohr — a tail artefact of the STO expansion. Under `Ψ²` it is invisible; under `|Ψ|`
it carries **0.98% of the norm**, and it is a genuine node, so the He run reports
`E_kin^nda ≈ 0.005` from it (plateau over `ε ∈ [0.02, 0.05]`, 26–148 configurations in the tube).
A "nodeless" closed shell is only nodeless up to the basis.

**A5. Is an nda number a measure of the node? Neither candidate scalar survived the null test.**
The obvious use of all this is to print an nda number alongside every VMC energy and rank nodes
by it: HF against CASSCF against backflow, over several systems. Table 2 of the paper argues
against the raw scalar, from the paper's own numbers. Write `Δ = E^nda - ⟨H⟩`, the bracket of Eq. (31), which the paper expects to be positive:

| WF | `E^nda` | `⟨H⟩` | `Δ` |
|---|---|---|---|
| HF, no Jastrow | -14.1214(4) | -14.5731(2) | 0.4517(5) |
| 2-conf, no Jastrow | -14.1399(5) | -14.6129(3) | 0.4730(6) |
| 2-conf + Jastrow (VMC) | -14.6290(2) | -14.6627(1) | 0.0337(2) |
| the same, DMC | -14.6409(4) | -14.6670(1) | 0.0261(4) |

Row 2 → row 3 adds a Jastrow to the *same* determinant part. **The node does not move at all** — a
Jastrow is positive — and Δ falls by a factor of 14. Row 1 → row 2 improves the node and leaves
the amplitude alone, and Δ *rises* by 0.021 with an error bar of 0.001. So Δ tracks the amplitude,
not the node. It has to: `E_kin^nda` depends on `|∇Ψ|` on the node *and* on the normalization over
all space, and only for an exact eigenstate does the node fix everything. The components are
better behaved — `E_kin^nda` climbs `1.126 → 1.141 → 1.191 → 1.221` and `E_pot^nda` falls
`-15.248 → -15.281 → -15.820 → -15.862` as the function improves — but they too move more with the
Jastrow than with the node.

The campaign is therefore a test of that hypothesis, cheapest first, not a survey:

0. **The Jastrow null test — RUN 2026-09-04, and Δ fails it.** `Be Slater` against
   `Be Slater-Jastrow`, byte-identical `stowfn.data`, so the determinant and the node are the same
   function; 10⁷ steps each, `vmc_decorr_period = 10` in both.

   | | Slater | Slater-Jastrow |
   |---|---|---|
   | `⟨H⟩` | -14.573160(534) | -14.650061(75) |
   | `E_pot^nda` | -14.3042(32) | -14.9809(33) |
   | `E_kin^nda` (ε = 0.023) | 0.3249(79) | 0.3721(85) |
   | `E^nda` | -13.9794(85) | -14.6088(91) |
   | `Δ` | **0.5938** | **0.0413** |

   Δ collapses by a factor of 14.4 with the node held fixed, reproducing the paper's own
   `0.473 → 0.034`. **`Δ = E^nda - ⟨H⟩` is not a measure of the node.** Do not print it as one.

   The failure is not uniform across the two components: of the 0.72 that `E^nda` moves,
   `E_pot^nda` carries 0.68 and the surface term `E_kin^nda` only 0.047. That made `E_kin^nda` look
   like a candidate nodal measure, so it was tested properly — small ε only, 10⁸ steps, `mpirun
   -n 4`, VMC accumulation skipped: `small_eps.py` next to this file calls
   `nodal_domain_accumulation(np.geomspace(0.002, 0.024, 9))` directly, which is also how to run
   any further measurement without paying for a VMC energy nobody asked for.

   | ε (bohr) | Slater, inside | `E_kin^nda` | Jastrow, inside | `E_kin^nda` | ratio |
   |---|---|---|---|---|---|
   | 2.729e-03 | 239 | 0.3210(208) | 295 | 0.3962(231) | 1.234 |
   | 3.722e-03 | 440 | 0.3175(151) | 549 | 0.3962(169) | 1.248 |
   | 5.078e-03 | 832 | 0.3226(112) | 1019 | 0.3951(124) | 1.225 |
   | 6.928e-03 | 1542 | 0.3213(82) | 1857 | 0.3869(90) | 1.204 |
   | 9.452e-03 | 2918 | 0.3266(60) | 3508 | 0.3927(66) | 1.202 |
   | 1.289e-02 | 5335 | 0.3209(44) | 6434 | 0.3869(48) | 1.206 |
   | 1.759e-02 | 9956 | 0.3217(32) | 12059 | 0.3897(35) | 1.211 |
   | 2.400e-02 | 18677 | 0.3243(24) | 22316 | 0.3874(26) | 1.195 |

   Both curves are **flat** at these ε — there is no `O(ε)` bias to extrapolate away below 0.024,
   and the apparent rise seen in the 10⁷ tables sets in only above ≈0.03. The limits read off
   directly, `0.322(2)` against `0.388(3)`: `E_kin^nda` **is not Jastrow-invariant either**, by
   0.066 ± 0.004, seventeen standard errors, with the ratio flat across eight tubes and no trend
   towards 1. `E_pot^nda` is `-14.308377(1015)` against `-14.984182(1045)`.

   **But it fails exactly as the identity says it must, and that hands over the fix.** With
   `Ψ = J·D` and `D = 0` on the node, `∇Ψ = J∇D`, so

   ```
   E_kin^nda[JD] / E_kin^nda[D] = ⟨J⟩_node / ⟨J⟩_volume = 1.205 ± 0.010
   ```

   the first average over the node with weight `|∇D|dS`, the second over space with `|D|dR`. The
   measured 1.205 is a physical statement: the Jastrow is 21% larger on the node than in the bulk,
   because the node sits where the electrons are spread out (see the `r_min` table above) and a
   Jastrow suppresses the configurations where they are close.

   **The fix: weight by `Φ = 1/J`.** Then the numerator is `∫|∇D|dS` and the denominator `∫|D|dR`,
   so the quantity belongs to the determinant part alone and is Jastrow-invariant by construction.
   No reweighting code is needed to get it: `Φ|Ψ| = |D|` means it is simply **the same wave
   function run with `use_jastrow F`**. Our `0.322(2)` already is that clean descriptor, and the
   `0.388(3)` is the same thing spoilt by the Jastrow.

1. **Vary the node, `use_jastrow F` throughout. — DONE 2026-09-04 over Be, N and Ne, and the
   descriptor does not rank nodes.** HF and backflow determinants — the backflow parameters
   optimized with a Jastrow, the measurement taken without one — against the fixed-node DMC energies
   already in `examples/*/Backflow_dmc`. The question was never "does nda differ" but "does it rank
   nodes the way `E_FN` does, more cheaply". It does not.

   `stowfn HF QZ4P CBCS`, `mpirun -n 4`, ε grid from `nodal_descriptor.py`; the row is the widest
   tube, and every table was checked across all nine ε:

   | | `ΔE_FN` | steps | `E_kin^nda` HF | `E_kin^nda` backflow | change |
   |---|---|---|---|---|---|
   | Be | 6.47 mHa | 10⁷ | 0.3305(75) | 0.3220(73) | −2.6%, 0.8 σ |
   | N | 7.40 mHa | 10⁸ | 3.0223(135) | 3.1906(137) | **+5.6%, 8.7 σ** |
   | Ne | 9.47 mHa | 10⁷ | 10.156(117) | 7.751(96) | −23.7%, 16 σ |

   Three nodes improved by nearly the same amount in DMC — 6.5, 7.4, 9.5 mHa — and the descriptor
   moves by 0, +6% and −24%. No correlation, and **N has the wrong sign at 8.7 σ**: its backflow
   node is 7.4 mHa better in DMC and 6% *higher* in the descriptor. That is not a fluctuation and
   not an `O(ε)` artefact — the ratio backflow/HF sits at 1.06 ± 0.01 across all nine tubes of the
   10⁸ run, from ε = 1.03e-3 to 1.29e-2.

   Be's earlier 3.5 σ has to be withdrawn. The 10⁷ table drifts from 0.276 at ε = 6.9e-3 to 0.331
   at ε = 2.4e-2 — 20% of `O(ε)` bias, on the grid `nodal_descriptor.py` picks from the median σ,
   where the old hand-chosen `geomspace(0.002, 0.024, 9)` had looked flat. A 3.5% difference read
   off the widest tube is smaller than the drift beneath it, so it was never a plateau statement.

   **Diagnosis — the descriptor measures how diffuse `D` is, not where its node is.** The potential
   component comes free from the same walk and gives it away:

   | | `E_pot^nda` HF | `E_pot^nda` backflow | difference | `E_kin^nda` change |
   |---|---|---|---|---|
   | Be | -14.3150 | -14.0352 | 0.28 | 2.6% |
   | N | -54.9064 | -54.1899 | 0.72 | 5.6% |
   | Ne | -130.4354 | -118.5577 | **11.88** | 24% |

   The two runs sample wildly different densities, and the size of that difference tracks the size
   of the descriptor shift. The formula says it must: with `Φ = 1/J` the estimator is exactly
   `P(σ < ε)/ε²` under the measure `|D|dR`, i.e. **the fraction of the density that sits within ε of
   the node**, divided by ε². A determinant stripped of the Jastrow it was optimized with is a much
   more diffuse object — backflow spreads the electrons and nothing pulls them back — so less of its
   density lies near the node and the descriptor falls for a reason that has nothing to do with the
   node's shape. Ne's `|D|` is diffuse enough to move `⟨V⟩` by 11.9 au, hence −24%; N's geometry
   happens to win over its 0.72 au and pushes the other way.

   `E^nda` fails the same way and more bluntly: HF beats backflow in **all three** systems
   (-13.98 vs -13.71, -51.88 vs -51.00, -120.28 vs -110.81), three times opposite the truth,
   because the HF `D` is the more compact one.

   **So the choice of `Φ` is a vice, not a knob.** `Φ = 1` drags in the Jastrow (21% on Be, six
   times the putative node signal). `Φ = 1/J` removes the Jastrow identically and leaves the
   determinant's amplitude, which is worse. **No `Φ` built from the wave function's own `Ψ` puts two
   different wave functions on a common measure**, and without a common measure the ratio
   `∫Φ|∇Ψ|dS / ∫Φ|Ψ|dR` cannot be a property of the node alone.

   Costs, 4 processes, HF + backflow: Be 10⁷ 137 + 407 s; N 10⁷ 218 + 858 s; N 10⁸ 2619 + 11453 s;
   Ne 10⁷ 296 + 1508 s. Logs kept as `examples/{Be,N,Ne}_*.log`, untracked.

2. **The common-measure repair — worked out, then abandoned on Bressanini. Do not build it.**
   The idea was to walk once under a single fixed density and ask both wave functions for their σ,
   so the amplitude cancels. The algebra is worth keeping because it explains the whole family:
   `σ = |Ψ|/|∇Ψ| ≈ d`, the distance to the node, with `|∇Ψ|` cancelling — so `{σ < ε}` is a purely
   geometric tube of half-width ε. Under the measure `|Ψ|dR` the density itself vanishes linearly
   inside the tube, which is where both the `ε²` and the `|∇Ψ|` of `E_kin^nda` come from. Under a
   *fixed nodeless* density ρ the occupancy is instead linear in ε and

   ```
   P_ρ(σ_Ψ < ε) / 2ε  →  ∫_node ρ dS / ∫ ρ dR
   ```

   the ρ-weighted **area** of the node, free of Ψ's amplitude by construction, and with occupancy of
   order percent rather than the 0.018% of `E_kin^nda` — two to three orders of magnitude cheaper.

   **It is still the wrong functional, and Be says so exactly.** Area is an integral of a local
   quantity over the surface; the avoided crossing is local surgery near a `3N−2` set, so it moves
   the area by a little and `λ₁` by 10 mHa. The sign is wrong too: the reconnection replaces a
   crossing by two branches bending away from it, which are *longer* than the straight sheets they
   asymptote to — so the area most likely **rises** exactly where the node becomes correct. And the
   three stage-1 pairs could never have shown this: per §III C backflow cannot change topology at
   all, so HF↔backflow is pure smooth deformation, the friendliest possible case for a metric
   descriptor, and it still failed on N. Two independent reasons, one empirical and one structural.

   Node quality is carried by **connectivity**, which no smooth surface integral sees. Go count
   domains instead — the section near the top of this file.

**A6. The one-particle product weight — built and measured 2026-09-06. It works.** The one repair
A5 left open, and the section "§4.1 of the paper" above is the theory. `Φ = Π η(r_i)` with
`η = e^{-ζr}`, `V₀` obtained by inversion rather than by solving anything, and Eq. (18) for the
volume part. Fixed per Hamiltonian, so all wave functions of one system share a measure — the
single thing stage 1 lacked. **Do not start from `Φ⁰_B`**: it is the same code with a bosonic
optimization campaign attached and, being approximate in practice, loses the property that
motivates it.

*Where the code is.* `wfn.nodal_surface_integrand` returns six components — `log|Ψ|`, `|∇lnΨ|²`,
`V`, `Σ r_iI`, `Σ 1/r_iI`, `Σ_i |Σ_I r̂_iI|²`. **ζ deliberately does not enter it**, so a whole grid
of ζ comes out of one walk; `nodal_domain_sums(integrand, epsilon, zeta)` builds
`Φ = exp(-ζ Σ r_iI)` and `V_Φ = ζ²/2 Σ_i|Σ_I r̂_iI|² - ζ Σ 1/r_iI` from those sums, and
`nodal_domain_accumulation(epsilon, zeta)` carries a ζ grid beside the ε grid.
`nodal_descriptor.py -z 0,0.125,0.25` drives it. The third sum is the one worth not forgetting: it
is the number of electrons for one nucleus and is not for several, where inverting `η` leaves a
`∇η·∇η` cross term.

*Reweighting is what makes the grid free, and it holds.* The walk stays `|Ψ|`, `Φ ≤ 1` bounds the
weights from above so nothing can dominate, and what it costs is the effective sample size, printed
per ζ. Measured on Be with four electrons: **100% at ζ = 0, 87% at 0.125, 70% at 0.25, 54% at
0.375, 42% at 0.5, and 0.6% at ζ = 2.** So ζ ≲ 0.5 is free and ζ ≳ 1 is not; sampling `Φ|Ψ|`
directly (`vmc.power`, or a walk targeting `Φ_{ζ*}|Ψ|` and reweighting in a window around ζ*) is
what buys the larger ζ, and it was not needed for the result below.

**Read the ESS as a spread and never as a distance.** It is (Σw)²/Σw² and it measures how much
`Σ r_iI` varies, not how large it is: a density that has moved out as a whole scores *high*. The
one wave function of the nine whose density had moved was the one with the highest ESS of all, and
it is the one reweighting cannot reach — see C = 0.25 below. When the ESS on one run stands out
from its neighbours in either direction, that is the signal, and only the potential split says
which way.

*Test 1 is done and came out stronger than planned.* `TestWeightedNodalDomainAverage` in
`casino/tests/test_nodal.py` does not check `-Z²/8` at `ζ = Z` alone; it checks the ζ-independence
of Eq. (18) on ζ = Z/4, Z/2, Z. Closed forms, derived and verified: the surface term is
`(ζ + Z/2)²/6`, the volume term `(ζ - Z)(ζ + Z/2)/3 - ζ²/2`, and their sum is `-Z²/8` identically.
The sample is the walk's own `|Ψ|` so that the estimator builds Φ and `V_Φ` itself. The two terms
are asserted separately and tightly (0.5% and 12%) and their sum loosely (30%), because the sum is
a cancellation of two numbers four times larger than the answer. Mutation-checked: wrong sign on
the weight exponent 300%, factor 2 on `ζ²` 419%, wrong sign on `ζ/r` 818%, `V_Φ` added instead of
subtracted 74%, `V_Φ` built from Z instead of ζ 227%.

*Test 3 is done — the Be `c₂` scan, nine points at 10⁷ steps, ζ = 0…0.5.* **The bar is met on eight
of the nine.** At every ζ ≥ 0.25 the minimum of `E_kin^nda` sits at C = 0.15, the sampled point
adjacent to the true 0.164, and the rise to C = 0.20 that the constant weight could not produce is
there: `+0.129 ± 0.028` (4.7σ) at ζ = 0.25, `+0.096 ± 0.039` at 0.375, `+0.116 ± 0.053` at 0.5.
Correlation with `ΔE_FN` over the eight: `r = -0.09` at ζ = 0, then `+0.66, +0.98, +0.99, +0.98`.
The volume term is what does it — C = 0.20 has `⟨V⟩ = -8.76` au with no weight and
`⟨V - V_Φ⟩ = -14.76` at ζ = 0.25, back in line with the healthy points.

Two honest limits on that. The scale is unchanged: the descriptor spans 0.4 au where `ΔE_FN` spans
2.4 mHa, so it ranks and does not measure; and it still overstates C = 0.20, ranking it fourth
where DMC ranks it second.

*The ninth point, C = 0.25, is not rescued, and the reason is overlap, not a second disease.*
`⟨V - V_Φ⟩` stays at -1.55…-1.98 au against -14.8…-15.7 for everything else. Splitting the
potential (2026-09-09) settles what is happening:

```
              e-e        e-n         V
C = 0.15    +2.534    -16.922    -14.388
C = 0.20    +2.498    -17.147    -14.649
C = 0.25    +1.338     -2.904     -1.566
```

The e-e term is *smaller* at C = 0.25, not larger. From `e-n = -Z⟨Σ 1/r_i⟩` with Z = 4 the typical
electron-nucleus distance is 0.93 bohr on a healthy point and **5.5 bohr** there, and from the
six pairs of the e-e term the typical e-e distance goes 2.4 → 4.5 bohr. The cloud has expanded
about sixfold, uniformly. Same disease as C = 0.20 — a backflow left without the Jastrow it was
optimized with — several times worse.

**The ESS argument that said otherwise was wrong, and the mistake is worth keeping.** ESS =
(Σw)²/Σw² with `w = exp(-ζ Σ r_iI)` is sensitive to the *spread* of `Σ r_iI`, not to its size. A
cloud that has moved out as a whole has a narrow spread and therefore a **high** ESS — 0.96 at
ζ = 0.125 against 0.855 for the healthy points. Read as "not diffuse", that is exactly backwards.

So the weight is not too weak; there is nothing in the sample to reweight. Every configuration
sits at `Σ r_iI ≈ 22`, all the weights are equally tiny, and the average barely moves — hence
-1.55 → -1.98 across the whole ζ grid. `Φ|Ψ|` lives where this point's `|Ψ|` has almost no mass:
an overlap failure, which no ζ fixes while the walk is over `|Ψ|`. The old numbers already showed
it starting — at ζ = 0.5 the ESS at C = 0.25 falls to 0.200 against 0.42 for the healthy points,
crossing over from the highest to the lowest. **Excluding C = 0.25 was not pre-registered — say
so**, though the grounds (13 au on the volume term) are independent of the outcome.

*Dead end, recorded so it is not retried.* The ζ-drift of `E^nda` — Eq. (18) is ζ-independent for
an eigenstate and not for a trial function, so the drift measures how far `Ψ` is from one. It
correlates at `r = 0.92` over the seven healthy points and collapses to 0.40 over eight. Worse than
the descriptor itself. Dropped.

*Direct sampling of `Φ|Ψ|`, built 2026-09-09, and the nine points measured on one measure at last.*
`vmc` carries a `zeta` field beside `power`; the acceptance is `power·Δlog|Ψ| + ΔlogΦ` at three
sites — `simple_random_step`, `one_electron_step` (through `proposal`, so the single-electron move
stays O(1)) and `log_ratio_walk` — and `wfn.log_weight` / `log_weight_1e` supply `logΦ`.
`nodal_domain_sums(integrand, epsilon, zeta, sampled)` separates the measure the average is taken
on from the one the chain walked, the weight being `exp(-(ζ-sampled)Σr)` and one when they agree;
**`V_Φ` goes by `zeta` alone in either case** — it belongs to the average, not to the route taken
to it, and getting that wrong is silent. `nodal_domain_accumulation(..., direct=True)` puts the ζ
loop outside the walk, with its own equilibration and dtvmc per ζ. `nodal_descriptor.py -w`.

*Validation.* On C = 0.15, where reweighting works, the two agree: `E_kin^nda` within 0.7σ at
ζ = 0.25 and 1.2σ at 0.7. The potentials differ by 0.10 au, nominally 2.9σ, which is the
`⟨V⟩` error bar being optimistic on a `1/r` estimator rather than a discrepancy. Every direct row
reports ESS equal to the full sample, which is the check that `sampled == zeta` took the right
branch.

*The threshold is sharp and it is ζ ≈ 1.* `⟨Σ r_iI⟩` of C = 0.25 goes 22.4 (ζ=0), 21.8 (0.25,
reweighted — nothing moves), 13.6 (0.7, direct), **5.48 (1.0, direct)**, against 5.62 for C = 0.15.
Below 1 the confining weight loses to the wave function's own spread; at 1 it wins.

*The nine points, direct, ζ = 1.0, 10⁷ steps.* **The bar is met on all nine, no exclusions.**

```
  C     dE_FN     <Σr>    <V-Vφ>    e-e     E_kin^nda        E^nda
 0.00   2.379    5.419   -15.885   4.278   1.9213±0.0190   -13.963
 0.01   2.068    5.465   -15.684   4.226   1.8645±0.0183   -13.819
 0.02   1.781    5.551   -15.477   4.143   1.8067±0.0176   -13.670
 0.05   1.087    5.569   -15.518   4.120   1.7660±0.0174   -13.752
 0.10   0.235    5.678   -15.548   4.048   1.6391±0.0166   -13.909
 0.12   0.077    5.669   -15.596   4.048   1.6631±0.0167   -13.933
 0.15  -0.022    5.626   -15.848   4.085   1.6287±0.0165   -14.219
 0.20   0.065    5.191   -16.405   4.325   1.8122±0.0182   -14.593
 0.25   0.537    5.442   -16.528   4.304   1.7443±0.0375   -14.783
```

`⟨Σr⟩` spans 9% where it spanned a factor of 2.3 at ζ = 0, and the volume term 1.05 au where it
spanned 13.5. Minimum at C = 0.15, the sampled point next to the true 0.164; the rise to C = 0.20
is **7.5σ**; `r` with `ΔE_FN` over all nine is **+0.836**.

*The residual failure has moved to C = 0.20.* The descriptor ranks it sixth of nine where DMC ranks
it second; without it `r` over the other eight is **+0.955**. C = 0.25, the point that used to need
a post-hoc exclusion, now ranks third against DMC's fourth. And the reason is the same one as
always, one order of magnitude down: C = 0.20 has the *smallest* `⟨Σr⟩` and the *largest* e-e of
the nine, so it is the one still furthest from the common density, and the descriptor charges it
for that.

*A defect found and closed by the same run.* The ε grid comes from a pilot walk, and the pilot ran
over `|Ψ|` while the measurement ran over `Φ|Ψ|`. On C = 0.25 the pilot's median σ came out 0.089
bohr against 0.184–0.195 elsewhere, so its widest tube was 2.2× narrower and held a fifth of the
configurations. Re-reading all nine at a common ε = 1.11e-2 changes nothing — `r = 0.839`, same
minimum, rise 3.8σ — so the conclusion is not resting on it, but the pilot now runs at the
sampling ζ.

*Unchanged, and it is the ceiling.* The descriptor spans 0.31 au over the nine where `ΔE_FN` spans
2.4 mHa: 129 mHa of descriptor per mHa of fixed-node error, the same ratio measured at ζ = 0.25 by
reweighting. And `E^nda` itself lands between -13.67 and -14.78 against a true `E_FN` of -14.667
(the jastrowless function shares the node, hence the energy), with a ζ-drift of 0.1-0.3 au within
one point. Eq. (18) is an identity for an eigenstate and this is the size of the deviation for a
trial function. **It ranks; it does not measure**, and no amount of sampling changes that.

### A6b. The optimization algorithm — specified 2026-09-09, estimator built, optimizer not

Everything above measures a node. This is how it is *improved*.

**The target is the backflow of a single determinant.** The question the whole programme is aimed
at: given one Slater determinant, can its node be improved by optimizing the backflow against the
nodal functional rather than against the energy? So `p` is the backflow parameters — `η, μ, Φ, Θ,
Ω` — and not the Jastrow, which is optimized separately by emin and needed only at stage 3. MDET
coefficients move the node too and the algorithm takes them, but the `c₂` scan was the calibration
polygon, not the goal.

**The test bed already exists and it is C = 0.00 of that scan.** At `c₂ = 0` there is no MDET block
at all: a bare HF node, one determinant, with backflow and Jastrow emin-optimized, and its
fixed-node energy is known — **2.379 ± 0.054 mHa** above the optimum of the scan. The experiment is
one line: re-optimize that backflow on `F`, run DMC, and see whether it goes below 2.379. Same
ansatz, same topology, same system; the only thing that differs is the functional the backflow was
optimized against. That is the question with nothing in between.

**The ceiling is known in advance and must be stated.** Backflow is a continuous transformation and
cannot change the topology of the node — Bressanini §III C, and Pablo's argument in the t=225
thread that a smooth `x_i = r_i + ξ_i` cannot turn a sphere inside out without `ξ_i` diverging.
Be's HF node has four domains, the exact one two. So the optimization runs **inside the
four-domain topology** and its limit is the best four-domain node. "Improved" here means recovering
part of the 2.379 mHa, not reaching zero.

**Stage 0.** emin the Jastrow at fixed `p`. Every evaluation of the objective then runs with
`use_jastrow F` / `use_gjastrow F`: `J` does not move the node, `Φ = 1/J` removes it exactly, and
"weight by `1/J`" and "do not switch `J` on" are the same operation.

**Stage 1, fix the constants once.**

- `ζ`: run 10⁶ steps of direct sampling at ζ ∈ {0.25, 0.5, 1, 1.4, 2} on two or three deliberately
  different `p`, and take **the smallest ζ at which `⟨Σ r_iI⟩` agrees between them to a few
  percent**. That is the condition that the measure stops following the density as `p` moves. On Be
  it is ζ = 1.0, and the threshold is sharp — 2.3× spread at ζ = 0, 9% at ζ = 1. Larger ζ is worse,
  not better: both terms grow while their sum does not, so the cancellation sharpens.
- `ε`: one value from the plateau of the ε scan at that ζ, **frozen for the whole optimization**.
  Re-deriving it from a pilot at each `p` redefines `F` as `p` moves; the C = 0.25 mismatch, a
  factor 2.2 in ε, cost a factor 5 in tube occupancy.

**Stage 2, minimize `F(p) = E_kin^nda`.** Not `E^nda`: with the volume term added the function
falls monotonically to the edge of the c₂ range and has no interior minimum at all, because the
volume term is amplitude.

*The kernel must change first.* As built, `δ_ε(σ) = |σ|/ε²` cancels against `|∇Ψ|/|Ψ|` and leaves a
plain **count** of configurations — bounded, Poisson, and discontinuous in `p`, so no gradient
exists. Replace it by

```
δ_ε(σ) = 3|σ|(ε - |σ|)/ε³        ⟹  after the cancellation:  3(ε - |σ|)/ε³
```

which normalizes, keeps the cancellation, stays bounded (`3/ε²` at σ = 0) and **vanishes at the
tube edge**, so `F` becomes continuous and differentiable in `p`. One line in `nodal_domain_sums`.
Then

```
dF/dp = ⟨ δ'_ε(σ)·∂σ/∂p ⟩ + cov[ δ_ε(σ), ∂ln|Ψ|/∂p ]
```

the first term through `σ = 1/|∇lnΨ|` from `gradient_parameters_d1`, the second the ordinary score
term from `value_parameters_d1`; both exist. `Φ` carries no `p` and drops out of the score.
Derivative-free (Powell) would do for a handful of MDET coefficients, but backflow has tens to
hundreds of parameters, so **for the actual target the gradient is not optional and the kernel
change is the first thing to build** — without it there is no algorithm, only an estimator.

*A failure mode specific to optimizing the backflow, which did not exist for `c_k`.* `F` is
measured on the jastrowless object while the optimizer is free to deform the backflow. What that
produces has already been measured on this very scan: C = 0.20 and C = 0.25 blew up precisely
because their backflow was optimized *with* a Jastrow and evaluated *without* one, the density
expanding by a factor of 2.3 in the worst case. There it happened by accident; here the optimizer
will look for such deformations on purpose, because they lower `F`. The fixed weight `Φ` is what
holds it back and mostly does — but C = 0.20, the point with the smallest `⟨Σ r_iI⟩` of the nine,
is still the one the descriptor misplaces.

So carry `⟨Σ r_iI⟩` as a **constraint and not as a diagnostic**: it is already printed, and a step
that moves it by more than a few percent is rejected. Keep the backflow cutoffs fixed and optimize
the linear parameters only.

Sample size from the measured conversion, **129 mHa of `F` per mHa of `E_FN`**: 10⁷ steps give
0.017 au on `F`, i.e. 0.13 mHa of fixed-node energy.

**Stage 3, a DMC line search, and it is not optional.** Take `p(t) = p₀ + t(p* - p₀)`, three or
four `t` near 1, each with its Jastrow re-optimized and DMC on a `dt·N_w = const` ladder. On the
test bed the answer is read straight off it: below 2.379 mHa and the backflow of a single
determinant has been improved by the nodal functional; at or above it, and emin was already doing
as well as this can. The
reason is structural, not statistical: `⟨H⟩` is a Rayleigh quotient and errs at second order, while
Eq. (18) is an identity only at `Ψ_FN` and errs at **first** order in either direction
(Eqs. 24-25), so its stationary point is displaced. Measured: the minimum of `F` lies between
c₂ = 0.10 and 0.15 against the true 0.1639 ± 0.0013, worth 0.01-0.27 mHa out of 2.4. What the first
two stages buy is that a many-parameter node search becomes one-dimensional.

**Why not just emin.** `⟨H⟩ = ∫Ψ²E_L/∫Ψ²` weights the node quadratically to zero — the set that
defines the node contributes almost nothing to the objective — and `⟨H⟩` and `E_FN` are different
functionals, so a better variational energy can come with a worse node. `F` integrates over the
node and over nothing else. Sampling `|Ψ|` rather than `Ψ²` is what makes the tube visible at all:
`|Ψ| ~ σ` against `Ψ² ~ σ²`, so its share of the sample goes as ε² instead of ε³.

*Not yet measured, and it is the missing calibration:* `E_VMC(c₂)` with each point's own Jastrow
and backflow is literally the curve emin would descend if c₂ were free, so the distance from its
minimum to 0.164 is how far energy minimization misses the node optimum, in mHa. The archive
cannot give it — 1024-step VMC phases, 4-7 mHa against a 2.4 mHa curve. Nine VMC runs at 10⁷.

*Next, in order.*

1. ~~Split `wfn.coulomb` into e-e and e-n.~~ **Done 2026-09-09.** `wfn.coulomb_parts` returns the
   pair, `coulomb` is their sum plus the nuclear repulsion, the integrand carries both. The
   `e-e interaction` and `e-n interaction` lines of the *VMC* block are still empty — that needs
   `vmc_energy_accumulation` to carry a multi-component observable, which this did not do.
2. ~~A weight with an e-e factor.~~ **Cancelled**: the e-e term at C = 0.25 is smaller than on the
   healthy points, so there is no e-e pathology to cure.
3. ~~Direct sampling of `Φ|Ψ|`.~~ **Done, and it delivered the result above.**
4. **C = 0.20.** The one point out of place, and the only lead left on the residual density
   sensitivity. Its `⟨Σr⟩` is 5.19 against 5.42-5.68; a larger ζ tightens every density but not
   obviously the spread (at ζ = 1.4 the three-point subset sat at 4.85/4.56/4.48, at 2.0 at
   4.03/3.86/3.77). Worth one nine-point run at ζ = 1.4 to see whether its rank moves.
5. **The Jastrow null test, with the question corrected.** Still not run. With a fixed `Φ` the
   quantity is an *estimate of the energy*, not a node invariant, so demanding Jastrow-invariance
   (the A5 test) is the wrong demand. What must hold: `E^nda` approaches `⟨H⟩` as `Ψ` improves.
   The nine-point table above is the first evidence on it and it is not encouraging — `E^nda`
   scatters over an au around the right answer.
6. **The true `Φ_B`** — still only if everything above holds, and knowing what the section on it
   says.

**On gaussians.** `gwfn` runs work and the cusp correction is supported (`cusp_correction`
defaults to T for a gaussian basis, and `nodal_surface_integrand` goes through the same
`log_value`/`drift_velocity`). The nuclear region is harmless either way — with the correction
`|∇lnΨ| → Z` so `σ → 1/Z`, without it a gaussian has zero radial derivative at the origin and
`σ → ∞`; both are far outside the tube. The **tail** is the problem: the denominator `∫|D|dR`
weights the far tail as `|D|` rather than `D²`, and that is exactly where a gaussian basis is
qualitatively wrong, decaying as `exp(-αr²)` instead of `exp(-Z_eff r)`, and where contractions
with negative coefficients put spurious nodes — `stowfn He` already has one at `r = 6.973` bohr
carrying 0.98% of the `|Ψ|` norm. So compare **within** one basis, which is all stage 1 needs;
absolute values across `gwfn` and `stowfn` carry that systematic. A free calibration if wanted:
the same `gwfn` run with `cusp_correction` T and F differ by a small, controlled node shift inside
`r_c`.

Practical constraints for any such run: no code change is needed for backflow or a multi-
determinant, the estimator takes the full `∇Ψ` through `drift_velocity` (Track B's defect 5 does
not apply here); all-electron only; ε below ≈0.024 needs no extrapolation, but check flatness per
wave function, because a Jastrow adds e-e cusps to `|∇lnΨ|` and moves the σ scale; and the cost at
10⁸ steps is 23 min for a Slater Be and 45 min with a Jastrow on 4 processes, the tube holding
0.02% of the sample.

### Track B — curvature. Diagnostics and the backflow question.

Only worth starting once Track A works, or if the backflow question becomes urgent.

**B1. Make the kernel trustworthy.** It is deleted, so this now starts by restoring it from
`git show 9d0622c5:casino/wfn.py`. Then defects 1–4 and 6: copy `r_e` before projecting; drop or
replace `K`; remove the `print()`s; Householder tangent basis instead of SVD. Report an error bar
with every average.

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

### Track C — variational node improvement. A5 says do not start it at all in this form.

The scheme is Eq. (31) of `2109.01734.pdf`: minimize `E_VMC,nda = ⟨H⟩ + λΔ` with
`λ = αω/(1+αω)` and `Δ = E^nda - ⟨H⟩ > 0`; Lüchow's direct nodal-hypersurface optimisation
(`10.1063@1.2716640.pdf`) is the other entry point. A research project, not a feature. What A5
measured says four things about it in advance:

- **The penalty steers the Jastrow, not the node.** Δ is 0.594 without a Jastrow and 0.041 with
  one at a fixed node, so `∂Δ/∂(Jastrow)` dwarfs `∂Δ/∂(determinant)`. The optimizer will spend
  itself pulling the Jastrow away from its energy minimum to shrink a term that has nothing to do
  with the node — worse VMC energy, node unmoved.
- **λ is caught in a vice.** At λ = 0.1 the penalty is 59 mHa without a Jastrow and 4.1 mHa with
  one, against a fixed-node error of order 2 mHa for Be. A λ small enough not to distort the
  Jastrow is too small to move the node; a λ large enough to move the node wrecks the Jastrow.
- **So the penalty must act only on node-moving parameters** — determinant coefficients, orbital
  rotations, backflow — with the Jastrow frozen. Better still, use the `Φ = 1/J` form of A5, which
  removes the Jastrow from the surface term identically rather than approximately.
- **The statistics are the harder obstacle.** The effective sample size of the surface term is the
  tube occupancy, not the number of steps: 18 677 of 10⁸ at ε = 0.024. That is two to three orders
  of magnitude more steps per iteration than emin's usual 10⁴–10⁵ configurations. And the present
  kernel makes it worse — after the `|σ|` cancellation the surface sum is an integer count, so an
  infinitesimal parameter change moves configurations across the tube boundary in jumps and the
  correlated-sampling differences that emin lives on are pure boundary noise. A kernel vanishing
  smoothly at `|σ| = ε` fixes the derivatives and the correlated differences at once; these are one
  problem, not two.

And the motivation itself is gone. The scheme was proposed to *amplify the nodal error signal*;
the null test showed what gets amplified is the Jastrow, and stage 1 showed that removing the
Jastrow exactly leaves the determinant's amplitude in its place. Minimizing `⟨H⟩ + λΔ` would
therefore drive `D` towards whatever is compact rather than towards a better node — and on N it
would drive it the wrong way outright, since the better of the two nodes there scores 6% worse.
Track C is dead unless A6 works, in which case it has to be rebuilt on `E^nda(Φ)` with the fixed
weight rather than on `Δ` — and note that A6 removes the first of the four objections outright, since
a `Φ` that does not contain `J` cannot be steered by it.

**Priority: count the domains of the trial function.** That is the measurement with a known,
measured 46× signal on Be, it is cheap, and it answers a question the geminal and pfaffian work is
already asking. **But know in advance what it cannot do:** the `c₂` scan shows the count is 2 for
every `C ≠ 0`, over which `E_FN` still varies by 2.09 mHa with a clean quadratic minimum. So the
counter is a *gate*, not a ranker — it separates the four-domain forms from the two-domain ones and
says nothing about the 2 mHa of geometry left inside the correct topology. Track A is **negative in
everything tried so far, with one designed repair left** — A1–A4 done, A4 closed on our side by
reproducing the analytic Table 1 and leaving Table 2 to ref. [17] and the authors; A5 failed with
all three candidate scalars (`Δ` and `E_kin^nda` on the Jastrow null test, the `Φ = 1/J` descriptor
on the three-system stage 1), and stage 2 is disqualified on paper before being built. **A6 — the
`Ψ`-independent one-particle-product weight of Eq. (18) — is the exception**: every A5 failure is
explained by a weight built from the measured `Ψ`, A6 is the only candidate that is not, its first
two tests are cheap and analytic, and the `c₂` scan is now a proper benchmark for the third. Expect
no number from the rest of A. B is a separate, harder project
with a real chance of being infeasible for the systems we care about — the acceptance fraction in
B2 decides that, and it is cheap to measure. Do not treat B1 as progress; it is repair work on a
path that may not be taken.

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
