# Functional form of the optimized Jastrow factor and backflow transformation

The mathematical definitions (forms, cusp conditions, window, weights, metrics, mean-field χ, local-density hole)
are in [`FORMALISM.md`](FORMALISM.md).

The question: can the CASINO polynomial terms (Jastrow u, χ, f and backflow η, μ, Φ/Θ) be replaced by
simpler analytic forms with fewer parameters and a physical meaning? This is a study of function shapes only.
No VMC or DMC calculation was run.

**Two caveats that apply to every conclusion below:**

- **(a) A good profile fit does not tell us the energy loss.** All errors here are errors in the *shape* of
  the optimized functions, weighted by the HF electron density. How much energy a replacement form loses
  can only be measured by VMC (energy and variance, re-optimizing the new parameters) after the form is
  implemented in PyCasino.
- **(b) The molecular conclusions are preliminary.** They rest on two small molecules, B2H6 and O3.

**Main result: the Jastrow factor.** u and χ collapse onto universal one-scale curves, and f is close to a
function of two variables. For the backflow we found no reliable reduced form: its profiles are not
reproducible between optimizations.

## 1. Data

`collect.py` reads every directory with the PyCasino readers (`casino/readers/jastrow.py`,
`casino/readers/backflow.py`). Dependent coefficients (cusp and no-cusp constraints, α₁, β₁, d₀, d₁, and the
f / Φ constraint systems) are completed by the readers. Everything is dumped to `data/parameters.json`
(248 parameter sets).

| source | systems | orbitals | directories |
|---|---|---|---|
| `examples/gwfn/*/HF/cc-pVQZ/CBCS` | He, Be, N, Ne, Ar, Kr, B2H6, O3 | AE, Gaussian (cusp-corrected) | Jastrow_emin, Jastrow_varmin, Backflow_emin, Backflow_varmin |
| `examples/gwfn/Be/MP2-CASSCF(2.4)/cc-pVQZ/CBCS` | Be | AE, multideterminant | same |
| `examples/stowfn/*/HF/QZ4P/CBCS` | He, Be, N, Ne, Ar, Kr, O3 | AE, Slater | same |
| `examples/ppotential_HF/*/HF/aug-cc-pVQZ-CDF/CBCS` | H, B, C, N, O, F, Ne, B2H6 | PP (in B2H6 only H is a PP; B is AE) | same |
| `.../Jastrow_emin/linear`, `.../Backflow_emin/linear` | as above | PyCasino emin (linear method) | used only as a noise reference |
| `examples/jastrow/3_1/NM`, `examples/backflow/*/NM` | Be (STO) | f only / Φ only, orders N_eN, N_ee = 1..5 | expansion-order scans |

**Stage used.** In every directory we use the last stage, the highest N in `correlation.out.N`:
`correlation.out.4` for emin (opt_plan: 1 varmin cycle + 3 emin cycles), `correlation.out.2` for varmin,
`correlation.out.4` in `linear/`, and `correlation.out.1` in the scans.

**Not readable.** Most `Backflow_emin/linear/correlation.out.*` files (all except stowfn He, Be, N) are not
read by the current reader: a single η cutoff is written with flag 1 while spin_dep = 1. They are skipped
(`casino/` was not modified).

What is in the data:

- Every χ has `Impose electron-nucleus cusp = 0`. The Kato e-n cusp is carried by the orbitals, so the
  constraint on χ is χ'(0) = 0. On u the constraint is u'(0) = 1/4 for parallel and 1/2 for antiparallel
  spins. The C = 3 truncation is used everywhere.
- Every μ and Φ at an AE nucleus has cusp type 1. The AE cutoff is L_g = 1 bohr everywhere.

**Cutoff lengths were barely optimized.** The optimized cutoffs stay within about 10 % of the hand-chosen
round starting values (4, 5, 6, 8 bohr; see `initial_cutoffs` in the JSON). The dependence of L on Z in the
CASINO data therefore mostly reflects those choices, not physics. The physical length scales below come
from the shapes of the profiles instead.

## 2. Method

- **Profiles.** `terms.py` evaluates the terms exactly as `casino/jastrow.py` and `casino/backflow.py` do.
  The e-e and e-n terms are radial profiles for every spin channel and nuclear set. f, Φ and Θ are
  evaluated on triplets (r₁, r₂, r₁₂).
- **Weights.** An unweighted comparison is meaningless: μ of He reaches −0.6 at r = 2 bohr, where there are
  almost no electrons. `density.py` evaluates the HF orbitals from gwfn/stowfn (`Slater.value_matrix`, no
  Monte Carlo) and builds two weights:
  - for χ and μ, the spherically averaged density around each nucleus, weight w = 4πr²ρ_I(r);
  - for u and η, the distribution of pair distances between independent samples of ρ.
- **Error metric.** ε = √(Σw(y_fit − y)² / Σw y²). Because u and χ can trade a constant (§3.2), we also
  report the same metric for the derivative, ε_d.
- **Noise floor.** The same metric between two optimizations of the same system:
  - CASINO emin vs PyCasino emin: the same objective, different sampling and implementation;
  - emin vs varmin.

  A form whose error is below the noise floor reproduces the profile to within what the optimization
  itself determines.
- **Candidate forms.** `forms.py` defines each form as f(r)·w(r/L) with the window
  w(x) = (1 − x)^C (1 + Cx). Since w(0) = 1 and w'(0) = 0, the cusp is set by f alone and does not depend on
  L. At L the window vanishes with C − 1 continuous derivatives, like the CASINO (r − L)^C factor.
- **Fitting.** `fit_radial.py` does a weighted nonlinear least-squares fit, with one L shared by all spin
  channels, as in CASINO.
- **Baseline.** The CASINO polynomial of reduced order N = 2, 3, 4 with the same constraints, refitted with
  an optimized L. This is the honest baseline: it shows how much of the saving comes simply from lowering
  the expansion order.
- **Symbolic regression.** `symreg.py` uses gplearn; PySR is installed but its Julia download is blocked by
  the network proxy. It runs on profiles pooled over all systems, put on a common scale using scales read
  off each profile (no fitted model assumed), with a penalty on expression length.

## 3. Jastrow factor

### 3.1 u(r_ij): an exponential correlation hole

On the scaled axes x = r/b₀ with b₀ = −u(0)/γ, every hole-like u of the 22 systems falls on one curve
(`plots/universal_profiles.png`):

    u(r) = −γ b e^{−r/b} · w(r/L),   γ = 1/2 (↑↓), 1/4 (↑↑)

- **Symbolic regression.** Without being given any form, it returns exactly this function:
  `div(-1.003, exp(X0))` for ↑↓ at every parsimony setting, and `-1.029·exp(-X0)` for ↑↑
  (`results/symreg.md`).
- **Interpretation.** b is the radius of the Coulomb correlation hole. The Kato cusp fixes the depth,
  u(0) = −γb, so each spin channel has **one shape parameter**.
- **Hole radius vs Z.**
  - AE atoms: b is independent of Z, b ≈ 0.8–1.4 bohr from He to Kr, with Be (2.9) as the outlier (power
    law α = 0.05, R² = 0.05). The pair weight is dominated by the valence shell.
  - PP atoms: b ∝ Z^−1.05 (R² = 0.95), i.e. b tracks the radius of the valence shell.
- **Cutoff.** With free L the fitted L goes to the upper bound or to 6–10 bohr: the exponential needs no
  cutoff inside the weighted region. For implementation, L ≈ 5–6 b is enough.
- **Where one parameter is not enough.** Parallel-spin channels with u(0) > 0: Be ↑↑ (1s↑–2s↑) and N ↓↓
  (1s↓–2s↓). These are not holes but a radial "shell" correlation of core–valence pairs. He ↑↓ has a small
  bump at 3.5 bohr.

  For these channels, and for accuracy at the noise level everywhere, use the 3-parameter form `exp2`:

      u = −γ b₁ e^{−r/b₁} − A₂(1 + r/b₂) e^{−r/b₂}

  It is the cusp hole plus a cusp-free long-range tail or bump of depth A₂ and range b₂.

Results for u (22 systems, `results/radial_fits.md`):

| form | params (1 channel) | median ε | median ε_d | ε_d ≤ CASINO/PyCasino noise |
|---|---|---|---|---|
| CASINO, N = 8 | 9 | 0 | 0 | – |
| CASINO refit, N = 2 | 3 | 0.009 | 0.039 | 18/22 |
| CASINO refit, N = 3 | 4 | 0.005 | 0.031 | 20/22 |
| **exp** (b) | 1 + L | 0.029 | 0.084 | 11/22 |
| pade −γb/(1 + r/b) (baseline) | 1 + L | 0.032 | 0.090 | 11/22 |
| RPA −(2γF²/r)(1 − e^{−r/F}) (baseline) | 1 + L | 0.024 | 0.083 | 12/22 |
| **exp2** (b₁, A₂, b₂) | 3 + L | 0.005 | 0.028 | 20/22 |
| noise: CASINO vs PyCasino emin | – | 0.159 | 0.091 | – |
| noise: emin vs varmin | – | 0.377 | 0.168 | – |

The exponential is as good as the RPA and Padé baselines with the same single parameter, and it has the
simplest interpretation. Only 11 of 22 are within the same-method derivative noise; the failures are the
shell-correlation channels described above.

### 3.2 χ(r_iI): a Gaussian bell tied to u

On the scaled axis x = r/r_half, all 24 χ profiles fall on a Gaussian.

- **Symbolic regression** finds `exp(-0.673·X0²)`, which is exp(−ln2·x²) with ln 2 = 0.693.
- **Interpretation.** A: the one-body boost at the nucleus. The width: the region where the e-e hole is
  compensated.
- **Width vs atom size.** r_half ≈ 1.8 r_out for AE atoms (R² = 0.85) and 2.3 r_out for PP atoms
  (R² = 0.93), where r_out is the radius of the outermost shell maximum of 4πr²ρ. In PP atoms r_half ∝ Z^−1.0
  (R² = 0.93).
- **Amplitude vs Z.** χ(0) grows like Z^0.6 (AE) and Z^1.2 (PP), with a poor fit (R² 0.5–0.8).
- **The window alone is the shape.** In fits with free a and L, a runs off to ∞: the window
  (1 − x)³(1 + 3x) is itself a bell. The 2-parameter form χ = A·w(r/L) (A, L) is as good as the 3-parameter
  Gaussian, and L becomes the physical width, r_half ≈ 0.39 L.

**χ is mostly determined by u** (`chi_meanfield.py`, `results/chi_meanfield.md`). To first order in J, the
one-electron density of |D e^J|² is ρ_HF·exp(2χ + 2V), where
V_σ(r) = Σ_σ' (N_σ' − δ_σσ')/N ∫ρ(r') u_σσ'(|r − r'|) d³r'. The χ that keeps the HF density unchanged is
therefore −V + const. A weighted regression χ ≈ α + β V gives:

- R² = 0.83–0.99 for every AE atom except He (0.6);
- β = −0.7 … −2.1 for AE atoms, and −2.1 … −4.7 for PP atoms.

So most of χ undoes the density distortion caused by u. Only the residual, a few percent, actually improves
the density. This gives a **1-parameter χ**: χ = β[V(r) − V(L)]·w(r/L), with V precomputed from the HF
density. It also explains why the u constant and χ trade off. Adding c to every u and subtracting (N − 1)c/2
from χ leaves J unchanged up to normalization for all electrons inside the cutoffs. This is why the value
noise (16–38 %) is much larger than the derivative noise.

| form | params | median ε | median ε_d | ε_d ≤ CASINO/PyCasino noise |
|---|---|---|---|---|
| CASINO, N = 8 | 9 | 0 | 0 | – |
| CASINO refit, N = 2 / 3 | 3 / 4 | 0.012 / 0.003 | 0.134 / 0.064 | 17/26 / 25/26 |
| **window** A·w(r/L) | 1 + L | 0.019 | 0.201 | 11/26 |
| gauss / yukawa / lorentz / sech | 2 + L | 0.014–0.017 | 0.16–0.19 | 13–16/26 |
| **yukawa2** (inner and outer shell) | 4 + L | 0.003 | 0.058 | 25/26 |
| mean-field β V (β, α) | 1 + L | 0.02–0.06 (AE) | – | – |
| noise: CASINO vs PyCasino emin | – | 0.238 | 0.190 | – |

### 3.3 f(r_iI, r_jI, r_ij): a function of the mean distance and r_ij

Triplets are drawn from the independent-electron distribution around the nucleus. We ask which reduced model
reproduces f (`manybody.py`, `results/manybody_separability.md`, 52 spin channels):

| model | coefficients per channel | median R² | fraction with R² > 0.95 |
|---|---|---|---|
| additive A(r₁) + A(r₂) + H(r₁₂): pure duplication of χ and u | 9 | 0.939 | 0.38 |
| **F((r₁+r₂)/2, r₁₂)** | 15 | 0.982 | 0.77 |
| F(\|r₁−r₂\|, r₁₂) | 15 | 0.642 | 0.15 |
| F(r₁+r₂, \|r₁−r₂\|): no e-e dependence | 15 | 0.948 | 0.50 |
| cubic in (r₁+r₂, (r₁−r₂)², r₁₂) | 20 | 0.996 | 0.88 |
| **rank 1: g(r₁) g(r₂) h(r₁₂)** | 10 | 0.980 | 0.79 |

Conclusions:

1. **f largely duplicates u and χ.** With `no_dup_u = no_dup_chi = 0` in every file, a median 94 % of the
   variance of f is the additive part. This part can be moved into u and χ.
2. **The remaining part depends on the mean distance from the nucleus, not on |r₁ − r₂|.** It is a product
   of 1D functions (rank 1, 10 coefficients) to R² ≈ 0.98.
3. **Physical interpretation: a local-density correlation hole.** Along r₁ = r₂ = R, the effective pair
   function u + f(R, R, r₁₂) keeps the Kato slope. Fitted by c(R) − γ b(R) e^{−r₁₂/b(R)}, the local hole
   radius b(R) of all 18 atomic data sets (AE and PP, He…Kr) falls on one curve against the local
   Wigner–Seitz radius r_s(R) = (3/4πρ(R))^{1/3} (`plots/f_local_hole.png`, `f_slices.py`):

       b = 3.24 r_s / (r_s + 2.71)      (275 slices, R²(log) = 0.92)

   In the high-density limit b ≈ 1.2 r_s, as for the homogeneous electron gas. In the valence region b
   saturates at 1.5–3 bohr. For heavy atoms alone b ∝ r_s^0.8–0.87 (Ar, Kr, R² 0.95–0.99), reproduced by
   gwfn and stowfn to 2 digits.

   This suggests replacing u + f by u(r₁₂; ρ̄) = −γ b(r_s(ρ̄)) e^{−r₁₂/b} with ρ̄ the density at the pair.
   That is two universal parameters (B, c) for all atoms, possibly one scale per species.
4. **Expansion-order scan (Be, STO).** The rank-1 product stays at R² > 0.95 for most (N_eN, N_ee) up to
   (5, 5), while the polynomial models degrade. Higher orders add radial structure but not non-separable
   structure.

### 3.4 Recommendation for the Jastrow factor

| term | recommended form | parameters | CASINO (typical) | status |
|---|---|---|---|---|
| u | −γ b e^{−r/b}·w(r/L) per channel; exp2 (b₁, A₂, b₂) for shell-correlated parallel channels and for noise-level accuracy | 1–3 per channel + L | 8 per channel + L | interpretable: hole radius, tail depth and range |
| χ | A·w(r/L) or A e^{−(r/a)²}·w; alternatively β·V_u(r) (mean field from u) | 1–2 per channel + L | 8 + L | interpretable: boost at the nucleus, width ≈ 2 r_out; β: density-preserving response |
| f | product g(r₁)g(r₂)h(r₁₂); prospectively the local-density hole b(r_s) with 2 universal parameters | 10 per channel + L (product) | 26–40 per channel | product: partly interpretable; local hole: interpretable |

Parameter counts summed over all 23 systems of the main dataset (`param_count.py`, `results/param_count.md`).
*minimal* = one-scale forms; *accurate* = forms within the CASINO/PyCasino noise for almost all systems.

| term | CASINO | minimal | accurate |
|---|---|---|---|
| u | 407 | 73 (18 %) | 155 (38 %) |
| χ | 259 | 56 (22 %) | 143 (55 %) |
| f | 1483 | 587 (40 %) | 1147 (77 %) |
| **Jastrow total** | **2149** | **716 (33 %)** | **1445 (67 %)** |

With the local-density hole in place of f, the Jastrow would come to about 130–300 parameters in total,
6–14 % of the current count. This is not yet shown by any energy calculation, see caveat (a).

**Honest comparison with the polynomial.** A CASINO polynomial of order N = 3 (4 parameters per channel)
reproduces u and χ as well as `exp2`/`yukawa2`. Much of the saving therefore comes simply from the fact
that N = 8 is far above what the optimization determines: the profiles vary by 10–20 % between two emin
runs. The new forms add three things the polynomial does not have:

- fewer parameters at the 1-parameter level;
- parameters with meaning (b, r_half, β) that follow simple laws (b constant for AE atoms, r_half ∝ r_out,
  b(r_s));
- correct behavior without a polynomial fighting the cutoff.

**Forms without interpretation (flagged).** The reduced-order polynomial refits (N = 2–4), the cubic
F(r₁+r₂, (r₁−r₂)², r₁₂) for f, and the rank-1 cubic factors g and h.

## 4. Backflow

**Noise.** The same metric between two optimizations of the same system:

| term | CASINO vs PyCasino emin | emin vs varmin |
|---|---|---|
| η | ε = 0.98, ε_d = 1.55 (3 systems) | ε = 0.73 |
| μ | ε = 5.4 (3 systems) | ε = 1.04 |

The backflow profiles are not determined by the optimization to any useful accuracy. Every form tried is
"within noise", which means the data cannot discriminate between forms. Symbolic regression on pooled η
finds no universal shape: the best ε is 0.86, and η(0) changes sign between systems
(`plots/universal_profiles.png`).

| term | form | params | median ε | notes |
|---|---|---|---|---|
| η | exp: A e^{−r/b} (↑↓), A(1 + r/b) e^{−r/b} (↑↑, zero slope) | 2 per channel + L | 0.107 | A: strength at coalescence, b: range |
| η | exp_osc (× cos kr) | 3 per channel + L | 0.072 | k: shell oscillation, weakly interpretable |
| μ | shell: A (r/a)² e^{−r/a} (AE, ∝ r² at the nucleus), A(1 + r/a) e^{−r/a} (PP) | 2 + L | 0.301 | well at r = 2a |
| μ | ring: A[e^{−((r−r₀)/s)²} + e^{−((r+r₀)/s)²}] | 3 + L | 0.146 | r₀: radius of the displaced shell |
| η, μ | CASINO refit, N = 3 | 5 / 3 | 0.015 / 0.104 | not interpretable |

Φ/Θ has no low-dimensional structure. Even a 20-coefficient cubic reaches a median R² of only 0.70 (Φ) and
0.58 (Θ), and no model reaches R² > 0.95 for Θ. In the Be scans the reduced models do not converge with
order. At the same time Φ/Θ holds **3737 of the 6620 backflow + Jastrow parameters** (56 %).

Recommendation: η and μ can use 2–3-parameter forms (107 and 81 parameters in total instead of 467 and 267).
Whether Φ/Θ can be simplified or dropped altogether can only be decided by VMC, since the profiles carry no
signal.

## 5. Molecules (preliminary, B2H6 and O3 only)

`molecules.py`, `scaling.py`; `results/molecules.md`, `results/scaling.md`, `plots/molecule_vs_atom.png`,
`plots/molecule_anisotropy.png`.

**1. χ and μ of the same species in a molecule vs the isolated atom.** For O we compared O3 with the PP O
atom and with the AE N and Ne atoms (there is no AE O atom in the data). For B we compared B2H6 (B is AE in
both B2H6 sets) with the PP B atom.

| nucleus in molecule | atom | χ: ε (all r) | χ: ε (r > 1 bohr) | χ noise, emin/varmin |
|---|---|---|---|---|
| O in O3, central (gwfn / stowfn) | PP O | 0.34 / 0.30 | 0.87 / 0.75 | 0.40 / 0.41 |
| O in O3, terminal (gwfn / stowfn) | PP O | 0.35 / 0.12 | 0.88 / 0.40 | 0.43 / 0.50 |
| B in B2H6 (gwfn / pp-B2H6) | PP B | 0.48 / 0.41 | 1.38 / 1.15 | 0.43 / 0.53 |

- χ: the atom–molecule difference is of the same size as the optimization noise.
- μ: the differences are 1.4–13 (relative), but μ is not determined anyway (§4).
- H in B2H6 has a χ although the H atom has none (a single electron), so χ of H is a purely molecular
  effect.

**2. Is one form per species with parameters depending only on Z sufficient?** The window form with laws
fitted over AE atoms, A = 0.90 Z^0.58 and L = 6.15 Z^−0.16, predicts:

| nuclei | ε_d, Z-law | ε_d, own fit | ε_d, same-method noise |
|---|---|---|---|
| O in O3 | 0.20–0.47 | 0.12–0.27 | 0.15–0.22 |
| B in B2H6 | 0.64–0.68 | 0.25–0.28 | 0.25–0.31 |
| H in B2H6 | 1.8 | – | – |

- For O the Z-law prediction is about 2× the noise. For B it is off by as much as 2–3× the noise; B is
  interpolated between Be (Z = 4) and N (Z = 7), and Be is an outlier (near-degeneracy).
- It fails completely for H.
- Conclusion: **not sufficient** at the current noise level for B and H. It is roughly within a factor of 2
  for O. A species-level parameter set, fitted once per element in a reference molecule, is the realistic
  option.

**3. Anisotropy.** Along the bond lines we compare, at r₁₂ = 1 bohr, the three-body terms with the
two-body ones. For f the reference is |u(1)|; for Φ/Θ it is the pair-weighted rms of the η displacement
|η(r) r|.

| molecule | f / \|u\|: at nuclei | f / \|u\|: bond midpoint | Φ/Θ / \|η r\|: at nuclei | Φ/Θ / \|η r\|: bond midpoint |
|---|---|---|---|---|
| O3 (AE) | 0.54–0.86 | 0.25–0.49 | 2–4 | 8–15 |
| B2H6 (AE B, gwfn) | 0.17 | 0.20–0.25 | 1.6 | 0.7–0.8 |
| B2H6 (PP H, pp) | 0.02–0.03 | 0.07–0.11 | 1.7 | 0.9–1.3 |

- **Jastrow:** f modifies the pair correlation by 20–50 % in AE molecules, but it is *not* larger between
  nuclei than near them. The radial χ plus the f of each nucleus (i.e. the local-density hole of §3.3) is
  enough; no extra bond-centered term is indicated.
- **Backflow:** in O3 the e-e-n displacement at the O–O midpoint is 8–15 times the typical η displacement,
  so a radial μ + η is *not* enough for O3. Whether this is physics or a symptom of the undetermined Φ
  (§4) cannot be settled without VMC.

**4. Cutoff length vs internuclear distances.**

| molecule | nearest distances (bohr) | L_χ (bohr) | nuclei within L_χ |
|---|---|---|---|
| O3 | O–O 2.40, O…O 4.09 | 4.0–4.3 | 1–2 |
| B2H6 | B–H_t 2.24, B–H_b 2.48, B–B 3.32 | 5.0–5.8 | 7 (all) |

- L_χ ≈ 1.7–2.6 × the shortest bond: every χ reaches past the nearest neighbors.
- L_f, L_μ and L_Φ are 3–6 bohr, similar in range.
- The cutoffs were not really optimized (§1), so these are the starting values within ±10 %.
- The physical widths are shorter than L: χ r_half ≈ 1.35–1.45 bohr (O), 1.8–1.9 (B), 2.8 (H); u hole
  radius b₀ ≈ 1.0–1.1 bohr (O3) and 1.7–1.8 (B2H6). These are comparable to the bond lengths, not to L.

## 6. Recommendations

1. **Implement first:** u = −γ b e^{−r/b}·w(r/L) with the exp2 extension, and χ = A·w(r/L) or a Gaussian.
   Then run VMC on Ne, Ar, O3 and B2H6 against the CASINO N = 8 Jastrow, comparing energy and variance. This
   saves about 80 % of the u and χ parameters. Cost is the same as or lower than the polynomial (one exp
   per pair).
2. **Then f:** first as the rank-1 product (−60 % of f). Next the local-density hole with
   b = B r_s/(r_s + c), whose two universal parameters could replace most of u + f.
3. **Also test:** the density-preserving χ = β V_u (1 parameter per species). If VMC confirms it, χ stops
   being an independent term.
4. **Backflow:** η with 2 parameters per channel, μ with 2–3; for Φ/Θ, test first by VMC whether it can be
   reduced or dropped. It holds most of the parameters and the profiles show no reproducible shape.
5. **Independently of new forms:** lowering the expansion orders (N_u, N_χ = 3–4) already reproduces the
   N = 8 profiles within the optimization noise. This is the cheapest thing to test by VMC.

All these conclusions are about profile shapes (caveat a) and, for molecules, about two molecules only
(caveat b).

## 7. Implementation in PyCasino

`casino/jastrow.py` implements two of the recommended forms (see `FORMALISM.md` §2):

- **u:** the exponential hole u = −γ b e^{−r/b} w(r/L), with one parameter b per spin set. γ = 1/4 (↑↑, ↓↓)
  or 1/2 (↑↓) is taken for each pair, so the Kato cusp holds exactly for any b and L.
- **χ:** the bell χ = A w(r/L), with one parameter A per spin set. χ'(0) = 0 by construction.

Both have analytic value, gradient, Laplacian, single-electron versions, and first derivatives w.r.t. the
parameters. The u term also has the second derivative w.r.t. b. Cutoff derivatives are numerical, as for the
polynomial. There are no constraints, so they drop out of the projector.

The form is selected per set by an optional line in `correlation.data`. Without the line the set is the
CASINO polynomial, and the file stays readable by CASINO. The expansion order must be 0:

```
 START SET 1
 Spherical harmonic l,m
   0 0
 Functional form (0=polynomial; 1=exponential hole -gamma*b*exp(-r/b)*w(r/L))
   1
 Expansion order N_u
   0
 Spin dep (0->uu=dd=ud; 1->uu=dd/=ud; 2->uu/=dd/=ud)
   1
 Cutoff (a.u.)     ;  Optimizable (0=NO; 1=YES)
   6.0                               1
 Parameter values  ;  Optimizable (0=NO; 1=YES)
   1.3                               1       ! b_1
   1.1                               1       ! b_2
 END SET 1
```

- **χ set:** the same line, `Functional form (0=polynomial; 1=bell A*w(r/L))`, placed after
  `Impose electron-nucleus cusp`, which must be 0.
- **Default b:** a u parameter left at 0 starts from b = 1 bohr, the typical hole radius found in §3.1.
- **Examples:** `examples/stowfn/{He,Be,N,Ne,Ar,Kr,O3}/HF/QZ4P/CBCS/Jastrow_emin_analytic` have
  exponential u and bell χ (no f term), with the same input as the neighbouring `Jastrow_emin` (emin
  optimization).
  - For He the ↑↑ channel has no pairs and its b is excluded from the optimization.
  - N, Ne, Ar, Kr and O3 are written by `make_examples.py`. The starting values are the §3 fits to the
    CASINO profiles: `exp` b for the parallel and antiparallel channels, L_u limited to 8 bohr, and the
    `window` A and L per χ set.
  - For N the CASINO spin dep 2 of u is reduced to 1: the ↓↓ channel is shell-like and does not fit the hole.
- **Test:** `TestJastrowAnalytic` in `casino/tests/test_jastrow.py` compares the analytic derivatives with
  numerical ones.

### 7.1 Forms without cutoff (`Functional form = 2`)

The cutoff is needed by the polynomial, a truncated Taylor series that departs from the function it
approximates at large r. The analytic forms decay by themselves, so form 2 drops it:

- **u:** u = −γ b e^{−r/b}. One parameter b per spin set, expansion order 0.
- **χ:** χ = A e^{−(r/a)²}, the Gaussian found by the symbolic regression. Two parameters (A, a) per spin set,
  expansion order 1. χ'(0) = 0 by construction.
- **Cutoff:** the cutoff line of the set is read but ignored. Internally it is ∞ and not optimized, and it is
  written back as `inf`.
- **Derivatives:** all derivatives w.r.t. the parameters are analytic, including the second derivatives
  used by emin.
- **Positive lengths:** the hole radius b (forms 1 and 2) and the Gaussian width a are optimized as ln b and
  ln a, so they cannot become ≤ 0. The Be run with form 1 had drifted to b = −3.79, a growing exponential.
- **Examples:** `examples/stowfn/{He,Be,N,Ne,Ar,Kr,O3}/HF/QZ4P/CBCS/Jastrow_emin_uncut`, written by
  `make_uncut_examples.py`.
  - **f-term:** the f-term of the last CASINO stage is kept. The CASINO u and χ were optimized together
    with it, and for Be f is as large as u + χ (rms 1.64 vs 1.70 over random configurations).
  - **Start values:** the least-squares fit of Σ u + Σ χ to the same sum of the CASINO Jastrow, on
    configurations sampled by VMC from the CASINO Slater–Jastrow wave function, plus a constant.
  - **Why not the profile fits:** fitting the radial profiles one by one weights each function by the
    density, not by how it enters Ψ. For Be such a start gives VMC −14.467 against −14.650, the
    configuration fit gives −14.570.
- **Be result** (this start, CASINO f, varmin + 3 emin with the example input): −14.6463 against −14.6504
  for the CASINO polynomial, variance 0.079 against 0.042.
  - Varmin makes no step here: the first Gauss–Newton step changes the cost by less than
    ftol = 2/√(N−1). Emin does the work.
  - The parallel hole radius keeps growing (3.9 → 7.1 bohr). The Be ↑↑ channel is a shell correlation
    (§3.1) that one exponential does not describe; this is the remaining 4 mHa.
- **Test:** `TestJastrowUncut`.

**Numba cache.** The functions of `casino/wfn.py` are cached together with the Jastrow code they inline, and
the cache is invalidated only by a change of `wfn.py` itself. After updating `casino/jastrow.py`, delete the
cache (`find casino -name '*.nbi' -delete; find casino -name '*.nbc' -delete`), otherwise the old Jastrow
code keeps running inside `Wfn`.

### 7.2 Product f against the full CASINO Jastrow (plan step 1 A)

`product_f.py` fits the full CASINO Jastrow J = Σu + Σχ + Σf, plus a constant, on 50 000 VMC configurations
of the CASINO Slater–Jastrow wave function (last emin stage). Two models are compared:

- **(C):** uncut u and χ (form 2) with the CASINO f fixed, as in the `Jastrow_emin_uncut` examples.
- **(A):** uncut u and χ with a rank-1 product f per f set, nucleus and spin channel, with the CASINO cutoff L:
  f = (1 − x1)^C (1 − x2)^C g(x1) g(x2) h(x12), x = r/L.
  - g = 1 + C x + g2 x² + g3 x³ satisfies the e-n no-cusp condition; g(0) = 1 fixes the scale of the product.
  - h = h0 + h2 x² + h3 x³ satisfies the e-e no-cusp condition h'(0) = 0.
  - 5 parameters per spin channel.

The product f is also fitted alone to the CASINO f. Residuals are rms over the configurations, a.u.
(`results/product_f.md`). "f params" counts the nonzero stored CASINO f coefficients, not the independent ones.

| system | configs | spread | f spread | (C) | f alone | (A) | (A)/(C) | f params | b_par, b_anti |
|---|---|---|---|---|---|---|---|---|---|
| He | 49996 | 0.084 | 0.251 | 0.020 | 0.017 | 0.010 | 0.51 | 58 / 10 | 1.00, 1.58 |
| Be | 49999 | 0.154 | 1.469 | 0.229 | 0.044 | 0.037 | 0.16 | 116 / 10 | 1.07, 3.45 |
| N | 50000 | 0.142 | 0.674 | 0.041 | 0.036 | 0.029 | 0.70 | 174 / 15 | 1.69, 1.45 |
| Ne | 50000 | 0.162 | 1.187 | 0.060 | 0.043 | 0.039 | 0.65 | 116 / 10 | 1.06, 0.97 |
| Ar | 49998 | 0.194 | 1.625 | 0.248 | 0.092 | 0.088 | 0.35 | 116 / 10 | 0.55, 1.16 |
| Kr | 50000 | 0.191 | 1.109 | 0.330 | 0.110 | 0.120 | 0.36 | 116 / 10 | 0.00, 0.73 |
| O3 | 49999 | 0.341 | 1.190 | 0.147 | 0.106 | 0.086 | 0.58 | 232 / 20 | 0.77, 1.05 |

- **(A) passes on all seven systems:** its residual is 0.16–0.70 of the (C) one, with 10–20 f parameters instead
  of 58–232.
- **The product f reproduces the CASINO f:** fitted alone, its residual (0.02–0.11) is 3–15 % of the spread
  of Σf.
- **Σf is large:** it is 3–10 times the spread of J itself. The CASINO u + χ and f largely cancel.
- **(C) is poor for Be, Ar, Kr:** the (C) residual exceeds the spread of J. The uncut u and χ cannot take the
  place of the CASINO u + χ while the CASINO f stays. This explains the poor Be start of the `Jastrow_emin_uncut`
  example (§7.1). Refitting f together with u and χ removes the problem.
- **Kr parallel hole radius:** b_par runs to 0, as in `make_uncut_examples.py`. It needs the same bound.
- **Caveat:** this is a fit of ln Ψ on the CASINO distribution, not an energy. Only emin decides (caveat a).

## 8. Files and reproduction

Run the scripts in this order from `research/jastrow_form`. They need numpy, scipy, numba and matplotlib,
plus gplearn for `symreg.py`. The whole pipeline takes about 30 min; symreg alone takes about 20 min.

```
python collect.py        # data/parameters.json
python density.py        # data/densities.json (HF densities, pair-distance weights)
python plot_profiles.py  # plots/profiles_*.png
python fit_radial.py     # results/radial_fits.{md,json}, plots/{u,chi,eta,mu}/
python chi_meanfield.py  # results/chi_meanfield.md, plots/chi_meanfield/
python manybody.py       # results/manybody_separability.md
python f_slices.py       # results/f_local_hole.md, plots/f_local_hole.png
python symreg.py         # results/symreg.md
python plot_universal.py # plots/universal_profiles.png
python scaling.py        # results/scaling.md, plots/scaling_laws.png (needs radial_fits.json)
python molecules.py      # results/molecules.md, plots/molecule_*.png
python param_count.py    # results/param_count.md
python make_examples.py   # examples/stowfn/{N,Ne,Ar,Kr,O3}/HF/QZ4P/CBCS/Jastrow_emin_analytic
python make_uncut_examples.py  # examples/stowfn/*/HF/QZ4P/CBCS/Jastrow_emin_uncut
python product_f.py          # results/product_f.md (VMC sampling, hours for Kr and O3)
```

| file | content |
|---|---|
| `terms.py` | evaluation of the CASINO terms |
| `forms.py` | candidate forms and the window |
| `radial.py` | targets, weights, fitters and the noise metric |
