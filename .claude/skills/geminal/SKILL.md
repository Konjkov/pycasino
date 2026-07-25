---
name: geminal
description: >
  Use this skill when working on the CASINO multi-geminal (MAGP) wave function —
  ~/bin/CASINO/src/geminal.f90, the GEMINAL block in parameters.casl, psi_s:geminal,
  the pfaffian design doc (docs/source/tutorial/pfaffian.rst), or examples/geminal/.
  Covers the CASINO Fortran bugs found and fixed (figem indexing, tol_log_softzero
  threshold, and the backflow+geminal set: orb_sderiv_valid dimension, Hgem
  down-spin sderiv index, emin invalidation, emin zero-wfn guard), why the
  backflow+geminal emin crash is specific to CASINO's emin and cannot occur in
  PyCasino, open-shell support for unequal numbers of up- and down-spin electrons
  (unpaired-orbital columns, the u_n,k CASL parameters),
  the off-diagonal-as-orbital-rotation construction technique, the
  3-geminal junk-cancellation trick, CASL constraint syntax, construction
  pitfalls (degenerate starting points), and the ansatz-choice debate with
  Pablo López Ríos (perfect backflow / FermiNet Appendix B / size consistency),
  including the hold-correspondence-until-practical-results decision. See also
  the qmc skill for how a geminal
  wavefunction interfaces with the rest of PyCasino, and the backflow/omega-backflow
  skills for the unrelated pfaffian-adjacent backflow machinery.
---

# CASINO multi-geminal wave function (MAGP)

**Status (2026-07):** SIX real CASINO bugs found and patched. Two are geminal-only
(VMC, `geminal.f90`): **Bug 1** figem indexing, **Bug 2** tol_log_softzero. Four
more only manifest in the **backflow + geminal** path under `emin` (see the
dedicated section below): **Bug 3** orb_sderiv_valid dimension, **Bug 4** Hgem
down-spin sderiv index, **Bug 5** emin invalidation of stale Hgem/Farray/Harray,
**Bug 6** sticky FPEinfo_wfn flag never reset (the actual emin-crash root cause);
plus a robustness **guard** in `emin.f90` (drop zero-wfn configs instead of
errstop). Consolidated patches: `geminal_jastrow_fixes.patch` (bugs 1+2, verified,
ready for upstream) and `geminal_backflow_fixes.patch` (bugs 3–6 + guard).
**Bug 7 (2026-07-16, `stowfdet.f90`)**: STO basis ignores `load_all_orbitals`
(set by `psi_s=geminal`, `monte_carlo.f90:5167`) — gaussian/pw/blip modules call
`gen_fake_excitations` to put ALL orbitals into the orbmap, stowfdet never did,
so the geminal pool = neu occupied orbitals only and any `g` element referencing
a virtual dies with `PARSE_GMAT_EL ... larger than the total number of orbitals
allowed (2)`. Fix mirrors `gwfdet_setup`: in `stowfdet_setup`, build occupancy
over `maxval(num_molorbs)` bands (nk=1) and call `gen_fake_excitations`, then
recompute `excite` so the existing excitation branch fills `sto_orbmap`; the
generic mask loop in `sto_orb_eval` already evaluates all masked MOs. Verified
on Be/QZ4P (`examples/geminal/Be/HF/QZ4P/EBES/Jastrow_varmin`): pool 2 → 40,
same casl as the working cc-pVQZ example parses. Patch
`geminal_sto_load_all_orbitals.patch`.
**Bug 8 (2026-07-19, `geminal.f90`)**: CASL round-trip broken for constrained
parameters — any parameters.N.casl written during geminal optimization with a
`Constraints` section cannot be re-read (DMC restart or opt continuation dies
in CHECK_G_CONSTRAINT). Chain: `update_geminal_casl` writes values (`%u1` only,
no flag) for determined group members; on re-read `parse_gmat_el` gives any
flagless parameter the default optimizability (fixed); `check_g_constraint`
treats any non-undefined flag as explicit and errstops on the "contradiction"
(even "fixed whereas fixed"). Editing flags in the file does NOT work around it —
the bare-value determined entries must be deleted, leaving one declared member
per group (`apply_constraints` regenerates the rest from the reference on every
read). Fix applied: `update_geminal_casl` now skips non-optimizable g (was: only
fixed) and non-optimizable c (was: also wrote determined), matching
`set_geminal_casl` which already skipped determined. Needs recompile.
**Feature (2026-07-22, `geminal.f90`)**: open-shell support — the module assumed
`nele(1)==nele(2)` everywhere and gave no error if it did not hold. Implemented
via unpaired-orbital columns; see the dedicated section below. Patch
`geminal_unpaired_electrons.patch`. Validated 2026-07-23 on B (1 column) and N
(3 columns), EBES and CBCS, 10x, all agreeing with ORCA SCF.
**VERIFIED 2026-07-15**: with all six fixes, geminal+backflow+Jastrow emin on
all-electron Ne/cc-pVQZ ran 4 cycles × up to 8 linear-method iterations with ZERO
dropped configurations and no errors, converging to E_VMC = -128.9323 vs exact
-128.9376 (~98.6% of correlation energy at VMC level; HF = -128.5435). The guard
never fired — proving every earlier EMIN_MATRIX_GEN crash was the stale flag
(bug 6), not genuinely pathological configurations.

**Ne DMC results (2026-07-15) — key negative result on nodes.** Reference frame:
HF limit -128.5471, exact -128.9376, Ec = 0.3905 (values used by Holzmann-Moroni,
`pdfs/backflow/orbital_backflow.pdf`, PRB/arXiv:1910.07167, Table I).

| wave function                      | E_VMC (Ha)     | E_DMC (Ha)      |
|------------------------------------|----------------|-----------------|
| OBF (Holzmann-Moroni 2019)         | -128.91956(21) | -128.93129(28)  |
| Slater+BF+J, CASINO, no geminal    | —              | -128.93229(44) 50k pts |
| MAGP+BF+J, radial p only           | -128.9323(53)  | -128.93231(41) full run |
| MAGP+BF+J, p+d after emin (2026-07-16) | — | -128.93048(64) |

Conclusions (FINAL, run complete, DMC series was -128.93181(92) ->
.93186(60) -> .93209(52) -> .93228(44) -> .93231(41), fully stable):
(a) our VMC is 12.7±5.3 mHa below OBF VMC (~2.4 sigma, needs longer run to
harden) with only ~200 parameters; (b) CASINO's standard inhomogeneous backflow
alone already beats published OBF DMC by 1.0±0.5 mHa on Ne; (c) **the geminal
gives ZERO DMC (nodal) gain on Ne**: Delta(gem - nogem) = -0.02±0.60 mHa —
the nodal surfaces of the radial-p^2 geminal and the single determinant are
equivalent, because the radial p^2 block is an amplitude effect and does not
move Ne's nodal surface (consistent with Mitas: Ne single-det nodal topology is
already correct). Both leave 5.3 mHa of fixed-node error. The node-moving
channel is the ANGULAR d^2 one (see improvement ladder below). Strategy consequence: Ne
demonstrates machinery+amplitude; the clean NODAL demonstration system is Be
(single-det topology genuinely broken there; geminal already reproduces MDET
nodes at varmin level) — Be single-det DMC vs geminal DMC is the "practical
result" to show Pablo.

**Ne d^2 experiment (2026-07-16) — SECOND negative result, and this one is
significant: emin DEGRADES the nodes.** Setup: generator section with
`--channels p:2 d:2 --mirror` (p levels rotationally closed, d levels demoted to
diagonal-only ties by the D2h-purity check — which is fine: the diagonal sum
Sum_m d_m(1)d_m(2) IS the 1S recoupling, the node-moving channel). Optimized
parameters after emin (variance decreased vs pre-emin — necessary-but-not-
sufficient sanity signal): g_5,5(2p^2) = -0.3119, g_8,8(3p^2) = -0.0662,
g_14,14(3d^2) = -0.0428, g_30,30(4d^2) = -0.0094, g_5,8 = +0.0053. Healthy
geometric decay (1 : 0.21 : 0.14 : 0.03); the ZERO-seeded g_30,30 activated —
confirms det-lemma cross-term gradients drive dormant channels (zero seeds are
safe when the mirror block rank stays >= Nup). BUT the DMC verdict:
E = -128.93048(64), i.e. Delta(gem - nogem) = **+1.81 +/- 0.78 mHa, 2.3 sigma
ABOVE the control** (an earlier shorter reblock gave -128.93108(88), +1.2 sigma;
lengthening moved it further up — not noise). Interpretation: VMC emin has
almost no nodal signal (fixed-node error is a tiny fraction of E_VMC), so the
optimizer pumps the amplitude-profitable radial channel — prime suspect is
g_5,5 = -0.31, a large 2p^2->3p^2 admixture distorting the 2p radial nodes —
while the genuinely angular d^2 weights stay small (-0.043/-0.009) and cannot
compensate. Isolating experiment (not yet run): same casl with p diagonals
zeroed by hand, d diagonals kept — if DMC returns to the control, the suspect
is confirmed. **Lesson: on a single-reference system the HF-orbital geminal is
an amplitude instrument; letting emin move it without a nodal signal actively
harms DMC. The nodal demonstration must be done on Be** (2s^2->2p^2
quasi-degeneracy, CAS(2,2) nodes worth ~10 mHa in DMC — 30x this experiment's
error bar; textbook case, and natively seniority-0 = one diagonal p block).

**Symmetry selection rule for off-diagonal g (why the generator ties only
equal-size shells):** for a 1S atom, cross-l off-diagonals (s-d, p-d, p-f, ...)
are IDENTICALLY useless — parity ((-1)^(l+l')) and/or absence of L=0 in
l (x) l' for l /= l' forbid any rotation-invariant combination; the invariant
Sum_m phi_l,m(r1) phi_l',m(r2) exists only for l = l'. By Wigner-Eckart the
energy gradient of such a parameter is exactly zero — the optimizer would only
random-walk it on sampling noise. CASL/CASINO would accept e.g. `g_5,14`
syntactically; do not add it. (For molecules the rule is per-irrep, not per-l —
cross-l blocks within one irrep are allowed there.)

**Pfaffian context and the "native RAS replacement" framing.** Bajdich-Mitas
(PRL 96, 130201 (2006); PRB 77, 115112 (2008)) show the SAME picture for
Pfaffians: nodal gains where HF nodes are broken by quasi-degeneracy (Be, B, C
~ small-MCSCF quality), near-nothing on single-reference atoms; multi-Pfaffian
expansions converge slowly. Practical formula: geminal/Pfaffian + Jastrow +
backflow = a native-CASINO replacement for the "external RAS -> molden2qmc ->
mdet" pipeline for PAIR-type (seniority-0) nodal effects — no external code,
parameters ~ #shells not #dets, and coefficients optimized WITH the Jastrow
(external CI coefficients are optimal for the bare determinant, not J*D).
Coverage: seniority-0 doubles + orbital rotations (off-diagonal g) = the class
that matters for quasi-degeneracy/bond-breaking; unpaired doubles/triples/spin
recouplings need full RAS or the STU Pfaffian (triplet pairing, pfaffian.rst).
Size-consistency ladder (for the Pablo debate): CASSCF with union active space
IS size-consistent (full CI in the active space factorizes); RASSCF is NOT
(excitation-count truncation, CISD-like); DOCI is; bare AGP is not (factorized
lambda_i*lambda_j coefficients); **J*AGP is — Neuscamman, PRL 109, 203001
(2012)**: a flexible Jastrow restores AGP size consistency, so the objection is
weak against exactly the J*geminal combination CASINO uses (proved for
Hilbert-space JAGP; "in the flexibility limit" argument for real-space J).

**Ne geminal improvement ladder (priority order, from the natural-geminal
expansion Phi = sum_l c_l R_l R_l sum_m Y_lm Y_lm — currently only l=1 radial):**
1. **d^2 channel** for the 2p pair: tied diagonals over all 5 real-harmonic m of
   the first d shell (one 1S-invariant constraint group), plus tied
   inter-d-shell off-diagonals; cc-pVQZ has 3 d shells. This is the channel that
   MOVES NODES. Mirror G3 gets the same block (only increases its rank — safe).
   **TESTED 2026-07-16 — NEGATIVE**: under emin the d weights stay small while
   the radial p channel inflates and DMC ends 2.3 sigma ABOVE the no-geminal
   control (see the d^2 experiment block above). Ladder items below remain
   theoretically ordered but the Ne line of attack is closed; effort moves to Be.
2. Free c_2 (weight of G2 vs G1) — one parameter.
3. 2s correlating s-block {2s,3s,4s,...} (2s is currently a FIXED anchor);
   2s^2->np^2 arises via det-lemma cross terms; if weights conflict, dedicate a
   G4/G5 mirror pair to the 2s pair (toward strongly orthogonal geminals).
4. CISD initialization: use ORCA CISD + molden2qmc (project-to-CISD machinery)
   to seed ALL channels with correct natural-expansion coefficients instead of
   hand seeds — removes the degenerate-starting-point pitfalls entirely.
5. f^2 channel (~3-4x smaller than d^2), then 1s core pair (last).
Ceiling of the singlet-AGP form: interpair angular recouplings need triplet
pairing = STU Pfaffian (pfaffian.rst roadmap; Bajdich showed PF>AGP on atoms).
After every addition: short varmin sanity check (KEI/TI/FISQ agreement) before
emin — six dormant bugs found so far, assume more exist.

**Optimized Ne g-block structure (evidence the geminal does real pairing):**
diagonalizing the optimized 4x4 correlating block {5,8,17,40} (occupied 2p +
3 virtual p shells, one Cartesian component) gives lambda = -0.1364, -0.0455,
-0.0167, -0.0036 (ratios 1 : 0.33 : 0.12 : 0.03), rank-1 residual 34% —
a genuinely MULTI-RANK natural-geminal spectrum with geometric decay (true
pairing, NOT representable by any single-determinant orbital rotation). Contrast
Be, where the optimized block collapsed to rank-1 (|l2/l1|~4%) = disguised
orbital rotation. First direct evidence the optimizer used geminal freedom on Ne;
explains the VMC gain (amplitude), while the missing d^2 explains the absent DMC
gain (nodes).

Be milestone closed
(off-diagonal geminal from HF orbitals reproduces the 4-det MDET nodal surface,
`E=-14.6666(2)` vs `-14.6669(9)`). Ne in progress: same off-diagonal recipe hit two
NEW degenerate-starting-point failure modes (see below) before converging; both
fixed via parameters.casl changes, not code changes. PyCasino (`casino/`) does not
yet read/evaluate GEMINAL blocks — that is Milestone 1 of the pfaffian plan (see
`docs/source/tutorial/pfaffian.rst` and memory `pfaffian-plan`).

Origin: MAGP was implemented in CASINO for a 2014 PhD project (Bugnion), tested
only on electron-gas orbitals and small systems; per the original author
(Pablo Lopez Rios, forum thread t=215) "has seen little use since." This explains
why both bugs below went unnoticed for over a decade — they only manifest for
Gaussian-orbital multi-electron atoms, never tested in that regime before now.

---

## Theory: what a geminal is, and why off-diagonal = orbital rotation

CASINO's multi-geminal ansatz (per `geminal.f90:107-117` and the forum):

```
Psi = Σ_n c_n · det[Φ_n]          Φ_n(r_i, r_j) = Σ_{m,k} g_mk^(n) φ_m(r_i) φ_k(r_j)
```

`Φ_n` is an `Nup×Nup` pairing FUNCTION (for `neu=ned`); `g^(n)` is a symmetric
`norb×norb` matrix per geminal `n`. `det[Φ_n]` means: build the `Nup×Nup` electron
matrix `M_ij = Φ_n(r_i, r_j)` and take its determinant — same machinery as an
ordinary Slater determinant, reused via `wfdet`.

**Diagonal-only reduces exactly to a Slater determinant.** If `g` is diagonal with
weight 1 on exactly the occupied set, `M = AᵀB` where `A_ki=φ_k(r_i↑)`,
`B_kj=φ_k(r_j↓)` are the ordinary orbital-value matrices, so
`det(M) = det(A)det(B) = D_HF` — an EXACT identity (Cauchy-Binet / matrix product),
not an approximation. This is why `Geminal 1: diag(1,1,...,1)` over the occupied
set is always the pure-HF reference term.

**Off-diagonal elements of g ARE orbital rotations.** Any symmetric `g` diagonalizes
as `g = U λ Uᵀ`; substituting `φ → Uφ` absorbs `U` into a redefinition of the
orbital basis, turning the full (off-diagonal) `g` into a pure diagonal `λ` in the
rotated basis. So optimizing off-diagonal `g` elements between an occupied orbital
and same-symmetry virtuals **is** CASSCF-style orbital relaxation, done inside VMC
via the geminal parameters rather than a separate SCF step. Verified end-to-end on
Be: a full symmetric 4×4 block over HF `{2p,3p,4p,5p}` (occ+virt in ONE Cartesian
component) optimized under `emin` converges to a **rank-1** matrix
(`|λ2/λ1|≈4%`), dominant eigenvector `(-0.83,-0.55,+0.11,0.00)` = the compact
NO-like 2p reshaped from diffuse HF virtuals — i.e. the optimizer found the CASSCF
natural orbital on its own, starting from bare HF.

**Cauchy-Binet junk and the 3-geminal fix.** For a 2-electron-per-spin correlating
pair (Be's 2s), a geminal built as `anchor(1s, weight 1) + virtual-p-block(weight ε)`
generates, via the generalized matrix-determinant-lemma expansion of
`det(Σ_a c_a u_a⊗v_a)` over N-subsets of the rank-1 decomposition, BOTH the wanted
`1s²p_i²` CSF terms (linear in ε, from anchor+one-virtual picks) AND a spurious
"both members of the pair went virtual, core empty" term (quadratic in ε in Be's
2-electron case) that costs `~3ε⁴·ΔE_core` (`ΔE_core≈10 Ha` for Be's 1s → huge
penalty at the ε≈-0.19 found by optimization). The fix: a third geminal
`c=-1, SAME block, no anchor` exactly cancels this term —
`G1(HF) + G2(anchor+εΣp²) − G3(εΣp²)` was verified `≡` the 4-det MDET expansion to
`1e-13` via a numpy arbiter (independent `slater.value_matrix` computation), then
confirmed on real CASINO VMC runs (`-14.617(2)` vs MDET `-14.6169(6)` before
optimization; `-14.6666(2)` vs `-14.6669(9)` after `emin`).

**For Ne (all of 1s,2s,2p occupied — no virtual near-degeneracy shell), the same
recipe generalizes** by putting the OCCUPIED orbital itself inside the correlating
off-diagonal block (not as a separate fixed anchor), letting the optimizer rotate
it into virtuals of the same symmetry. The combinatorial "junk" argument is weaker
here (moving the *whole* Ne 2p³ shell into virtuals needs 3 simultaneous virtual
substitutions, a much higher-order and much smaller penalty than Be's core-vacating
term — no `ΔE≈10 Ha` core gap to pay), so the 3rd-geminal mirror is defensive
insurance rather than a proven-necessary term. There is no independent exact target
(unlike Be's CAS(2,4)) — Ne's off-diagonal geminal is capturing ordinary dynamic
correlation, judged by VMC energy improvement under `emin`, not by an exact identity.

---

## Ansatz-choice debate: Pablo's objections, FermiNet Appendix B, our framing

Correspondence with Pablo López Ríos (2026-07) raised two objections to the
geminal/Pfaffian program; both were analyzed in depth and the counter-positions
below are settled project positions.

**Pablo's argument 1 — "perfect backflow".** In principle a DISCONTINUOUS
backflow transformation can change nodal topology arbitrarily or return the exact
wave function; only in practice is backflow a homeomorphism. Related argument:
FermiNet paper Appendix B (Pfau et al., PhysRevResearch 2, 033429; local copy
`pdfs/MACHINE_LEARNING/pfau2020.pdf`), where the universality of a single
generalized determinant is credited to **Marcus Hutter (personal communication)**.
Read the actual construction before citing it: sort electrons by first coordinate,
`φ1 = 1_{j=π(1)}·ψ(sorted)`, `φi = 1_{j=π(i)}` — every row has ONE nonzero entry
and `det = σ(π)·ψ(sorted) = ψ`. The determinant is trivialized to a
sign-of-permutation gadget; the ENTIRE wave function (nodes included) is packed
into one matrix element, so "learning φ1" is exactly as hard as the original
problem. The two gadgets differ instructively: perfect backflow keeps the
determinant working but puts discontinuities on the (a priori unknown) nodal
surface; Hutter trivializes the determinant with discontinuities on the (known)
sorting-tie hyperplanes. In BOTH cases the composite ψ is smooth — at a tie
crossing σ flips AND ψ(sorted) flips by antisymmetry, so `det ≡ ψ` identically;
only the PARAMETERIZATION is discontinuous. Hence the correct rebuttal is NOT
"infinite kinetic energy" (the represented ψ is fine) but REACHABILITY: no smooth
path in parameter space leads to either gadget; gradients through indicators
vanish. Pfau et al. concede it verbatim: the φi are "not learnable by the
FermiNet", which "may partially explain why … in practice we still require
multiple determinants".

**The settled framing:** universality lives in the CLOSURE of an ansatz class;
optimization lives in its SMOOTH INTERIOR. For coordinate backflow the smooth
interior is homeomorphisms → topology frozen (Pablo's own "in practice"). Whether
the smooth interior of single generalized determinants is universal is left open
by Pfau et al. (empirically they need several). The geminal program = choosing a
class whose smooth interior ALREADY contains the right topology. Note MAGP +
backflow is itself a generalized determinant with two-particle equivariant
entries `M_ij = Φ(x_i(R), x_j(R))` — one rung up from `φ(x_j)` without a network.
Empirical footnote: the runaway emin steps (max|x−r| ~ 25 Bohr vs cutoffs ~3)
are precisely the injectivity of the polynomial map breaking down — "beyond the
homeomorphism" observed as a numerical pathology, not a resource.

**Pablo's argument 2 — pocket count vs system size, and size consistency.**
(a) Wrong pocket count matters for Be but maybe not at large N — TRUE for total
energy (Mitas, PRL 96, 240402 (2006): nodal cells of generic fermionic states
fuse to 2 as N grows; multi-pocket pathology is a small-system/high-symmetry
effect), but chemistry lives on energy DIFFERENCES, which are local: a Be-like
topology error at the bond being broken is an O(1) error in a barrier/binding
energy and does not dilute with N. (b) Backflow is size-consistent; single AGP is
NOT (common pairing function pairs electrons across fragments), MAGP inherits the
CISD disease. Mitigation, natural in the g-matrix representation: BLOCK-LOCALIZED
g (fragment-diagonal pairing = strongly orthogonal geminals/GVB) restores size
consistency and keeps parameter count linear — same sparsity machinery as
`prepare_sparse`. Piquant detail: the FermiNet paper itself worries its
determinant sum "bears some resemblance to CISD" w.r.t. size consistency and
finds empirically that equivariant orbitals LEARN size-consistent solutions —
the network analogue of imposing block structure on g (learned vs built-in).
Bottom line for both points: the net buys topology and size consistency with
expressivity and compute; we buy them with structure (pairing + block-g) plus
backflow for smooth nodal deformation. Not "geminal vs backflow" but a hybrid.

**Correspondence status / strategy (2026-07-15):** a reply paragraph to Pablo
covering the Appendix B reading exists in the session log, but the decision is to
NOT continue the theory correspondence for now — the road is long and the
argument must be won with working numbers (geminal+backflow emin on Ne, then the
PyCasino milestones), not words. Write to Pablo again only when there is a
practical result to show. The tested-fixes patch (`geminal_jastrow_fixes.patch`)
+ email draft were already prepared; the backflow patch set
(`geminal_backflow_fixes.patch`) goes out only after verification completes.

---

## Bug 1: stale FIGEM buffers after accepted one-electron moves

`accept_move_geminal`, "Update FIGEM and LAPGEM" block. Validity flags were copied
for ALL electrons, but the log-gradient DATA was copied only for the moved
electron, and addressed with the spin-relative index `ie` where `figem_scr` /
`figem_valid` are dimensioned over the ABSOLUTE electron index
`(3,ngems,netot,nscratch)`. After an accepted move the destination buffer claimed
valid per-electron gradients it didn't actually hold (or held in the wrong slot
for down-spin electrons).

```fortran
! before
do igem=1,ngems
 if(figem_valid(igem,ie,js))figem_scr(:,igem,ie,is)=figem_scr(:,igem,ie,js)
enddo
! after
do i=1,netot
 do igem=1,ngems
  if(figem_valid(igem,i,js))figem_scr(:,igem,i,is)=figem_scr(:,igem,i,js)
 enddo
enddo
```

Plus two guard sites (`get_figem`, `get_lapgem`) that gated the inverse-refresh on
the WRONG index: `if(.not.figem_valid(igem,ie,is))` → must be `ii` (absolute), not
`ie` (spin-relative) — same absolute/relative mixup.

**Symptom:** kinetic-energy estimator triplet inconsistent (`FISQ − KEI ≈ 15 mHa`
at 20σ) in VMC with `psi_s:geminal`; invisible in the wave-function VALUE (only
derivatives were stale), so a `runtype:plot` line scan matched an independent numpy
recomputation exactly — only the KEI/TI/FISQ triplet exposed it. Fixed; posted to
the CASINO forum (vallico.net/casino-forum, thread t=215, post p1180).

## Bug 2: `tol_log_softzero` threshold miscalibrated by ~23 orders of magnitude

`geminal.f90:91-98`, used in `caldet` (both real/complex branches) and two other
sites (`ratio` computation, log-determinant-from-ratio accumulation). A geminal
determinant is declared `FPEhardzero` (singular / excluded from the wavefunction
sum) if `log(det) < tol_log_softzero = -30.0`. The equivalent Slater-module
constant is `-690.0` (`slater.f90:127`) — 23 orders of magnitude in log-space
stricter. For a 5-electron atom (Ne) with a large Gaussian basis (cc-pVQZ), the
PERFECTLY VALID, non-degenerate 5×5 geminal determinant routinely computes to
`log(det)≈-33…-37` (nowhere near double-precision underflow, `exp(-37)≈8e-17`) —
comfortably inside the Slater module's tolerance but outside the geminal module's
`-30` cutoff, so the code discarded the ONLY geminal at essentially every sampled
configuration, collapsing the wavefunction to zero and crashing with an FPE.
**Fix:** `tol_log_softzero = -690._dp` (matches slater.f90). One-line PARAMETER
change; propagates to all use sites automatically. Confirmed via `ke_verbose:T`
KE-check going from "Analytical derivatives misbehave" to "gradient: optimal,
Laplacian: optimal", and a sane VMC energy (`-128.598(64)` vs ORCA HF reference
`-128.543469658951`, 1σ-consistent).

Never manifested for Be (N=2 pairs stay well above -30) — same "narrow original
test coverage" story as Bug 1.

---

## Backflow + geminal under emin: three more bugs, a guard, and why PyCasino never sees this

Turning ON backflow with a geminal and optimizing under `emin` (linear method) on Ne
exposed a whole second cluster of decade-old bugs, none of which fire for
geminal-only VMC or for backflow-only Slater. All found on
`examples/geminal/Ne/HF/cc-pVQZ/{EBES,CBCS}/Backflow_{varmin,emin}`.

**Bug 3 — `orb_sderiv_valid` allocated with the wrong 2nd dimension.**
`geminal.f90:713` allocated `orb_sderiv_valid(nemax,real1_complex2,nscratch)` but the
array is indexed by SPIN (`orb_sderiv_valid(i,spin,is)` at ~4397, and `do j=1,2` in
`accept_move_geminal`); its siblings `orb_val_valid(nemax,2,...)` etc. use `nspin=2`.
For a real wavefunction `real1_complex2=1`, so the 2nd dim is half-size and the
down-spin (`spin=2`) validity flags are written one slot past the end → heap
corruption (`free(): invalid next size (fast)`), or under a `-fcheck=bounds` build
`Index '2' of dimension 2 of array 'orb_sderiv_valid' above upper bound of 1`. Fix:
`real1_complex2` → `nspin`. Patch `geminal_orb_sderiv_valid.patch`.

**Bug 4 — `get_Hgems_down_ii` indexes second derivatives with `idir` instead of
`kdir`.** In the down-spin Hgem routine the first index of `gem_rsderiv_pair_scr`
used `idir` (which runs `{1,2,3,1,1,2}` over the direction PAIRS) where it must use
`kdir`, the running `1..6` counter that maps a direction-pair to the sderiv storage
order `(xx,yy,zz,xy,xz,yz)`. The up-spin routine already used `kdir` correctly. With
the bug, the `xy/xz/yz` cross terms read `xx/xx/yy` instead → the Laplacian degrades
ONLY when backflow is on: `KEI 112.5` vs `FISQ 132.2`, and varmin drifts to an
**unphysical energy below the exact** `-128.9376` (a giant red flag — an upper-bound
method cannot beat the exact energy; when you see it, suspect a broken KE estimator,
not a good wavefunction). Fixed → varmin returns a physical `-128.93(3)`. Patch
`geminal_hgem_down_sderiv.patch`.

**Bug 5 — emin's one-at-a-time parameter perturbation leaves stale Hgem/Farray/Harray
under backflow.** `emin` computes numerical derivatives by perturbing one parameter
at a time via `invalidate_param1_geminal` / `invalidate_gmat_element`. Under backflow
these did NOT invalidate `gem_sderiv_pair_valid`, `Hgem_valid` (g-matrix change) nor
`Farray_valid`/`Harray_valid` (c-coefficient change), so `get_Harray` summed a stale
Hgem from before the perturbation → wrong derivatives → `emin` cycles diverge /
FPE. Fix adds the missing invalidations in both branches. Patch
`geminal_emin_invalidate.patch`.

**The guard — `emin_matrix_gen` must drop zero-wfn configs, not errstop.** Even with
Bugs 3–5 fixed, `emin` still crashed *stochastically* with
`ERROR : EMIN_MATRIX_GEN / Zero wave function encountered` (`emin.f90:1151`). Diagnosis
via a temporary `get_logwfn` print of `max|x−r|` and `min|x_i−x_j|`: on the failing
iteration the wavefunction was a hard zero (`FPEinfo_det=2`) on ~99% of configs, with
**all** geminals zero — *including the fully-fixed Geminal 1*, which the optimizer
cannot touch. That rules out the g-matrix and points upstream to backflow: a
linear-method step blows up the backflow polynomial coefficients so the
quasi-particle displacement explodes (`max|x−r|` **6–25 Bohr** with cutoffs only ~3
Bohr), carrying electrons into the exponential tail where every orbital underflows and
the `Nup×Nup` geminal matrix goes near-singular. It is **not** electron coalescence
(`min|x_i−x_j|` stays ~0.6 Bohr) — it is displacement magnitude. Such a config carries
a negligible reweighting factor and genuinely cannot contribute, so the fix makes
`emin_matrix_gen` **skip** it (`cycle config_loop`, count in `nskip`, report
`Dropped N configuration(s)`) exactly as `emin_eval_fderivs` already discards bad
perturbed points; `Snorm=1/S_matrix(1,1)` self-corrects because `S_matrix(1,1)`
counts contributing configs.

**The pathological config surfaces in TWO forms and the guard must catch both.** A
first pass that dropped only `isZero` (hard-zero `det`, below the `FPEhardzero`
threshold) just moved the crash to the SIBLING branch: the same runaway step, on a
different config, leaves `det` tiny-but-above-threshold so `Ψ != 0` (isZero false)
while the geminal matrix INVERSE overflows and the local energy comes out `NaN/Inf`
(`0*Inf`) -> `isNaN/isInf` errstop `Floating-point exception ... when calculating
energies` (`emin.f90:1149`). Both are the same worthless near-singular config, so the
final guard drops `isNaN.or.isInf.or.isZero` together (base-config eval AND the
derivative loop), keeps the report loud (always on `am_vp_master`, not gated by
`opt_info`), and adds a collapse safety net: if EVERY config is dropped,
`S_matrix(1,1)==0` and a dedicated errstop fires before the `1/S_matrix(1,1)`
division instead of producing `Inf`. Patch `geminal_emin_zero_wfn_guard.patch` (this
one is in **`emin.f90`**, not `geminal.f90`). This fixes the *symptom*; the actual
root cause turned out to be Bug 6 below.

**Bug 6 — sticky `FPEinfo_wfn` flag: THE root cause of the emin crashes.** The
guard converted the crash into `Dropped 2500` (ALL configs of a rank) + collapse
errstop, and the crash survived even `opt_maxiter 1` with FRESH configs each cycle
(died at cycle 6) — impossible for real zeros, since fresh configs are sampled from
`|Ψ|²>0`. Contradiction resolved by tracing the flag lifecycle: `FPEinfo_wfn` is
initialized to `FPEnone` ONCE at module allocation (`geminal.f90:595`), set to
`FPEsoftzero` in `get_logwfn` when the geminal sum evaluates to exactly zero, copied
between scratches (`:1267`) — and NEVER reset on a subsequent successful evaluation:
the healthy branch of `get_logwfn` wrote only `logwfn_renorm_scr`, and
`clear_scratch_geminal` does not touch it (`FPEinfo_det` is fine — `caldet` assigns
it fresh on every call). So ONE genuinely-zero trial evaluation during the emin line search
(a runaway backflow trial x) permanently poisons the optimization scratch `is0`:
every later `wfn_logval` reports zero and every `loggrad/loglap` reports NaN for
perfectly healthy wave functions (the loggrad/loglap isNaN logic reads
`FPEinfo_wfn/=FPEnone`). Hence: line search stops early on "bad" points but Succeeds
with a pre-poison point ✓, next `matrix_gen` fails on EVERY config ✓, no debug
prints at crash time (wfn healthy, flag stale) ✓, survives fresh configs across
cycles (scratch lives for the whole run) ✓, stochastic ✓. Diagnostic asymmetry that
cracked it: `emin_matrix_gen` evaluates with `prefetch_aderiv=.true.` while the line
search (`emin_energy_recalc`) does not — but both read the same poisoned flag.
Fix: one line, `FPEinfo_wfn(is)=FPEnone` in the successful branch of `get_logwfn`.
Patch `geminal_fpeinfo_wfn_reset.patch`; also folded into
`geminal_backflow_fixes.patch` (consolidated bugs 3–6 + emin guard, applies on top
of `geminal_jastrow_fixes.patch`).

**Verification (2026-07-15):** full emin run on Ne, 4 cycles × up to 8 iterations,
ZERO dropped configs (the guard never fired — all previous crashes were the stale
flag), E_VMC = −128.9323 vs exact −128.9376 (~98.6% of correlation energy). The
runaway-trial-step phenomenon during the line search is real but benign — such
trial points are correctly rejected once the flag is honest.

**Why PyCasino never hits this crash (asked directly; important).** It is the SAME
linear method mathematically (`H·Δp = E·S·Δp`), but three implementation differences
in `pycasino.py:vmc_energy_minimization_linear_method` remove every ingredient:

1. **No geminal evaluation at all** — PyCasino only *reads* the GEMINAL block; the
   pairing-orbital matrix that underflows to a hard zero simply does not exist yet.
   This alone is decisive.
2. **Fresh sampling every iteration.** PyCasino calls `self.vmc.random_walk(...)` each
   linear-method step, so configs are drawn from `|Ψ_current|²` → `Ψ≠0` on all of them
   *by construction*. CASINO `emin` instead **reuses a stored config set** and
   evaluates *updated and perturbed* parameters on those fixed configs, plus a
   "manipulation constant" line-search over trial parameter sets on the same stored
   configs — so a step can zero `Ψ` on a stored config. No reweighting of old configs
   with new parameters ever happens in PyCasino.
3. **Analytic derivatives.** PyCasino builds `S`/`H` from analytic
   `value_parameters_d1` / `energy_parameters_d1`; CASINO uses numerical
   finite-difference derivatives with the one-at-a-time `invalidate_param1` machinery
   where Bugs 4–5 lived. That entire class of stale-buffer bug cannot exist with
   analytic gradients.

Consequence: even once PyCasino gains geminal evaluation, this crash still won't
reproduce under the current scheme (fresh sampling + analytic derivatives) — a bad
step just yields a high energy on the next iteration and is corrected, rather than
poisoning a reused config set. The bug cluster is intrinsic to the
*stored-config + numerical-FD + line-search* design of CASINO's `emin`.

---

## GEMSELFCHK: the debugging pattern used to find both bugs

When chasing a geminal-derivative or FPE inconsistency, re-add self-check
instrumentation (compare cached vs freshly-recomputed value/gradient/laplacian,
plus a periodic finite-difference check against the log-wavefunction) directly in
`geminal.f90`, gated by a module-level `DEBUG_GEMINAL` flag:

- `get_logwfn`: on a valid-cache hit, recompute from scratch (clearing the scratch
  buffer and any one-electron-move parent buffer) and compare.
- `wfn_loggrad_geminal` / `wfn_loglap_geminal`: same cached-vs-fresh comparison for
  gradient/Laplacian; every ~19th call also compares the fresh analytic Laplacian
  against a central finite difference of `get_logwfn` (`h=1d-3`).
- Print at most 50 failures (shared counter), with particle index, `|r|` from
  origin, `Re loggem`, and `FPEinfo_det` on FD failures.
- Print a summary (fail/total per check type) in `finish_geminal`.

**One false-positive pattern to recognize immediately:** a "value" cache-check
failure where the REAL part matches exactly and the imaginary part differs by
exactly `2π` (`6.283185307180...`) is NOT a bug — it's a branch-cut artifact of
comparing `COMPLEX(dp)` log-values without reducing modulo `2πi`; `exp()` of the
two is identical, so the physical wavefunction and all VMC statistics are
unaffected. Confirmed by parsing all failure blocks and checking `real` parts are
bit-identical (only `imag` differs by `2π`) before concluding "no bug."

**Always remove this instrumentation once confirmed clean** — it is not meant to
ship. Two clean-up passes have been done in this project (search history for
`GEMSELFCHK`); the local `geminal.f90` diff from upstream should contain ONLY the
bugfixes packaged as `~/bin/CASINO/*.patch` (Bugs 1–5), nothing else
(`DEBUG_GEMINAL=.false.`, no `gemchk_*` names, no temporary `max|x−r|` /
`min|x_i−x_j|` displacement prints — those were diagnostic-only).

---

## CASL `parameters.casl` GEMINAL block: syntax notes

```yaml
GEMINAL:
  Default g optimizability: fixed
  Default c optimizability: fixed
  Geminal 1:
    Parameters:
      c: [ 1.0, fixed ]
      g_1,1: [ 1.0, fixed ]
      ...
  Geminal 2:
    Parameters:
      c: [ 1.0, fixed ]
      g_1,1: [ 1.0, fixed ]          # anchor, untouched this round
      g_5,5: [ -0.05, optimizable ]  # reference element of a tied group
      ...
  Geminal 3:
    Parameters:
      c: [ -1.0, fixed ]             # everything else comes from Constraints
  Constraints:
    2^g_5,5=2^g_4,4=2^g_3,3=3^g_5,5=3^g_4,4=3^g_3,3
```

- Long-form constraint line: `gem^g_row,col = gem^g_row,col = ...`, one bare line
  per tied group (no leading keyword). Also supports block form
  `Equate 1: [ Diagonal: 3:5, Geminals: 2:3 ]` / `Off-diagonal: ...` /
  `All: (r1:r2,c1:c2)`.
- **Exactly one member per constraint group may carry an explicit
  `fixed`/`optimizable` flag** — that member becomes the reference value; ALL
  OTHER members must be left undeclared anywhere (they become `determined` and
  are overwritten by `apply_constraints` from the reference value on every
  parameter update AND immediately after parsing). Declaring two members of the
  same group errors out (`check_g_constraint` / `check_c_constraint`).
- Any `g` element never mentioned in ANY Parameters block or Constraints group
  defaults to **exactly 0** (`set_default_gmat`, confirmed in source: "CURRENTLY
  SET ALL PARAMETERS TO 0 BY DEFAULT" — no special-cased identity default despite
  an unused code comment suggesting one was once planned).
- `c` coefficients: constraints support equality only (no sign) — put the sign in
  a separately-declared `fixed` `c` value (e.g. Geminal 3's `c: [-1.0, fixed]`),
  never inside a c-constraint.
- The **written** `parameters.N.casl` snapshots during optimization only update
  the REFERENCE element of each constraint group — tied `determined` members keep
  printing their stale initial value in the file (cosmetic only: in-memory state
  and any restart are correct, since `apply_constraints` re-syncs immediately
  after parsing). Reported upstream (forum post p1181) as a minor inconsistency
  (the analogous `c`-coefficient write path DOES include `determined` values; the
  `g`-matrix write path in `update_geminal_casl` does not).

---

## Open-shell systems: unpaired-orbital columns (2026-07-22)

Upstream `geminal.f90` is written for `nele(1)==nele(2)` and **does not check it**:
with more up than down electrons the pairing matrices are allocated
`nele(1) x nele(2)` while every BLAS call and every low-level kernel uses the
square leading dimension `nemax`, so the code reads past the end of `gphi_scr`
and takes a determinant of a matrix that was never filled. Silently wrong, not
refused. Fixed by completing the matrix with one column per unpaired electron:

```
M = [ Phi(r_i^up,r_j^down) | phi^u_k(r_i^up) ]      (nele(1) x nele(1))
phi^u_k(r) = sum_n u_n,k phi_n(r)
```

**Key structural fact that makes this cheap:** every low-level kernel — `caldet`,
`calc_inv1` (Sherman-Morrison), `calc_det_ratio`,
`compute_grad/lap_gems_from_pairs`, `update_matrix_from_chscr` — is ALREADY
dimensioned `nemax x nemax` and becomes correct untouched once the matrix is
genuinely square. Likewise `get_figem`/`get_lapgem`, which only slice a row or
column and call those kernels. Changes are confined to: allocations
(`gem_matrix_scr`, `gem_l/rgrad_pair_scr`, `gem_l/rlap_pair_scr`, `gem_inv_scr`
→ square; `neu_ned_r1c2`), the column count in the dgemms that fill them
(`nemax` → `nele(2)`) plus one extra dgemm for the unpaired block in value /
left gradient / left Laplacian, the split of the `do ie=1,nemax` orbital loop in
`get_matrices_igem_nobf` (it asked `which_ii` for a non-existent down electron),
and the up-spin row branch of `get_chscr_igem`. Right derivatives are w.r.t. a
down-spin electron and vanish identically on unpaired columns.

**CASL syntax** — `u_n,k` alongside `g_n,m` in a geminal's `Parameters` block:

```yaml
  Geminal 1:
    Parameters:
      c: [ 1.0, fixed ]
      g_1,1: [ 1.0, fixed ]     # doubly-occupied orbitals
      g_2,2: [ 1.0, fixed ]
      u_3,1: [ 1.0, fixed ]     # orbital 3 -> unpaired column 1
```

`n` is the orbital index in the pool (1:norb), `k` the index WITHIN the unpaired
block (1:nunpaired), NOT the column of `M` (column `k` lands in `nele(2)+k`).
Both ranges are checked in `parse_umat_el`. Column order is arbitrary (permuting
flips only the global sign); several nonzero `u_n,k` with the same `k` make that
column a linear combination. `check_umat` rejects an empty unpaired column
(structurally singular at every configuration). Unpaired orbitals are FIXED:
`read_geminal` errstops for `nele(2)>nele(1)` (swap the spin channels) and, when
`nunpaired>0`, for `complex_wf`, `use_backflow` or `opt_geminal`.

**Why this is the exact HF test.** With `g` diagonal over the first `ned`
orbitals and `u` picking orbitals `ned+1..neu`, `M = U K` with `K` block
diagonal, so `det M = det(A^up) det(A^down)` — the HF determinant identically.
The identity does not care what the orbitals mean, so the casl above works with
any `gwfn.data` (HF or CASSCF NOs): it just mirrors "the first neu MOs", exactly
what the Slater determinant occupies. Hence a `psi_s:slater` run with the same
`random_seed` must reproduce the geminal energy digit for digit.

**Validation (B and N atoms, all-electron, no Jastrow, no backflow, 10x
statistics, 2026-07-23)**. B: `neu=3, ned=2`, `g_1,1=g_2,2=1`, `u_3,1=1`, one
unpaired column (`examples/geminal/B/HF/cc-pVQZ/{EBES,CBCS}/Geminal`). N:
`neu=5, ned=2`, `u_3,1 u_4,2 u_5,3`, three unpaired columns
(`examples/geminal/N/...`). Energies in au (correlation-time errors):

  atom  EBES x10        CBCS x10        ORCA SCF      EBES-CBCS
  B     -24.5327(3)     -24.5326(3)     -24.532967    -0.1+/-0.4 (0.25 sigma)
  N     -54.4044(4)     -54.4036(4)     -54.403718    -0.8+/-0.6 (1.4 sigma)

The acceptance criterion is EBES == CBCS: both schemes are unbiased estimators
of <H> for the SAME Psi, so they must agree, and they do (<1.5 sigma). This is
the exact discriminator between "sampling-scheme noise" and "a bug in one path":
EBES goes through the one-electron path (`get_chscr_igem`, unpaired row entries,
Sherman-Morrison), CBCS through the all-electron recompute
(`get_matrices_igem_nobf`). They build the same wave function. CBCS sits on SCF
for both atoms (1.2 and 0.3 sigma) — CBCS is the arbiter (full recompute, no
Sherman-Morrison). Pitfall recorded: the FIRST 1x runs looked alarming
(B EBES -24.5342(8), N EBES -54.406(1), both ~2 mHa low, 2.3 sigma weighted,
same sign, growing with Z — a plausible-looking systematic). 10x collapsed it
to noise: single VMC runs with a heavy local-energy tail (var ~11 for N) have
optimistic correlation-time error bars; judge geminal-vs-SCF only on-the-fly-
reblocked and at 10x. cusp_correction shifts VMC off SCF by ~0.3 mHa (Gaussian
cusp takes Psi out of the basis span; raises the energy), visible in B's +0.3
mHa but below the noise floor here.

CASINO's own `Kinetic energy check` reports **"Geminals - gradient: optimal,
Laplacian: optimal"** on BOTH atoms (1 and 3 unpaired columns) — the decisive
check (analytic derivatives vs finite differences of log-psi, the same check
that exposed bug 2), and it covers the unpaired columns. Note `TI = (KEI+FISQ)/2`
exactly, so it carries no independent information; virial gives mean FISQ = -E.

**Coverage of the two code paths** (both were edited, test both): EBES exercises
the one-electron path (`get_chscr_igem`, unpaired row entries,
Sherman-Morrison); CBCS exercises the all-electron path
(`get_matrices_igem_nobf`, `calc_gem_matrix`, `calc_lgrad_pair_matrix`, the
split orbital loop). B has ONE unpaired column, N has THREE (`u_3,1 u_4,2
u_5,3`) — a genuine multi-column block; both validated above.
Closed-shell systems take the identical old code path (`nunpaired=0`,
`nele(1)==nele(2)==nemax`, no extra dgemm reached).

---

## Construction pitfalls: avoid degenerate starting points

Two failure modes hit when writing an off-diagonal geminal ansatz for Ne, both
resolved by choosing STARTING VALUES that keep every geminal's determinant
non-degenerate (neither identically zero nor identical to another geminal) for
every electron configuration from the very first VMC evaluation — not just
"eventually, once the optimizer moves":

1. **Seeding the correlating block's own diagonal at the un-rotated HF value
   (e.g. `g_5,5: [1.0, optimizable]`) can make a second geminal EXACTLY equal to
   the reference geminal at t=0**, if that block also contains the same anchors.
   Mathematically harmless in isolation (VMC local energy is invariant to a
   global wavefunction rescale, and `Ψ=2·D_HF` is still just `D_HF`), but it is an
   UNTESTED code path (two geminals with bit-identical log-determinants
   simultaneously) and should be avoided rather than relied on.
2. **Starting a correlating block's ENTIRE diagonal+off-diagonal at exactly 0**
   (matching Be's successful "block never overlaps the reference term" pattern)
   can make that geminal's determinant PERMANENTLY (structurally, for every
   electron configuration, not just accidentally) singular if the block's
   available rank never reaches `Nup` for ANY nonzero seed elsewhere. This is a
   DIFFERENT situation from Be, where the same all-zero start was only singular
   at the literal t=0 point and became generically non-singular as soon as ANY
   optimization step moved parameters (Be's virtual-only pool had rank capacity
   comfortably above `Nup=2`). A geminal that is singular for literally every
   sampled configuration from the start crashed CASINO with
   `COMPUTE_FINAL_VARIANCE: variance is pathological or enormous` — an actual
   NaN/`>1e15` variance right at the very first VMC block, before any
   optimization step, not a slow/noisy convergence.
3. **The fix that worked:** seed a SMALL NONZERO value (not 0, not the reference
   geminal's value) at more than one radial shell of the correlating block (e.g.
   occupied-orbital diagonal `-0.05` AND first-virtual diagonal `-0.02`, tied by
   symmetry across all Cartesian components), so the block's rank comfortably
   exceeds `Nup` from the first evaluation onward on BOTH the `c=+1` geminal
   (which also has anchors, so needs less help) and the `c=-1` mirror geminal
   (which has NO anchors, so its own block alone must reach full rank). Concretely,
   for Ne: `Nup=5`, 2 anchors (1s,2s) + p-manifold. Seeding only the occupied-2p
   diagonal (3 tied terms, one per Cartesian component) gives the mirror geminal
   only rank 3 < 5 — still permanently singular. Adding the first virtual-shell
   diagonal (3 more tied terms) brings it to rank 6 ≥ 5 — resolved.

**Diagnostic signature to distinguish these from a genuine unrelated bug:** check
whether the pathology fires at the VERY FIRST VMC energy evaluation (before any
"Accepted step" optimizer output) — if so, suspect the starting `parameters.casl`
values first (rank-count the nonzero terms per geminal by hand against `Nup`)
before re-opening the CASINO source.

---

## File map

- CASINO source: `~/bin/CASINO/src/geminal.f90` (local copy carries Bugs 1–5
  vs upstream), `emin.f90` (`emin_matrix_gen` zero-wfn guard; linear-method /
  stored-config / manipulation-constant machinery), `slater.f90` (comparison
  constant), `vmc.f90:3852` (`COMPUTE_FINAL_VARIANCE`, `var_too_big=1e15`).
- Patches in `~/bin/CASINO/` — only THREE files are still on disk (2026-07-22);
  the per-bug patches named in the sections above (`geminal_figem.patch`,
  `geminal_orb_sderiv_valid.patch`, `geminal_hgem_down_sderiv.patch`,
  `geminal_emin_invalidate.patch`, `geminal_emin_zero_wfn_guard.patch`,
  `geminal_jastrow_fixes.patch`, `geminal_backflow_fixes.patch`) were folded in
  and deleted — their content is described here, not stored:
  `geminal_emin_residual_fixes.patch` (`geminal.f90` + `emin.f90`),
  `geminal_sto_load_all_orbitals.patch` (Bug 7, `stowfdet.f90`),
  `geminal_unpaired_electrons.patch` (open-shell, `geminal.f90`). Bug 2
  (`tol_log_softzero`) is a one-line PARAMETER edit in `geminal.f90`.
  To tell whether a patch is already in the tree, dry-run it BOTH ways:
  `patch -p1 --dry-run < p` failing while `patch -p1 -R --dry-run < p`
  succeeding means applied. All three above were applied as of 2026-07-22.
- PyCasino linear method: `casino/pycasino.py:868`
  (`vmc_energy_minimization_linear_method`) — fresh `random_walk` per step,
  analytic `value_parameters_d1`/`energy_parameters_d1`, single `dp` step.
- Design doc: `docs/source/tutorial/pfaffian.rst` — full Milestone 1/2 plan,
  "Spurious high-energy configurations and their cancellation" section.
- Examples: `examples/geminal/{Be,Ne}/HF/cc-pVQZ/{EBES,CBCS}/{Geminal,Jastrow_varmin,Jastrow_emin}/`.
  Open shell: `examples/geminal/B/HF/cc-pVQZ/{EBES,CBCS}/Geminal/` (one unpaired
  column) and N (`neu=5,ned=2`, three columns), EBES and CBCS, all validated
  at 10x against ORCA SCF (see the open-shell section's table).
  STO: `examples/geminal/Be/HF/QZ4P/EBES/Jastrow_varmin{,_p2,_p3}/` (stepwise p
  active space; casl hand-written — make_geminal_casl.py parses gwfn.data only).
  QZ4P Be p-MO map (from stowfn.data coefficients): 2p={3(x),4(y),5(z)},
  3p={12(x),13(z),14(y)} — NOTE y/z swapped vs 2p order — 4p={29(x),30(y),31(z)};
  off-diagonal ties must pair same Cartesian components: x:3-12-29, y:4-14-30,
  z:5-13-31. Only 3 p shells in QZ4P (vs 4 in cc-pVQZ).
  **QZ4P emin result (2026-07-17, `Jastrow_emin_p2/`):** p:2 block (2p+3p, 1s
  anchor, seed -0.19, mirror) with the y/z swap handled correctly converged
  -14.598(2) (geminal only) -> -14.6641(2), var 0.0177(5) — 96.6% of correlation
  (HF -14.5730, exact -14.66736), 2.5 mHa above the cc-pVQZ 4-shell milestone
  -14.6666(2). Optimized per-component 2x2 block [[-0.1476,+0.0402],
  [+0.0402,-0.0171]] is rank-1 (|l2/l1|~3.6%) — same disguised-orbital-rotation
  collapse as cc-pVQZ Be. Next: `Jastrow_emin_p3/` adds the 4p shell {29,30,31}.
  STO Ne: `examples/geminal/Ne/HF/QZ4P/EBES/Jastrow_varmin_p{1,2,3,4}/` (stepwise,
  anchors 1s+2s, seeds -0.05/-0.02/0/0 as in gaussian Ne). QZ4P Ne p-MO map
  (47 MOs, component order scrambled per shell): 2p={4(x),3(y),5(z)},
  3p={6(x),8(y),7(z)}, 4p={15(x),16(y),17(z)}, 5p={32(x),31(y),33(z)},
  6p={43(x),44(y),42(z)}; 3d=10-14, 4d=26-30. Ties: x:4-6-15-32, y:3-8-16-31,
  z:5-7-17-33 (z chosen as reference/free member). All four testrun-verified
  2026-07-17 (pool 47, 0 errors).
- CASSCF->AGP conversion (2026-07-18, `examples/geminal/Ne/MP2-CASSCF(8.13)/cc-pVQZ/EBES/Geminal/`):
  ORCA MP2-CASSCF(8,13)/cc-pVQZ Ne (full 2s2p->3s3p3d, E(CAS) = -128.7619,
  source `/mnt/sdb1/quantum_chemistry/!PROJECT/ORCA/MP2-CASSCF(8.13)/cc-pVQZ/Ne`;
  the (8.12) dir is a broken 4-of-5-d active space + def2-QZVP typo in the
  PrintWF jobs — do not reuse). Natural-orbital gwfn.data MO map: 1s=1, 3d=2,
  3p=3-5, 2p=6-8, 3s=9, 3d=10-13, 2s=14 (occupied NOT first — HF-style
  diag(1..5) casl is WRONG here). CI verdict: seniority-0 vs sen>=2 weight is
  30/70 — SAME split as CAS(8,8), adding d did NOT move weight into the pair
  channel (even within d: sen4 0.0046 > sen0 0.0024) — hard AGP-ceiling number
  for the Pablo debate. Pair amplitudes (intermediate norm): 2p^2->3p^2 same-
  component -0.0417, cross -0.0045, 2s^2->3s^2 -0.0190, 2p^2->3d^2 -0.002..-0.020
  (angular structure), 2s-hole channels smaller. Diagonal AGP is DEMOCRATIC (one
  lambda per orbital for every hole pair and component) -> L2-projected lambdas:
  3p -0.01459, 3s -0.01407, 3d -0.01031. Construction: TWO geminals only —
  G1(c=+1) = occupied diag 1 + correlating lambdas; G2(c=-1) = same minus 1s
  (cancels ALL core-vacated Cauchy-Binet junk exactly; no separate reference
  geminal needed, no cubic amplitude mapping — CI coeff = lambda directly).
  Frozen-lambda VMC (no Jastrow), full 10-block run: E = -128.585(2),
  var 31.4(3) — 41.5(2.0) mHa below HF = 19% of in-CAS correlation, consistent
  with 30% sen-0 weight x ~half lost to democracy. External ruler vs emin: physical 3p/2p amplitude ratio 0.042;
  emin p2 found ~0.10, d2-run ratios 0.21 (p) / 0.14 (d) — emin inflates
  amplitude channels 3-8x, confirming the DMC-degradation diagnosis.
  J.AGP emin (`../Jastrow_emin/`, cycle1 varmin J-only, cycles 2-4 emin all):
  E recovered exactly to J.Slater level (-128.904(3)/-128.902(3) vs -128.9015,
  var 3.6); lambda shrank non-uniformly — 3s died (-0.0141 -> -0.0011, pure
  radial, Jastrow eats it), 3p and 3d held at ~-0.0045/-0.0047 (2-3x below
  CASSCF; the part u-term can't express). Frozen-lambda J.Slater-vs-J.AGP was
  14 mHa WORSE before reoptimization (frozen CASSCF amplitudes double-count
  radial pair correlation under J).
  DMC verdict (2026-07-19, `../Jastrow_dmc/`, dt=0.003333, target 1024, no BF,
  emin lambdas frozen): two 50k-point J.AGP runs -128.9248(13) and
  -128.9242(5), combined -128.9243(5), vs pure-HF J.Slater control
  -128.9232(6) — Delta = -1.1 +/- 0.8 mHa (1.4 sigma), NOT significant.
  (A 20k interim reblock of run 2 showed -2.0 +/- 0.9, 2.2 sigma, then
  regressed — do not trust partial reblocks.) Consistent with the earlier
  radial-geminal zero-nodal result; lambda_3d nodal gain, if any, is <~2 mHa
  of the 14.4 mHa HF-node FN error.
  Cross-check: pure-HF STO-orbital control gave -128.9227(5) = GTO control
  within errors (Delta 0.5 +/- 0.8 mHa) — HF-node FN error is basis-independent
  at QZ level; STO run had N_corr 10.9 vs 4.5 (worse Jastrow/acceptance, same
  nodes).
- GEMINAL section generator: `~/PycharmProjects/molden2qmc/make_geminal_casl.py`
  (on PATH via `~/bin` symlink; standalone gwfn.data parser, no PyCasino
  dependency; default = HF-like Geminal 1 only; `--channels l:n --mirror
  --anchors --seed`). molden2qmc >= 4.0.5 also appends an EIGENVALUES section
  to gwfn.data (CASINO ignores it for molecules) for future energy-degeneracy
  level detection in molecules.
- Forum: vallico.net/casino-forum thread t=215 ("Multi-geminal wave functions") —
  figem patch (p1180), casl-write cosmetic issue (p1181), off-diagonal/HF-start
  Be result (p1182).
- Memory: `pfaffian-plan` (project memory file) has the running log of this work
  across sessions.
