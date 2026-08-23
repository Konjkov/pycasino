---
name: orca
description: >
  Use this skill when producing or diagnosing the orbitals a QMC run starts from: writing an ORCA
  input, reading orca.out / mol.out, converting a MOLDEN file to gwfn.data with molden2qmc,
  generating a multideterminant correlation.data from an ORCA CASSCF, or working out
  why a gwfn.data does not describe the state it was supposed to. Covers the two silent
  substitutions ORCA makes on open-shell atoms (RHF to UHF, and configuration-averaged ROHF on a
  degenerate shell), what an unrestricted or averaged orbital set does to a geminal or MDET
  wave function downstream, how to identify the basis, spin type, MO ordering and orbital
  directions from a gwfn.data alone, the layout of the ORCA calculations under
  /mnt/sdb1/quantum_chemistry/!PROJECT/ORCA, and why a VMC energy of a bare determinant is not
  the SCF energy. See the casino-run skill for the CASINO side, geminal for parameters.casl and
  qmc for the wave function in PyCasino.
---

# ORCA: producing the orbitals a QMC run starts from

Everything in `examples/` begins with an ORCA calculation converted by `molden2qmc`. The file that
reaches CASINO carries no record of how it was made, so a wrong keyword upstream shows up much
later as a wave function that quietly is not the one intended. This skill is mostly about that
gap.

Local ORCA is **6.1.0**, and `molden2qmc` code `3` covers it: the MOLDEN output has not changed
since 3.X. The converter's README still says 6.X is untested — the range in the README is stale,
not a warning.

## 1. The pipeline

```shell
orca mol.inp > mol.out                        # writes mol.gbw, mol.molden.input
molden2qmc 3 mol.molden.input gwfn.data --pseudoatoms none
```

`--pseudoatoms` has no default worth trusting: a MOLDEN file does not say whether a
pseudopotential was used, so `none`, `all` or a 1-based atom list must be given explicitly.
`molden2qmc` lives in `~/PycharmProjects/molden2qmc` (the user's own project, so bugs there are
fixable rather than facts of life); `make_geminal_casl.py` in the same repo writes the `GEMINAL`
block of a `parameters.casl` from a finished `gwfn.data`, and `multideterminant.py` writes an
MDET `correlation.data`:

```shell
multideterminant.py 3 casscf.out --excitation 0 --amplitude 0
```

Calculations live under `/mnt/sdb1/quantum_chemistry/!PROJECT/ORCA/<method>/<basis>/<system>/`,
holding `mol.inp`, `mol.out`, `mol.molden.input`, `mol.gbw`, `gwfn.data`, with the QMC runs in
`VMC_OPT/emin/<jastrow-variant>/` beside them. Geometries come from
`../../../../chem_database/<X>.xyz`. CASINO input templates (`vmc.tmpl`, `vmc_opt_energy.tmpl`,
`correlation.tmpl`, the backflow ones) sit in `!PROJECT/`.

## 2. Two silent substitutions on open-shell atoms

Both were found on boron in one session, both had already contaminated finished work, and neither
announces itself anywhere except in `mol.out`.

### RHF becomes UHF

```
! HF cc-pVQZ VeryTightSCF          →   WARNING: your system is open-shell and RHF/RKS was chosen
* xyzfile 0 2 B.xyz                      ===> : WILL SWITCH to UHF/UKS
```

The result is a spin-contaminated determinant (B: E = -24.53296714, ⟨S²⟩ = 0.7611 against 0.75)
and, more consequentially, a `gwfn.data` with `Spin unrestricted: .true.` carrying **two orbital
sets**. Grep the log for `HFTyp` before believing anything.

Downstream this is not cosmetic. CASINO evaluates orbitals per spin channel, so with two sets the
index p means a different function in the α and β channels — for a geminal, `g_p,q` pairs α-p
with β-q (`geminal.f90`, `get_orbvals` → `wfdet(rele, 1, ispin, ..., wfdet_orbmask(1,ispin), ...)`).
The orderings really do differ: on B, α runs 1s, 2s, 2p(occupied direction), then the empty 2p
pair, while β runs 1s, 2s, the 2p pair, then the occupied direction — so a `parameters.casl`
written for restricted orbitals silently pairs the wrong channels. Restricted orbitals are the
right default for any pairing wave function.

### ROHF becomes configuration-averaged ROHF

`! ROHF cc-pVQZ VeryTightSCF UseSym` on a doublet with a degenerate open shell does not give the
single-configuration state:

```
DETECTED OPEN SHELL STRUCTURE
  (1) From orb=  2 to orb=  4 deg= 3 Alpha=1 Beta=0
   TRIPLY DEGENERATE DOUBLET case found
FINAL ROHF SETTINGS
  operator   1:   3 orbitals   1 electrons n(mue)= 0.333
```

The printed occupations are ⅓ on each of the three 2p, and the energy (-24.52886200 for B) is the
value of an averaged functional, **not the variational energy of any determinant**. CASINO cannot
average: it builds 1s² 2s² 2p¹, so that number is not a target and will not be reproduced, and
orbitals averaged over three directions are not optimal for the state actually being sampled.

Force the single configuration with the manual's HIGHSPIN case — for one open-shell electron that
is one singly occupied orbital:

```
! ROHF ANO-pVQZ VeryTightSCF
%scf
  HFTyp       ROHF
  ROHF_case   HIGHSPIN
  ROHF_NEl[1] 1
end
```

Drop `UseSym`: the degeneracy detector that triggers the averaging runs off the symmetry
handling. Other `ROHF_case` values worth knowing: `CAHF` / `SAHF` (average over a configuration
or over a spin state — what you just got by accident, and genuinely useful as a CI starting
point), `USER_CSF` with `ROHF_REF { … }` and `AF_CSF` with `ROHF_AFORBS` for antiferromagnetic
couplings. `ROHF_Mode` (Pulay / GAMESS / Kollmar Fock operators) and `ROHF_Restrict` are
convergence knobs, not physics.

**Which to use is a real trade-off, not a rule.** Averaged orbitals make all three 2p equivalent —
one radial function, axis-aligned components, exact symmetry in a tied `parameters.casl` — at the
price of a higher reference energy. Single-configuration ROHF gives the better determinant but
its occupied orbital carries its own radial function, distinct from the empty ones, so ties
between occupied and virtual channels stop being exact. Decide it by measurement: a bare
determinant VMC run in each tree costs seconds.

## 3. Reading a `gwfn.data` you did not make

A `gwfn.data` records nothing about the method, but everything needed to reconstruct the
important parts is in the numbers. Use `casino.readers.wfn.Gwfn`:

- **Spin type**: the `Spin unrestricted` field, and `np.allclose(mo_up, mo_down)`.
- **Basis**: `shell_moments` plus primitives per shell. cc-pVQZ for B is segmented,
  `[9 9 1 1 1 | 3 1 1 1 | 1 1 1 | 1 1 | 1]` with the largest s exponent 23870; a general
  contraction like `[16 ×5 | 10 ×4 | 5 ×3 | 4 ×2 | 3]` with s exponents to 210400 is an ANO set
  (ANO-pVQZ and friends contract a 6Z-sized primitive set into a QZ-sized one). **Both have 55
  AOs and 15 shells**, so basis-set identity cannot be inferred from the AO count — this is how a
  file from a different basis went unnoticed for months in a directory named `cc-pVQZ`.
- **What each MO is**: for an atom, every p MO's coefficient matrix (3 Cartesian components ×
  radial functions) is exactly rank 1, `d ⊗ r`. An SVD gives its direction `d` and radial function
  `r`, which identifies shells, finds which orbitals are degenerate partners (equal `r`,
  orthogonal `d`), tells the occupied open-shell orbital from the virtuals of the same shell
  (its `r` differs — the open-shell Fock operator), and exposes relative signs. **Do not assume
  MOs are axis-aligned**: a degenerate shell comes out in whatever frame the SCF left it in, and
  UHF files in particular are rotated arbitrarily.
- **Which MO is which shell**, when the log is available: `ORBITAL ENERGIES` with `UseSym` prints
  irreps, which settles the assignment with no analysis at all (B/ROHF: `3,4,5 = 2p (B2u,B1u,B3u)`,
  `6 = 3s`, `7,8,9 = 3p (B3u,B1u,B2u)`, `10-14 = d`). ORCA numbers from 0, CASINO from 1.

## 4. CASSCF for multideterminant work

The project's recipe is two inputs. First MP2 natural orbitals:

```
! RI-MP2 def2-QZVP def2-QZVPP/C VeryTightSCF UseSym
%mp2
  density unrelaxed
  natorbs true
end
```

then CASSCF reading them, with two extra `NOITER` jobs whose only purpose is to print the
wave function in both forms:

```
! UHF def2-QZVP KDIIS VeryTightSCF UseSym
! MOREAD
%moinp "mol.mp2nat"
%casscf
  irrep 6
  nel 3
  norb 4
  CIStep ice
end
$new_job … PrintWF det        $new_job … PrintWF csf
```

`PrintWF det` is what `multideterminant.py` parses. Two facts measured on B, worth carrying into
any comparison: a CAS(3,4) for a ²P atom comes out as **three** determinants (reference plus the
two tied `2s²→2p²` doubles, no open-shell CSF), and VMC re-optimization under a Jastrow shrinks
the CI coefficient by about a quarter (-0.1707 → -0.1308), so external CI coefficients are
systematically too large for a J·D wave function.

## 5. A bare determinant's VMC energy is not the SCF energy

CASINO applies its Gaussian cusp correction, which changes the wave function near the nuclei, so
even a pure determinant lands elsewhere: on B/UHF/cc-pVQZ the VMC energy is -24.533651(227)
against an SCF -24.532967, 0.68 mHa *below* it. Use this to check that a converted file
represents the state intended — the agreement should be at the mHa level, not exact — and never
quote the difference as correlation energy. Comparisons across trees have to be VMC against VMC.

## Pointers

- Converter: `~/PycharmProjects/molden2qmc` (`molden2qmc.py`, `make_geminal_casl.py`,
  `multideterminant.py`, `ORCA/read_gbw.py`, `ORCA/orca_version.py`).
- Manual: `~/quantum_chemistry/orca_manual_6_1_0.pdf` — §3.1.1 is the ROHF options section.
- Calculations: `/mnt/sdb1/quantum_chemistry/!PROJECT/ORCA/`, geometries in
  `!PROJECT/chem_database/`.
- Worked example of every trap above, on boron: the `geminal` skill's B section, and the three
  trees `examples/geminal/B/{UHF, ROHF, ROHF_2P}`.
