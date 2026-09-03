---
name: casino-run
description: >
  Use this skill to run the Fortran CASINO code and read its results: writing or editing an
  `input` file, any CASINO input keyword (all 304 of them are dumped in
  references/keywords.md with type, level and default), `runqmc` and its options, the
  runtype recipes (vmc / vmc_opt varmin / vmc_opt emin / vmc_dmc), which files a run reads
  and writes (gwfn.data, stowfn.data, correlation.data, config.in/out, out, vmc.hist,
  dmc.hist, expval.data), extracting energies with envmc / quickblock / reblock /
  extrapolate_tau, and running CASINO on an examples/ directory to produce the reference
  numbers PyCasino is compared against. Trigger on: casino, runqmc, casinohelp, input
  keyword, `out` file, envmc, endmc, reblock, config.in, reference run, CASINO_ARCH.
  See the qmc skill for the PyCasino side and profiling for benchmarking.
---

# Running CASINO

Reference implementation PyCasino is validated against. Installed at `~/bin/CASINO`
(v3.1.0), binaries in `~/bin/CASINO/bin_qmc` (already in `PATH`),
`CASINO_ARCH=linuxpc-gcc-parallel.openblas`.

The codebase itself (src/ map, manual map, theory) is documented in
`~/bin/CASINO/SKILL.md` — read that when *modifying* CASINO, this one when *running* it.

---

## Keyword reference

`references/keywords.md` — all 304 input keywords: title, type, level, **default value**
(taken from the `esdf_*()` call site in `src/`, which `casinohelp` does not print), and the
full description. Grep it instead of guessing; it is the authoritative source when checking
`casino/readers/input.py` or `docs/source/tutorial/config.rst`.

Live equivalents:

```
casinohelp <keyword>       # one keyword: type, level, full description
casinohelp all             # one-line summary of every keyword
casinohelp search <text>   # keywords whose description mentions <text>
```

Regenerate the dump after a CASINO upgrade: `python references/gen_keywords.py`.

Types: `Logical`/`Boolean` = `T`/`F`, `Physical` = number + optional unit,
`Block` = `%block name … %endblock name`. Levels `Basic`/`Intermediate`/`Expert` only
affect how prominently the manual documents them, not validity.

---

## The `input` file

ESDF format: `keyword : value`, `#` starts a comment, order irrelevant, case-insensitive.
Blocks use `%block`/`%endblock`. A minimal all-electron VMC run:

```
neu               : 1
ned               : 1
periodic          : F
atom_basis_type   : gaussian
runtype           : vmc
newrun            : T
vmc_equil_nstep   : 5000
vmc_nstep         : 10000000
vmc_nblock        : 10
vmc_nconfig_write : 0
use_jastrow       : F
```

Notes that bite:

- `vmc_nstep` is the **total** over all MPI processes, not per process. Same for
  `vmc_nconfig_write`.
- `vmc_method` 1 = electron-by-electron (EBES, the default), 3 = configuration-by-
  configuration (CBCS); there is no method 2 any more. The examples tree splits on exactly
  this — `EBES/` inputs omit the keyword, `CBCS/` inputs set `vmc_method : 3`.
- `dtvmc` is ignored when `opt_dtvmc : 1` (auto-tune to ~50% acceptance) or when a `dtvmcs`
  block is present.
- `newrun : F` + `config.in` continues a run; `newrun : T` starts fresh and ignores
  `config.in`.
- `atom_basis_type` picks the orbital file: `gaussian` → `gwfn.data`,
  `slater-type` → `stowfn.data`, `plane-wave` → `pwfn.data`, `blip` → `bwfn.data`,
  `numerical` → `awfn.data`, `none` → HEG (no orbital file).

---

## Running

```
runqmc                     # run in the current directory, all cores
runqmc -p 1                # one MPI process
runqmc -p 4 -P             # 4 processes, echo `out` to the terminal as it is written
runqmc --check-only        # validate input files and stop — cheap sanity check
runqmc -B                  # detach; safe to log out
runqmc dirA dirB           # several directories (implies background)
runqmc -T 10h --auto-continue   # needs max_cpu_time/max_real_time in `input`
```

**On this machine use `-p 4`, never `-p 8`.** `nproc` reports 8, but those are 4 physical
cores with hyperthreading. Two MPI ranks on the two threads of one core share its FPU and
L1/L2, so the extra four ranks buy almost nothing on a BLAS-bound QMC run and make timings
non-comparable. `runqmc` with no `-p` takes all logical cores — always pass `-p 4`
explicitly (`nproc: 4` through the `casino` MCP server).

**Delete `out` before re-running in a directory that already has one.** `runqmc` *appends*,
it does not truncate, so a restarted or previously killed run leaves one file holding two
runs back to back — two `Geminal setup` banners, two sets of block energies, two FINAL
RESULTs. `envmc` then reads the concatenation and gives nonsense, and any hand-read energy
may come from the wrong run. Nothing warns you. The `casino` MCP server's `overwrite: true`
only lifts the refusal to start; it does not clear the file — **known gap, to be fixed in a
future server version (`~/PycharmProjects/casino-mcp`); until then clear by hand before
every re-run**: `rm -f out config.in config.out correlation.out.* parameters.[0-9]*.casl
vmc.hist dmc.hist`.

`runqmc` refuses to start on a directory that another instance has locked; clear a stale
lock with `-u`. `--force`/`-f` skips the input-file check. `-i` prints what runqmc thinks
the machine looks like. Debug build: `-d`/`--debug` (or `-g` for gdb) — requires that build
to exist.

Files consumed: `input`, the orbital file, `correlation.data` (Jastrow/backflow/determinant
parameters), `*_pp.data` (pseudopotentials), `config.in`, `expot.data`, `mpc.data`.
Files produced: `out` (main log), `vmc.hist`/`dmc.hist`, `config.out`,
`correlation.out.N` (one per optimization cycle), `expval.data`.

**Never overwrite a reference run.** The `out` files under `examples/` are committed
reference data. Copy the directory (or use a `workdir` subdirectory, the pattern
`examples/casino.sh` already uses) before running anything new.

---

## Runtype recipes

As used in this repo — each leaf under `examples/<basis>/<system>/<method>/<CBCS|EBES>/`
is one of these.

**`runtype : vmc`** — plain VMC on a fixed wave function (`Slater/`, `Jastrow/`,
`Backflow/`). Needs `correlation.data` unless `use_jastrow : F` and `backflow : F`.

**`runtype : vmc_opt`** — VMC + wave-function optimization (`*_varmin/`, `*_emin/`).
Set `opt_method : varmin|emin|madmin|varmin_linjas`, `vmc_nconfig_write` = size of the
optimization sample (equal to `vmc_nstep` in the examples), and the `opt_jastrow`,
`opt_backflow`, `opt_det_coeff`, `opt_geminal` switches. Cycles come from `opt_cycles` or
an `opt_plan` block:

```
%block opt_plan
1 fix_cutoffs=T
2
%endblock opt_plan
```

Each cycle writes `correlation.out.N`; the last one is what you copy over
`correlation.data` for the production run.

**`runtype : vmc_dmc`** — short VMC to generate `dmc_target_weight` configs, then DMC
(`*_dmc/`). `vmc_nconfig_write` must be ≥ `dmc_target_weight`. `dmc_method` 1/2 =
electron-by-electron / configuration-by-configuration. Set `dtdmc`,
`dmc_equil_nstep`/`dmc_stats_nstep` and their `*_nblock`. With pseudopotentials decide
`use_tmove`.

Other runtypes worth knowing: `opt` (one optimization from existing configs, no VMC),
`opt_vmc` (cycles the other way round), `dmc_equil`/`dmc_stats`/`dmc_dmc` (DMC alone,
starting from `config.in`; `dmc` is now a synonym for `dmc_dmc`), `gen_mpc`, `gen_blip`,
`gen_mdet_casl`, `plot`.

---

## Reading the results

```
envmc                # VMC energy + error + variance + CPU time, from `out`
quickblock           # reblocking of vmc.hist/dmc.hist, autodetects equilibration
reblock              # interactive reblocking, look for the plateau in the error
extrapolate_tau      # DMC energy extrapolated to dtdmc → 0 over several runs
clearup              # delete run products from a directory
```

`envmc` and `quickblock` work here. **`endmc` misparses numbers under the ru_RU locale**
(`printf: 0.08333: недопустимое число`) and prints garbage — use `quickblock` on
`dmc.hist` instead.

The `out` file itself carries the per-block energies, the final `FINAL RESULT` section, the
timing breakdown, and — for optimization — the parameter values and variance per cycle.

---

## Comparing against PyCasino

Both codes read the same directory. PyCasino is launched on the directory from outside
(`pycasino <path>`, or `mpiexec pycasino <path>` — see `examples/casino.sh`) and writes
`pycasino.log`; CASINO is launched from inside it and writes `out`. So a comparison is:

1. `cd` into the example directory, `runqmc -p N` → `out`, `config.out`.
2. From the repo root, `pycasino <path>` → `pycasino.log`.
3. Compare energy and variance from `envmc` against the PyCasino block statistics.

For a bit-level check of the wave function rather than the statistics, use CASINO's
`config.out`: it stores the configurations and their local energies, so evaluating the
PyCasino `E_L` on those exact positions isolates wave-function bugs from sampling
differences (agreement to ~1e-14 is expected).

Before either run, validate the directory:

```
pycasino --check <path>
```

`casino/readers/validate.py` checks the `input` against the same 304-keyword dictionary
(`casino/readers/keywords.py`, generated from `src/esdf_key.f90`): unknown keyword with a
did-you-mean, wrong value type, block written as a scalar, missing mandatory keyword for
the runtype, values PyCasino has not implemented, `vmc_nconfig_write` below
`dmc_target_weight`, `opt_backflow` without `backflow`, and the data files the run needs
(`gwfn.data`/`stowfn.data`, `correlation.data`, `parameters.casl`). It runs automatically
at the start of every PyCasino run and raises `InputError` before anything is computed.

It also warns, once per run, which keywords in the file PyCasino does not read — that is
the list to check whenever a CASINO and a PyCasino number disagree, since CASINO acts on
them and PyCasino does not.
