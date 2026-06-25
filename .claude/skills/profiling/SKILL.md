---
name: profiling
description: >
  Use this skill when profiling or benchmarking pycasino (casino/profiling.py),
  measuring timings of slater/jastrow/backflow/cusp/markovchain kernels, or when
  running/writing the test suite (casino/tests/, pytest). Trigger on: profiling,
  benchmark, timing, performance, slow kernel, run tests, pytest, test_slater,
  test_backflow, numerical derivative test, CI.
---

# Profiling & testing pycasino

## Profiling (`casino/profiling.py`)

`Profiler(Casino)` builds a full `Casino` instance from an example config, then
times each kernel by running it `steps` times over randomly displaced electron
positions (`dr = 3.0 AU`). Methods:

- `slater_profiling` — value / laplacian / gradient / hessian / tressian
- `cusp_profiling` — same set for the cusp correction
- `jastrow_profiling` — value/laplacian/gradient and their `_parameters_d1`
- `backflow_profiling` — value/gradient/laplacian and their `_parameters_d1`
- `markovchain_profiling` — full VMC random walk

Expensive kernels run fewer steps and report a `* N` multiplier in the log
(e.g. tressian uses `steps // 10` and prints `%8.1f * 10`).

### Running it

Paths are relative (`../examples/...`), so **run from the `casino/` directory**:

```bash
cd casino && python profiling.py
```

`__main__` loops over many molecules (He, Be, N, Ne, Ar, O3, Kr) for stowfn,
gwfn and Be HF/CASSCF — this takes **hours** (each kernel JIT-compiles, then
runs thousands of steps). To profile one thing, import `Profiler` and call a
single method on a single config instead of running the whole script.

### Logging gotcha

`casino/pycasino.py` calls `logging.basicConfig(filename='pycasino.log')` at
import, so all `logger.info` output goes to **`pycasino.log`**, not the console.
To see timings on stdout, configure logging before constructing `Profiler`:

```python
import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
from profiling import Profiler
```

### NRT allocation stats

`rtsys.get_allocation_stats()` raises `RuntimeError: NRT stats are disabled`
unless numba NRT stats are enabled (`NUMBA_NRT_STATS=1`). The `rtsys` lines are
commented out throughout the file for this reason — keep them commented unless
you explicitly enable stats.

### Known hotspot

`slater tressian` is the dominant cost: the third-derivative tensor is `N³`
(`N = nelec*3`) and falls out of L3 cache for O3/Kr. See the `qmc` skill section
"Tressian bottleneck" — the tensor is only ever needed contracted to a length-N
vector, so use `slater.tressian_dot` instead of materialising `N³`.

## Testing (`casino/tests/`)

`unittest.TestCase` classes run under pytest. Configured in `pyproject.toml`
(`pytest>=8.4`, ruff line-length 150, single quotes). CI (`.github/workflows/
tests.yml`) just runs `pytest`.

```bash
pytest                                   # whole suite
pytest casino/tests/test_slater.py -q    # one file
pytest casino/tests/test_slater.py::TestSlater::test_tressian   # one test
```

### Running rules (important)

- Run the suite **single-threaded** (`NUMBA_NUM_THREADS=1 python -m pytest ...`).
  Parallel runs trip over the numba cache.
- Before a clean run, **delete all numba caches** so stale compiled objects don't
  mask changes. Numba has **no clear-cache command/API** — the documented way is
  to remove the cache directory. Here only the cache files are removed (the
  `__pycache__` dir also holds `.pyc`):
  ```bash
  find casino -name "*.nbi" -delete -o -name "*.nbc" -delete
  ```
  (`.nbi` = cache index, `.nbc` = compiled object.) Caches live in `__pycache__`
  next to the sources; if that dir isn't writable numba falls back to
  `$HOME/.cache/numba` (or `$NUMBA_CACHE_DIR`). Never delete the cache while a
  numba process is running — it raises `OSError` at the compilation site.
- The project is an **editable install**, so imports resolve to the working-dir
  `casino/` (that's the cache that matters). But a stale physical copy may sit in
  `.venv/lib/python3.*/site-packages/casino/__pycache__` from an earlier
  `pip install .` — clear it too so it doesn't confuse you:
  ```bash
  find casino .venv/lib/python3.*/site-packages/casino \
    -name "*.nbi" -delete -o -name "*.nbc" -delete
  ```
- After deleting caches, the first run **recompiles everything** — wait several
  minutes for the caches to regenerate before judging timings or assuming a hang.

Test files and counts: `test_slater` (7), `test_backflow` (8), `test_jastrow`
(8), `test_cusp_factory` (7), `test_cusp` (4), `test_harmonics` (4).

### Pattern

Each `setUp` seeds `np.random.seed(1)`, reads a config from
`tests/inputs/<Kind>/<Mol>` (only **He** fixtures exist), builds `Wfn`, and
generates `n_vectors`. Analytic derivatives are checked against finite-difference
references via `pytest.approx`, e.g.:

```python
assert slater.gradient(n)  == pytest.approx(slater.numerical_gradient(n))
assert slater.tressian(n)[0] == pytest.approx(slater.numerical_tressian(n), rel=1e-4)
```

Higher derivatives need looser `rel` (hessian 1e-5, tressian 1e-4) because the
numerical reference loses precision.

### Why the coarse `rel` tolerances (don't tighten them)

The looser `rel=` are **not** sloppiness — they are the precision floor of the
finite-difference *reference*, not of the analytic code. Four comparisons use a
coarse tolerance:

| comparison | file:line | derivative order | `rel=` | required (elementwise) |
|---|---|---|---|---|
| hessian | `test_slater.py:43` | 2nd | `1e-5` | 1.8e-6 |
| tressian | `test_slater.py:46` | 3rd | `1e-4` | 7.5e-5 |
| jastrow `value_parameters_d1` | `test_jastrow.py:68` | 1st (params) | `1e-4` | 1.2e-5 |
| backflow `value_parameters_d1` | `test_backflow.py:79` | 1st (params) | `1e-4` | 1.9e-5 |

`gradient`, `laplacian`, `energy_parameters_d1` stay at the default `rel=1e-6`.

Central differences have two competing errors: truncation `O(h²)` and round-off
`O(ε/hⁿ)` for order `n`. Balancing them gives the optimal step and best accuracy:

| order | optimal `h` | best rel.err |
|---|---|---|
| 1 | `eps^(1/3)` | `eps^(2/3)` ≈ 4e-11 |
| 2 | `eps^(1/4)` | `eps^(1/2)` ≈ 1.5e-8 |
| 3 | `eps^(1/5)` | `eps^(2/5)` ≈ 1.5e-6 |

These steps are exactly `delta`, `delta_2`, `delta_3` in `casino/__init__.py`.
Empirically (He, seed 1, sweeping `h`) the U-curve minimum lands right on those
steps: hessian min ≈3e-8 at `eps^(1/4)`, tressian ≈1e-6 at `eps^(1/5)`. So:

- **hessian / tressian**: higher order ⇒ fewer correct digits in the reference
  ⇒ looser `rel`. Tightening below the table values makes the test fail on a
  correct implementation. tressian is the tightest (only ~1.3× margin).
- **parameter `value_parameters_d1`**: formally 1st order (optimum ~1e-11), but
  needs `1e-4` because the absolute step `delta` is **not scaled to the parameter
  magnitude** — params span a wide range (jastrow `[2.8e-7 … 5.4]`, derivative
  values up to ~1e3), so for large-curvature params truncation dominates and the
  optimum shifts to smaller `h`; at the fixed `delta` only ~1e-5 is achievable.
- The binding constraint is `pytest.approx`'s **per-element** relative error on
  small-magnitude entries — global-normalised error looks better than what the
  test actually requires (e.g. hessian 3e-8 global vs 1.8e-6 per-element).

### Notes

- Tests JIT-compile numba kernels on first call, so a cold run is slow
  (minutes); `cache=True` makes reruns fast. Heavy tests like
  `test_slater.py` can take many minutes — don't assume a hang.
- Duration (single-threaded): a full **cold** run (caches deleted, everything
  recompiles) is ~**38 min** for the whole suite (37 tests). With warm numba
  caches reruns are much faster. Budget accordingly — don't kill a run early.
- Do **not** modify fixtures under `tests/inputs/` unless explicitly asked.
- `test_backflow.py::test_wfn_energy_parameters_d1` exercises
  `wfn.kinetic_energy_parameters_d1` (the `tressian_dot` consumer) against a
  numerical derivative — run it after touching tressian/backflow derivative code.
