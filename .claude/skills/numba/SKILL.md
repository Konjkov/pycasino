---
name: numba
description: >
  Use this skill whenever working with Numba JIT compilation, @nb.njit, @nb.jit,
  parallel=True/False, prange, numba.typed, numba.cuda, numba.vectorize,
  numba.guvectorize, or debugging Numba compilation errors. Trigger on: slow
  numpy loops being accelerated, nopython mode errors, Numba type inference failures,
  reflected list warnings, cache=True, fastmath, nogil, Numba + NumPy interop,
  or any task involving @njit on scientific/numerical code.
---

# Numba: JIT Compilation for Scientific Python

## What Numba does

Numba compiles Python functions to native machine code at call time using LLVM.
It targets numerical code operating on NumPy arrays and Python scalars. The compiled
function runs at C/Fortran speed with no interpreter overhead.

**Key constraint:** only a subset of Python is supported inside JIT-compiled functions.
The supported subset is called *nopython mode* (NPM). When in doubt, always target NPM.

---

## Decorator reference

```python
import numba as nb
import numpy as np
```

### `@nb.njit` — the main decorator

```python
@nb.njit
def f(x, y):
    return x + y
```

Equivalent to `@nb.jit(nopython=True)`. Always prefer `@nb.njit` over `@nb.jit`.
`@nb.jit` without `nopython=True` silently falls back to object mode on failure —
this masks bugs and gives no speedup.

**Common options:**

```python
@nb.njit(
    parallel=False,   # True: enable automatic parallelisation + prange
    cache=True,       # persist compiled bitcode to __pycache__, avoid recompile
    fastmath=True,    # allow reassociation, FMA, no-NaN, no-INF (UNSAFE for QMC)
    nogil=True,       # release the GIL: allows Python threading around nb calls
    boundscheck=False # default False; set True only during debugging
)
def f(x): ...
```

⚠️ **`fastmath=True` is dangerous for QMC/quantum chemistry**: it allows
floating-point reassociation that breaks energy conservation. Use only for
non-critical kernels (geometry, index arithmetic).

### `@nb.njit(parallel=True)` + `nb.prange`

```python
@nb.njit(parallel=True)
def sum_rows(A):
    result = np.zeros(A.shape[0])
    for i in nb.prange(A.shape[0]):   # prange → parallel loop
        for j in range(A.shape[1]):
            result[i] += A[i, j]
    return result
```

`nb.prange` replaces `range` in the outermost loop only. Inner loops stay serial.
Only use `parallel=True` when iterations are independent (no loop-carried dependencies).

In pycasino, **all kernels use `parallel=False`**. Parallelism is achieved at the
MPI level (independent walkers). Do not add `parallel=True` without profiling — the
threading overhead can hurt for small arrays.

### `@nb.vectorize` — ufunc factory

```python
@nb.vectorize(['float64(float64, float64)'], nopython=True)
def clip(x, lo, hi):
    if x < lo: return lo
    if x > hi: return hi
    return x
```

Produces a NumPy ufunc that broadcasts automatically. Signature list is optional but
speeds up dispatch. Without it, Numba infers at first call.

### `@nb.guvectorize` — generalised ufunc

For functions that operate on arrays of fixed rank (e.g., matrix × vector):

```python
@nb.guvectorize(['void(float64[:,:], float64[:], float64[:])'],
                '(m,n),(n)->(m)', nopython=True)
def matvec(A, x, out):
    for i in range(A.shape[0]):
        out[i] = 0.0
        for j in range(A.shape[1]):
            out[i] += A[i, j] * x[j]
```

---

## Supported Python/NumPy inside `@njit`

### Works ✓

- Arithmetic, comparisons, boolean logic
- `if/elif/else`, `for`, `while`, `break`, `continue`, `return`
- Tuples (fixed-length), basic unpacking
- `range()`, `len()`, `abs()`, `min()`, `max()`, `round()`
- NumPy array creation: `np.zeros`, `np.ones`, `np.empty`, `np.full`
- NumPy operations: `np.sum`, `np.dot`, `np.cross`, `np.sqrt`, `np.exp`, `np.log`
- `np.linalg.norm`, `np.linalg.det`, `np.linalg.inv`, `np.linalg.solve`
- `np.linalg.eigh`, `np.linalg.svd` (limited)
- Array indexing, slicing, boolean indexing (basic)
- `math.pi`, `math.e`, `math.sin`, `math.cos`, `math.exp`, `math.sqrt`
- `numba.typed.List`, `numba.typed.Dict`, `numba.typed.Set` (0.66+)
- Calling other `@njit` functions (inlined automatically)

### Does NOT work ✗

- Python `list`/`set` **as arguments** from the interpreter: they are "reflected" types,
  work but raise `NumbaPendingDeprecationWarning` — pass `numba.typed.List`/`numba.typed.Set`
  (0.66+) or arrays. Built locally inside `@njit` (`set()`, `[]`, `{}`), all three work and
  are homogeneous; checked on 0.63.1. No jitted code in `casino/` uses a set
- `print` with f-strings (use `print(x)` with scalars only)
- `try/except`
- Generator expressions, list comprehensions (sometimes work, often don't)
- `isinstance`, `hasattr`, `getattr`
- Classes (except `@nb.experimental.jitclass`)
- `**kwargs` in JIT-compiled functions
- `scipy.*` (not supported inside njit)
- `np.einsum` — **NOT supported** in nopython mode; unroll manually or use dot/matmul

---

## Type system and type inference

Numba infers types from the first call arguments. **The compiled function is specialised
per type signature.** Calling with different dtypes creates multiple compiled versions.

### Explicit signatures (optional, locks types, speeds first call)

```python
@nb.njit('float64[:](float64[:], float64)')
def scale(arr, factor):
    return arr * factor
```

Signature syntax: `'return_type(arg1_type, arg2_type)'`

Common type strings:
| Type string | Python/NumPy meaning |
|---|---|
| `float64` | `np.float64` scalar |
| `float32` | `np.float32` scalar |
| `int64` | `np.int64` scalar |
| `boolean` | `bool` |
| `float64[:]` | 1-D C-contiguous float64 array |
| `float64[:,:]` | 2-D C-contiguous float64 array |
| `float64[::1]` | 1-D array, explicitly C-contiguous |
| `float64[:, ::1]` | 2-D array, C-contiguous (last dim contiguous) |
| `UniTuple(float64, 3)` | tuple of 3 float64 |

### Contiguity matters

Numba generates fastest code for C-contiguous arrays. If you slice a non-contiguous
view, pass `np.ascontiguousarray(x)` before calling the JIT function.

Inside `@njit` the layout is part of the type: `C`, `F` or `A` (any, i.e. strided).
`@` on an `A` operand raises

```
NumbaPerformanceWarning: '@' is faster on contiguous arrays, called on
    (Array(float64, 1, 'A', ...), Array(float64, 1, 'A', ...))
```

and numba then **allocates a C copy of every non-contiguous operand before the BLAS
call** (`make_contiguous`, numba/np/linalg.py). The warning carries no file or line.

#### Which patterns give `A`, and which do not

Verified with `f.nopython_signatures[0].return_type` (numba 0.61, 2026-09):

| expression on a C array | layout |
|---|---|
| `a[i, j]`, `a[i, j, :]`, `a[i, j, :k]` (integers on the leading axes) | `C` |
| `a[:, i]`, `a[:, :, d]`, `a[:, j, d1, d2]` (slice of a **non-last** axis) | `A` |
| `a.T` of a C array | `F` |
| `a3 + a2` (broadcast), any array expression | `C` |
| `np.linalg.inv(a)` | **`F`** |
| a variable unified across branches with different layouts | `A` |

The `inv` one is the trap: an F matrix has **contiguous columns and strided rows**, so
`inv[:, i] @ v` is free while `inv[j] @ v` copies. `.reshape` on an `A` array raises at
runtime, so a reshape in the code proves its input was contiguous.

#### What it costs (this machine, dot of two length-n vectors)

| n | strided `a[:, i] @ b[:, i]` | contiguous `a[i] @ b[i]` | explicit loop |
|---|---|---|---|
| 2 | 122 ns | 31 ns | 3.7 ns |
| 4 | 126 ns | 33 ns | 4.5 ns |
| 16 | 136 ns | 35 ns | 19 ns |
| 64 | 210 ns | 38 ns | 99 ns |

So the copy is ~90 ns per call, and below n ≈ 30 **BLAS itself is the wrong tool**: an
explicit loop beats even the contiguous `@` by 8–30×. Rules of thumb:

- vector dots over `neu`/`ned`-sized things inside nested loops → write the loop;
- matrix products → keep `@`, but hoist one `np.ascontiguousarray` out of the loop
  (or build the array with the sliced axis first) instead of paying a copy per call.

#### Finding the site the warning came from

Numba raises it while typing, and the typing constraint being solved is on the python
stack with its `loc`. Patch `warnings.warn` before the (uncached!) compile:

```python
def patched(message, *args, **kwargs):
    if 'contiguous' in str(message):
        for frame in inspect.stack():
            loc = getattr(frame.frame.f_locals.get('self', None), 'loc', None)
            if loc is not None:
                print(message, loc)   # first few frames give file:line
    return original(message, *args, **kwargs)
```

To recompile an `overload_method` impl on demand without touching the sources:

```python
impl = module.the_overload_generator.py_func(*[None] * nargs)   # returns the closure
nb.njit(cache=False)(impl)(struct_ref_proxy, *args)             # fresh compile, warns
```

An A/B of a fixed copy of the same impl (paste it into a scratch file, edit, `njit`)
is then a real measurement rather than a guess.

#### Warnings from branches that never run

A structref field typed `nb.optional(X_t)` makes numba compile **both** sides of
`if self.x is not None:`, so a wave function with no geminal still compiles — and warns
about — every `@` inside geminal.py. Those warnings cost nothing at run time; check
whether the site is even reachable before optimising it.

#### Measured payoff in this project (2026-09-08, do not redo)

The four warnings out of `wfn.kinetic_energy_parameters_d1` are all in `geminal.py`:
`pool_gradient` (`pool_grad_u[:, :, d].T @ full_c`, `a @ pool_grad_d[:, :, d]`) and
`pool_tressian_dot` (`inv[j] @ hd[:, j, d1, d2]`, `inv @ hd[:, j, d1, d2]`). Fixing all
four on Ne (neu = ned = norb = 5, one geminal):

| | original | fixed |
|---|---|---|
| `pool_gradient` | 10.98 µs | 10.54 µs (−4%) |
| `pool_tressian_dot` | 175.2 µs | 167.5 µs (−4.5%) |

against `geminal.gradient` 22.2 µs, `geminal.tressian_dot` 242 µs, `wfn.energy` 42 µs —
so **≈1% of a geminal VMC/DMC step, ≈3% of a geminal+backflow emin step, 0% of any run
without a geminal**. Worth doing as hygiene and to keep the log clean, not as a speedup.

---

## Debugging Numba errors

### Step 1: read the type error

```
numba.core.errors.TypingError: Failed in nopython mode pipeline (step: nopython frontend)
  - argument 0: cannot determine Numba type of <class 'list'>
```

→ A Python `list` was passed. Replace with `np.array(...)` or `numba.typed.List`.

### Step 2: use `nb.njit` with `cache=False` and inspect

```python
@nb.njit(cache=False)
def f(x): ...

f.inspect_types()        # shows inferred types for each variable
f.inspect_llvm()         # shows generated LLVM IR
f.inspect_asm()          # shows native assembly
```

### Step 3: common mistakes

**Reflected list warning:**
```
NumbaTypeSafetyWarning: unsafe cast from int64 to int32 (reflected list)
```
→ Numba is boxing/unboxing a Python list at the boundary. Move the list inside the
function or convert to `np.array` before calling.

**Object mode fallback (silent, `@jit` only):**
```
NumbaWarning: Function "f" was compiled in object mode without forceobj=True
```
→ Switch to `@nb.njit` to make this a hard error. Find and fix the unsupported construct.

**`cannot unify float64 and int64`:**
Numba sees two branches returning different types. Make types consistent:
```python
# Bad
if cond:
    return 0       # int
else:
    return 0.0     # float
# Good
    return 0.0
    return 0.0
```

---

## Memory and array patterns

### Output arrays: allocate outside, pass in

```python
@nb.njit
def fill(out, x):
    for i in range(out.shape[0]):
        out[i] = x * i

out = np.empty(N, dtype=np.float64)
fill(out, 3.14)
```

Avoids allocation inside hot loops.

### Returning arrays from `@njit`

Numba can return arrays allocated inside the function:

```python
@nb.njit
def make(n):
    out = np.empty(n, dtype=np.float64)
    ...
    return out
```

This is fine. The array is heap-allocated by Numba and ownership is transferred to Python.

### Slices and views

Array slices inside `@njit` are views (no copy). Safe to pass slices as arguments:

```python
@nb.njit
def process(row):   # row is a 1-D view
    ...

for i in range(A.shape[0]):
    process(A[i, :])   # no copy, passes a view
```

### Array expressions allocate — write the loop instead

An expression over arrays builds a temporary for every operation and then copies it into the
destination. Numba fuses element-wise expressions, but each *statement* still allocates, and an
in-place update on a strided slice allocates too: `a[:, e] += v` is `a[:, e] = a[:, e] + v`, so
it allocates the right-hand side and writes it back through the stride.

For small arrays the allocation is the whole cost. Measured in `vmc.shift_coordinates` on
krypton, three such statements over 36x36x3 and 1x36x3 arrays cost 1.25 us per call for about
250 flops of arithmetic — a third of the whole electron-by-electron sweep, which the scalar
loops took from 263 us to 150. Write the loops, they allocate nothing:

```python
# Bad — three temporaries per call, two of them strided
self.n_vectors[:, e] += shift
self.e_vectors[e] += shift
self.e_vectors[:, e] -= shift

# Good — no allocation
for atom in range(self.n_vectors.shape[0]):
    for i in range(3):
        self.n_vectors[atom, e, i] += shift[i]
for j in range(self.e_vectors.shape[0]):
    for i in range(3):
        self.e_vectors[e, j, i] += shift[i]
        self.e_vectors[j, e, i] -= shift[i]
```

The rule is not "never use array expressions": on large contiguous arrays the allocation is
amortised and the vectorised form wins. It is "in a function called once per proposed move,
count the statements, because each one is a `malloc`".

The obvious escape, writing into a destination array, has a catch: **Numba rejects `out=` as a
keyword argument.** The destination has to go in positionally. Checked under numba 0.63.1 and
numpy 2.2.6:

```python
np.add(a, b, out=c)    # TypingError: unsupported keyword arguments
np.multiply(a, b, c)   # works
np.sqrt(a, c)          # works
np.dot(a, b, c)        # works
```

So a ufunc can still write in place without a temporary, but only in the positional form, and
only where a ufunc covers what is wanted — a strided compound update like `a[:, e] += v` is not
one of them, and there the loop is the answer.

---

## Compilation cache

```python
@nb.njit(cache=True)
def f(x): ...
```

Compiled bitcode is stored in `__pycache__/`. On the next run Numba skips compilation
if the source has not changed. Essential for scripts that import many JIT functions.

**Invalidation:** cache is invalidated when the function source changes, the Numba
version changes, or the NumPy version changes. Delete `__pycache__` manually if
you see stale-cache bugs.

**It does not follow dependencies across modules, and this silently invalidates benchmarks.**
An `@overload_method` body is inlined into whatever `cache=True` function calls it, and only
that caller's own source is hashed. Edit the body of an overload in `casino/vmc.py`, re-run a
benchmark whose `@nb.njit(cache=True)` wrapper lives in another file, and you measure the old
code with no warning of any kind — the numbers simply do not move. This has bitten twice. A
change to a structref's *field list* does force recompilation, because the type and hence the
signature changes; a change to a *body* does not.

Before trusting any timing after an edit, either delete the `__pycache__` of every module in
the call chain, or run the benchmark under its own cache directory:

```
NUMBA_CACHE_DIR=/tmp/nbcache python3 bench.py
```

The second form is what to use while a long production run is in progress, since deleting the
project cache under a running process is not safe.

---

## Numba + pycasino patterns

### The `parallel=False` convention

All pycasino kernels use `parallel=False`. This is intentional: each MPI rank
occupies one core, and thread-level parallelism would conflict. Do not change this
without benchmarking.

### Calling `@njit` from Python (warm-up)

First call triggers compilation (can take seconds). For benchmarking, always call
once with representative input before timing:

```python
# Warm up
f(np.zeros(10, dtype=np.float64))
# Now benchmark
t0 = time.perf_counter()
f(real_input)
print(time.perf_counter() - t0)
```

Or use `nb.njit(cache=True)` to amortise the cost across runs.

### Passing structured data

Numba does not support Python dataclasses or arbitrary objects. The pycasino pattern
is to extract arrays from objects before calling JIT kernels:

```python
# Python side (outside njit)
def value(self, r_e):
    return _value_jit(r_e, self.coefficients, self.exponents, self.shell_map)

@nb.njit
def _value_jit(r_e, coefficients, exponents, shell_map):
    ...
```

This keeps class logic in Python and numerical hot loops in Numba.

### `np.linalg` inside `@njit`

Supported functions: `det`, `slogdet`, `inv`, `solve`, `norm`, `eigh`, `cholesky`.
Not supported: `lstsq`, `matrix_rank`, `pinv`.

For pinv, implement via SVD:
```python
@nb.njit
def pinv(A):
    U, s, Vt = np.linalg.svd(A)
    tol = 1e-12 * s[0]
    s_inv = np.where(s > tol, 1.0 / s, 0.0)
    return (Vt.T * s_inv) @ U.T
```

---

## Performance checklist

1. **Profile first** — use `cProfile` or `line_profiler` to confirm Numba functions
   are the bottleneck before optimising.
2. **Check dtypes** — `float32` is 2× faster on GPU/SIMD but loses precision.
   For QMC always use `float64`.
3. **Avoid Python objects inside hot loops** — every Python object lookup breaks
   the JIT. Keep everything as scalars or numpy arrays.
4. **Prefer 1-D loops over numpy broadcasting** inside `@njit` — Numba vectorises
   explicit loops well; broadcasting sometimes defeats optimisation.
5. **Use `nb.prange` only at the outermost loop** — nesting `prange` inside `prange`
   is not supported.
6. **`cache=True` for all stable functions** — eliminates recompilation overhead in
   production runs.
7. **Avoid `np.einsum`** — unroll with explicit loops or use `@` (matmul operator).

---

## Useful environment variables

```bash
NUMBA_NUM_THREADS=4      # number of threads for parallel=True kernels
NUMBA_CACHE_DIR=/tmp/nb  # redirect cache away from source tree
NUMBA_DISABLE_JIT=1      # disable all JIT (pure Python fallback, for debugging)
NUMBA_DEBUG_TYPEINFER=1  # verbose type inference log
NUMBA_DUMP_IR=1          # dump LLVM IR for compiled functions
```

`NUMBA_DISABLE_JIT=1` is invaluable for debugging: all `@njit` functions run as
plain Python, so `pdb`, `print`, and `traceback` work normally.

---

## Upstream: release notes and the issues this project is waiting on

Check these before blaming our own code for a regression after a Numba bump, and
before designing around a limitation that may already have been lifted:

- Release notes overview (all versions): <https://numba.readthedocs.io/en/stable/release-notes-overview.html>
- Current release notes: <https://numba.readthedocs.io/en/stable/release-notes.html>

Open Numba issues the user is tracking for pycasino:

| Issue | Subject | Why it matters here |
|---|---|---|
| [#5149](https://github.com/numba/numba/issues/5149) | access to `np.array` data / `tobytes` etc. | raw buffer access from inside `@njit`; `ndarray.tobytes()` itself landed in 0.62 |
| [#6972](https://github.com/numba/numba/issues/6972) | wrapper or type to avoid inlining | no supported way to force a call boundary; matters where inlining blows up compile time or defeats a shared kernel |
| [#9776](https://github.com/numba/numba/issues/9776) | parallelisation approach | worth trying as an alternative to the current `prange`/MPI split |
| [#9712](https://github.com/numba/numba/issues/9712) | allocations in Numba significantly slower than in NumPy | directly relevant — the hot kernels allocate per call (harmonics buffers, `ee_powers`/`en_powers`, the `_d1` result arrays). Reinforces the "Memory and array patterns" rule above: preallocate into struct fields and write in place rather than returning fresh arrays |

These are also listed at the top of `backlog.txt`.

### 0.62 → 0.67 and what it means for this code (audit 2026-09-11)

`pyproject.toml` now requires `numba>=0.66.0` and `requires-python >=3.10` (numba dropped 3.9
in 0.61 already). What each release added:

| version | additions | removals / breaks |
|---|---|---|
| 0.62 | `ufunc.reduceat`, `ndarray.tobytes()`, `np.frombuffer(offset=, count=)`, `np.nan_to_num(posinf=, neginf=)`, `is` on structref, `NUMBA_CACHE_LOCATOR_CLASSES`, LLVM 20 + New Pass Manager | **SVML dropped** (no vectorised Intel `exp`/`log`); failed overload resolutions are cached per compile session (`NUMBA_DISABLE_TYPEINFER_FAIL_CACHE` to undo) |
| 0.63 | Python 3.14, experimental free-threading, `math.exp2`, `np.unique` on non-numeric, `np.copy` on lists/scalars | gufunc refcount leak fixed |
| 0.64 | NumPy 2.4, `np.moveaxis`, scalar `np.all`/`np.any` | `np.trapz`, `np.in1d` gone with NumPy 2.4 |
| 0.65 | Python 3.14t, `a[None]`, scalar `np.min`/`max`/`mean`/`prod` | |
| 0.66 | **`numba.typed.Set`**, **full NumPy fancy indexing** (several index arrays, multidimensional indices, `np.newaxis`), type annotations on `jit`/`njit`, **faster compile** of functions with many single-assigned variables, LLVM 22 | |
| 0.67 | NumPy 2.5, `np.insert`, `ddof` in `np.nanvar`/`np.nanstd`, runtime `axis` in `np.sum`/`np.cumsum`, faster compile (liveness order) | `np.row_stack`, 2-D `np.cross` |

The installed 0.63.1 already has `SVML Operational: False` and LLVM 20, so the bump does not
change run-time speed; the only payoff is cold-compile time on the big jastrow/backflow/gjastrow
kernels (measure with the cache cleared, see "Compilation cache").

**Nothing in `casino/` needed changing.** Every workaround found is still required under 0.67:

| site | why it stays |
|---|---|
| `overload.py` `@overload(np.repeat)`, used by DMC branching on `(nwalk, ne, 3)` arrays | built-in `np.repeat` still has no `axis` and flattens |
| `overload.py` `polyval2d`/`polyval3d` | only `polyval` is supported |
| `wfn.py` `wfn_type` — one class for both jastrow kinds | still no union types |
| `gjastrow.py` — one array per rank | still no runtime ndim |
| `jastrow.py`/`gjastrow.py` hand-written `r_eI` norm | `@`/`dot` on a strided column still copies |
| `jastrow.py` "do not create temporary 1-d numpy array" loops, `backflow.py` explicit loops | array expressions still allocate |
| `geminal.py` `ascontiguousarray` before the pool products | contiguity rule unchanged |

New features checked and found to have no taker: fancy indexing (every gather in the code is a
single index array, e.g. `wfn_u[self.permutation_up[i]]`, which always worked; and the docs warn
that multi-array fancy indexing "can be slower than expected" — do not put it in a hot kernel);
`a[:, None]` instead of `np.expand_dims` (cosmetic only); `typed.Set` (every `set()` is in
pure-Python readers); scalar reductions (no `np.max(np.array([a, b]))` workarounds exist);
`np.moveaxis` and `nan_to_num(posinf=)` in `cusp.py` (plain Python class, not jitted). Runtime
`axis` and `np.insert` need 0.67, above the current floor.
