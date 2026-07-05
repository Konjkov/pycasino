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
- `numba.typed.List`, `numba.typed.Dict`
- Calling other `@njit` functions (inlined automatically)

### Does NOT work ✗

- `list`, `dict`, `set` (Python built-in containers — use `numba.typed.*` or arrays)
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
