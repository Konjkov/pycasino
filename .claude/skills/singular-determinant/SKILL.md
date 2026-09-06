---
name: singular-determinant
description: >
  Use this skill when a multideterminant run dies with `numpy.linalg.LinAlgError: Matrix is
  singular to machine precision` out of `np.linalg.inv` in casino/slater.py — from
  wfn.drift_velocity, wfn.energy, the nodal descriptor, or any other caller of
  slater.gradient / laplacian / hessian / tressian / tressian_dot / state. Covers why an
  individual determinant of an MDET expansion can be exactly singular where Ψ ≠ 0, why a
  single-determinant run can never hit it, the guard that fixes it at no cost and the 10% it
  costs in accuracy at the singular point itself, and why the `Array must not contain infs or
  NaNs` that a debug script sees on the same runs is the deliberate NaN marker of a rejected
  block rather than a second bug. Trigger on: LinAlgError, singular to machine precision,
  np.linalg.inv, MDET crash, gautol, det_coeff, adjugate, NaN in random_walk output.
---

# A singular determinant inside a non-singular wave function

## The defect

`slater.gradient` and its four relatives invert the Slater matrix of **every determinant of the
expansion separately**:

```python
tr_grad_u = (np.linalg.inv(wfn_u[self.permutation_up[i]]) * grad_u[self.permutation_up[i]].T).T.sum(axis=0)
...
c = 1 if single_det else self.det_coeff[i] * np.linalg.det(wfn_u[self.permutation_up[i]]) * np.linalg.det(wfn_d[self.permutation_down[i]])
```

With one determinant that is safe: a singular matrix means Ψ = 0, `log_value` is −inf, and the
Metropolis step to such a configuration is rejected with probability one — the walker can never
stand there. **With several determinants it is not.** `Ψ = Σ c_k det(D_k)` stays finite while one
`D_k` is exactly singular; `log_value` takes `slogdet` and quietly drops that term (`sign = 0`,
`log = −inf`), so the walker settles there like anywhere else, and the next call that needs a
gradient raises.

The fix is one line of arithmetic that is already in the code. At a singular `D_k` the weight
`c = det_coeff[i] * det(u) * det(d)` **is exactly zero**, so the formula already prescribes
`val += 0` and `grad += 0 * tr_grad`: the term contributes nothing. The function crashes only
because `inv` is evaluated before the zero has a chance to multiply it.

## Evidence, on the branch where it was found

`examples/noda_surface/be_c2/`, Be CAS(2,4), nine wave functions differing only in the fixed CSF
coefficient. `nodal_descriptor.py` at 10⁷ steps: **C = 0.00 finished (no MDET block at all, one
determinant), C = 0.01 died on all four ranks.** The band where it happens, one electron pushed
out along a general direction, `examples/nodal_0.01`:

```
r = 20  log|psi| = -21.609  |grad|^2 = 2.76e+02
r = 22  log|psi| = -23.750  LinAlgError
r = 25  log|psi| = -27.342  LinAlgError
r = 26  log|psi| =     -inf  LinAlgError
```

Ψ finite, gradient dead. The mechanism that makes an orbital exactly zero out there is the
gaussian truncation, `slater.py:111` and six more like it:

```python
if alpha * r2 < log_10 * self.gautol:
```

At the default `gautol = 7` every primitive of a shell is dropped once `alpha_min·r² ≥ 16.1`,
which for the most diffuse s exponent of Be/ano-pVDZ is r ≈ 17.5 bohr. Beyond it MO 1 is exactly
0.0 while MO 2 is not, so the determinants that share the vanished orbital go singular one at a
time. Measured directly:

```
r = 17.0  orbitals of the far electron: [ 5.661e-14 -7.751e-07  2.382e-07  4.204e-09 -2.803e-09]
r = 17.5  orbitals of the far electron: [ 0.000e+00 -4.962e-07  1.230e-07  2.109e-09 -1.406e-09]
```

The truncation radius is where it shows up on this system, but **the truncation is not the bug**:
any configuration where one determinant of the expansion happens to be singular does the same, and
raising `gautol` only moves the wall.

## The fix

`np.linalg.det` and `np.linalg.slogdet` do **not** raise on an exactly singular matrix in numba —
verified:

```
det      0.0
slogdet  (0.0, -inf)
```

Only `inv` does. So compute the weight first and skip the determinant when it is zero.

Five call sites, all with the same shape — `casino/slater.py`:

| function | line | what to guard |
|---|---|---|
| `slater_gradient` | 822–825 | `inv` of both channels |
| `slater_laplacian` | 858–866 | `inv_wfn_u`, `inv_wfn_d` |
| `slater_hessian` | 902–911 | `inv_wfn_u`, `inv_wfn_d` |
| `slater_tressian` | 961–987 | `inv_wfn_u`, `inv_wfn_d` |
| `slater_tressian_dot` | 1080–1104 | `inv_wfn_u`, `inv_wfn_d` |

In each: move the `c = ...` line to the top of the loop body, and

```python
if not single_det and c == 0:
    continue
```

**Do not compute `det` in the single-determinant branch.** There `c = 1` and `det` is never
called; adding it would cost an extra `getrf`, ⅔n³ on top of the 2n³ that `inv` already spends,
for a case that cannot occur. Keep the guard inside `if not single_det`.

`slater_state` (line 662) is the sixth site and the different one: it *stores* `inv_u[i]`,
`inv_d[i]` for the electron-by-electron updates, so there is nothing to skip. Zero-fill the pair
whose determinant is singular — `sign_det[i]` is already 0 and `log_det[i]` already −inf, so the
term drops out of `log_sum_exp`, and a zeroed inverse makes `q[i] = 0` in `ratio_1e`, which
`accept_1e` already handles: it returns False and asks the caller for a full recomputation
(`if self.age >= dbar_max_age or np.any(q == 0)`). No new path is needed downstream.

## Cost

None. In the MDET branch `det` is already computed for `c`; the change is the order of two lines,
and in the pathological case it *saves* the `inv`. The single-determinant branch is untouched.
Note in passing that the MDET branch factorizes each matrix twice as it stands — `inv` does
getrf+getri, then `det` does getrf again on the same matrix — which is worth about 25% of the
linear algebra in that loop, but sharing one factorization needs an LU that numba does not expose,
so leave it alone.

## What the guard is not — measured, and worse than it looks

It is not the exact limit, and the error is **O(1), not O(ε)**. `c · tr(D_k⁻¹ ∇D_k)` is
`tr(adj(D_k) ∇D_k) = ∇det(D_k)`, which is finite and in general non-zero at a singular `D_k`: a
determinant that contributes nothing to Ψ still contributes to ∇Ψ. The guard returns zero for it.

Measured on `casino/tests/inputs/MDET/Be`, both down electrons on the z axis, where two of the
four determinants have a zero row and their cofactors are O(1):

```
log|psi| = -7.606          (finite, the wave function is in no trouble)
|grad|^2 = 1290.2          |grad| = 35.9
max |analytic - fd| = 3.49  (~10%)
```

Ψ is smooth there — a vanishing `det(D_k)` is no kink — so the finite difference is right and the
analytic gradient is wrong. At an ordinary position the same call agrees with finite differences
to 2.7e-8.

Two things keep this acceptable rather than a second bug. The set where it happens has measure
zero, so it biases no Monte Carlo average. And in the configurations that actually raise — an
orbital cut to zero by `gautol` at 20 bohr and more — the singular determinant is *dying*, its
cofactors are the surviving orbital values, ~1e-8, so the dropped term is negligible against Ψ
itself. The z-axis configuration above is a far harsher case than anything a walker meets.

Getting it exactly right means accumulating the singular determinant through
`∇Ψ_k = c_k(det_d · tr(adj_u ∇D_u) + det_u · tr(adj_d ∇D_d))` instead of the `c · trace` form the
five functions are built on — a separate path, an adjugate by cofactors in the rare branch, and
for the hessian and the tressian the terms carrying two inverses have to be refactored as well.
Do that only if some use needs the value at such a point; for a crash that would otherwise stop
the run, the guard is the right trade. Say so where it matters instead of quietly claiming the
gradient is exact everywhere.

The regression is `TestMdetSingularDeterminant` in `casino/tests/test_mdet.py`, and it asserts
finiteness only — asserting the finite-difference gradient there would fail by the 10% above.

## Testing

`casino/tests/test_mdet.py` is where the regression belongs. A tracked MDET wave function to build
it on: `examples/gwfn/Be/MP2-CASSCF(2.4)/cc-pVQZ/CBCS/Backflow/`. The test needs a configuration
that makes one determinant exactly singular while Ψ stays finite; the cheap construction is one
electron at r ≈ 22 bohr in a general direction (not along an axis, which zeroes the p orbitals for
a different and uninteresting reason), then assert that `wfn.drift_velocity` and `wfn.energy` are
finite there and that `wfn.log_value` matches its finite-difference gradient. Check the radius
against the basis first — it is set by `gautol` and by the smallest exponent of the file, so it is
not 22 bohr for every example.

Run the whole suite before the merge: the five functions are the hot path of every VMC and DMC
move, and `test_slater.py`, `test_backflow.py`, `test_geminal.py`, `test_vmc.py`, `test_dmc.py`
all go through them.

## Not a second failure mode — the NaN rows of a walk are deliberate

A debug script that walks the sample itself will hit

```
numpy.linalg.LinAlgError: Array must not contain infs or NaNs.
```

within the first few thousand configurations, and it is **not a bug**. `vmc_random_walk` allocates
the sample full of NaN and writes a row only when the block of `decorr_period` sub-steps accepted
something:

```python
position = np.full(shape=(steps,) + self.r_e.shape, fill_value=np.nan)
...
if cond:
    position[i] = self.r_e
```

and `VMC.observable` reads that marker back, `res[i] = res[i - 1]` when `position[i, 0, 0]` is NaN.
A block that accepted nothing repeats the previous configuration's observable, which is what
Metropolis asks for, and it saves the recomputation. The frequency is `(1 - p)^decorr_period` — one
row in 1681 at p ≈ 0.53 and `decorr_period` 10, which is what a debug run measured.

So never call a wave-function method on rows of `random_walk` output directly. Go through
`vmc.observable`, or filter with `np.isnan(position[:, 0, 0])` first. Debug harnesses that forget
this produce a convincing-looking second bug that does not exist.
