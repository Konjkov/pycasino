.. _vmc:

Variational Monte Carlo
=======================

Variational Monte Carlo is implemented in the :class:`casino.vmc.VMC` class. It samples the
probability density :math:`|\Psi(\mathbf{R})|^2` of a trial wave function with the Metropolis
algorithm and estimates the expectation value of the Hamiltonian as the mean of the local
energy over the resulting chain. The Hamiltonian enters only through that average: the walk
itself is driven by :math:`|\Psi|^2` alone.

Three sampling modes are implemented, selected by ``vmc_method``:

- **EBES** (``vmc_method : 1``) — electrons are moved one at a time, each with its own
  accept/reject step;
- **DBDS** (``vmc_method : 2``) — determinant-by-determinant sampling;
- **CBCS** (``vmc_method : 3``) — all electrons are moved at once, single accept/reject.

The single free parameter of the walk is the step size ``dtvmc``. It is too small when
successive configurations are strongly correlated and too large when almost every move is
rejected, and the efficiency of the walk is flat enough between those extremes that any
acceptance from roughly 30 % to 70 % costs little [23]_; ``opt_dtvmc : T`` tunes it to 50 % in
:ref:`optimize_vmc_step <vmc-optimize-step>`, starting from the closed-form estimate in
:ref:`approximate_step_size <vmc-approximate-step-size>`. The rest of this page derives that
estimate, because the derivation also fixes what the optimizer can and cannot be expected
to find, and where the remaining empirical constant comes from.

.. _vmc-notation:

Notation
--------

Write :math:`\Lambda(\mathbf{R}) = \ln|\Psi(\mathbf{R})|` for the logarithm of the trial
function, :math:`\mathbf{F} = \nabla\Lambda` for the drift vector and
:math:`\mathsf{H} = \nabla\nabla\Lambda` for its Hessian, all in the full
:math:`3N_e`-dimensional configuration space.

Two averages appear below and must not be confused:

.. list-table::
   :widths: 25 40 35
   :header-rows: 1
   :width: 100%

   * - Symbol
     - Averaged over
     - Held fixed
   * - :math:`\mathbb{E}_\Delta[\,\cdot\,|\,\mathbf{R}]`
     - the proposed displacement :math:`\boldsymbol{\Delta}`
     - the configuration :math:`\mathbf{R}`, so :math:`\Lambda`, :math:`\mathbf{F}`,
       :math:`\mathsf{H}` are constants
   * - :math:`\mathbb{E}_\mathbf{R}[\,\cdot\,]`
     - the chain, with density :math:`|\Psi|^2 / \int |\Psi|^2`
     - nothing

The full average is :math:`\mathbb{E} = \mathbb{E}_\mathbf{R}\,\mathbb{E}_\Delta`. The order
matters: the acceptance probability is a non-linear function of the configuration, so the two
stages do not commute, and the whole of
:ref:`the last section <vmc-cost-of-interchange>` is about the price of interchanging them.

Three local kinetic energies are used. They are functions of :math:`\mathbf{R}`, hence
constants with respect to :math:`\mathbb{E}_\Delta`:

.. math::

    T_L(\mathbf{R}) = -\frac{1}{2}\frac{\nabla^2\Psi}{\Psi}, \qquad
    T_D(\mathbf{R}) = \frac{1}{2}\,\mathbf{F}\cdot\mathbf{F}
                    = \frac{1}{2}\sum_i |\nabla_i\Lambda|^2, \qquad
    T_M(\mathbf{R}) = \frac{1}{2}\left(T_L + T_D\right)

The identity :math:`\nabla^2\Psi/\Psi = \nabla^2\Lambda + |\nabla\Lambda|^2` ties them to the
Laplacian of the logarithm,

.. math::

    \nabla^2\Lambda = -2\,(T_L + T_D) = -4\,T_M

These are the three kinetic-energy estimators Casino reports: :math:`T_D` is ``FISQ``,
:math:`T_M` is ``TI``, and :math:`T_L = 2\,T_M - T_D` is ``KEI``, the one entering the total
energy. In Pycasino :math:`T_L` is what :ref:`wfn_kinetic_energy <kinetic_energy>` returns,
and :math:`T_D` is computed inside it as :math:`\mathbf{F}\cdot\mathbf{F}/2`.

The local energy :math:`E_L = T_L + V` plays no part in what follows, and neither does the
potential. The Metropolis criterion contains :math:`|\Psi|^2` only. The kinetic energies
appear as combinations of derivatives of :math:`\Lambda`, which is an identity, not physics;
:math:`V` is not differentiated anywhere.

Finally, :math:`\Phi` and :math:`\varphi` denote the standard normal distribution function
and density, and

.. math::

    s \equiv \Phi^{-1}(3/4) = 0.6744898, \qquad \varphi(s) = 0.3177766

.. _vmc-what-is-accepted:

What is accepted
----------------

CBCS displaces all electrons at once, :math:`\mathbf{R}' = \mathbf{R} + \boldsymbol{\Delta}`,
where the :math:`3N_e` components :math:`\delta_k` are independent and uniform on
:math:`[-\mathrm{dtvmc}, \mathrm{dtvmc}]`. The proposal is symmetric, so the Metropolis ratio
reduces to the density ratio alone:

.. math::

    a(\mathbf{R} \to \mathbf{R}') = \min(1, e^X), \qquad
    X = \ln\frac{|\Psi(\mathbf{R}')|^2}{|\Psi(\mathbf{R})|^2}
      = 2\left[\Lambda(\mathbf{R}') - \Lambda(\mathbf{R})\right]

Everything reduces to the statistics of the single scalar :math:`X`. The mean acceptance is a
two-stage average, in the order the algorithm performs it — the walker sits at
:math:`\mathbf{R}` and draws :math:`\boldsymbol{\Delta}`, and only then is :math:`\mathbf{R}`
itself distributed according to :math:`|\Psi|^2`:

.. math::

    A(\mathrm{dtvmc}) = \mathbb{E}_\mathbf{R}\left[\alpha(\mathbf{R})\right], \qquad
    \alpha(\mathbf{R}) = \mathbb{E}_\Delta\left[\min(1, e^X)\,\middle|\,\mathbf{R}\right]

.. _vmc-stage-one:

Stage I: averaging over the proposal
------------------------------------

Expanding to second order in the displacement,

.. math::

    X = 2\,\boldsymbol{\Delta}\cdot\mathbf{F}
      + \boldsymbol{\Delta}^{\mathsf T}\mathsf{H}\,\boldsymbol{\Delta}
      + O(\Delta^3)

The moments of the uniform cube are
:math:`\mathbb{E}_\Delta[\delta_k] = 0`,
:math:`\mathbb{E}_\Delta[\delta_k \delta_l] = (\mathrm{dtvmc}^2/3)\,\delta_{kl}` and
:math:`\mathbb{E}_\Delta[\delta_k \delta_l \delta_m] = 0`. Since :math:`\mathbf{F}` and
:math:`\mathsf{H}` are fixed by :math:`\mathbf{R}`, they come out of the average as constants:

.. math::

    \mathbb{E}_\Delta\left[2\,\boldsymbol{\Delta}\cdot\mathbf{F}\,\middle|\,\mathbf{R}\right]
    = 2\sum_k F_k\,\mathbb{E}_\Delta[\delta_k] = 0

The linear term dies from the symmetry of the proposal, not from any property of the drift.
The quadratic term survives as a trace,

.. math::

    \mathbb{E}_\Delta\left[\boldsymbol{\Delta}^{\mathsf T}\mathsf{H}\boldsymbol{\Delta}
    \,\middle|\,\mathbf{R}\right]
    = \sum_{kl} \mathsf{H}_{kl}\,\mathbb{E}_\Delta[\delta_k \delta_l]
    = \frac{\mathrm{dtvmc}^2}{3}\,\mathrm{Tr}\,\mathsf{H}
    = \frac{\mathrm{dtvmc}^2}{3}\,\nabla^2\Lambda

and in the variance the cross term vanishes because it involves only third moments of a
symmetric distribution:

.. math::

    \mathrm{Var}_\Delta[X|\mathbf{R}]
    = \mathrm{Var}_\Delta\!\left(2\boldsymbol{\Delta}\cdot\mathbf{F}\right)
    + \underbrace{\mathrm{Var}_\Delta\!\left(\boldsymbol{\Delta}^{\mathsf T}\mathsf{H}
      \boldsymbol{\Delta}\right)}_{O(\mathrm{dtvmc}^4)}
    + \underbrace{2\,\mathrm{Cov}_\Delta}_{0}
    = \frac{4\,\mathrm{dtvmc}^2}{3}\,\mathbf{F}\cdot\mathbf{F} + O(\mathrm{dtvmc}^4)

The mean is therefore of second order in the step size while the variance survives at first
non-vanishing order, and both are kinetic energies:

.. math::

    m(\mathbf{R}) \equiv \mathbb{E}_\Delta[X|\mathbf{R}]
    = -\frac{4\,\mathrm{dtvmc}^2}{3}\,T_M(\mathbf{R}), \qquad
    s^2(\mathbf{R}) \equiv \mathrm{Var}_\Delta[X|\mathbf{R}]
    = \frac{8\,\mathrm{dtvmc}^2}{3}\,T_D(\mathbf{R})

The asymmetry is essential: the mean sees the combination :math:`T_L + T_D`, the variance sees
:math:`T_D` alone, and the Laplacian form does not enter the variance at all.

At fixed :math:`\mathbf{R}` the leading part of :math:`X` is a sum of :math:`3N_e` independent
terms and is normal up to :math:`O(1/n_\mathrm{eff})`, where
:math:`n_\mathrm{eff} = (\sum_k F_k^2)^2 / \sum_k F_k^4` is the participation ratio of the
drift components. With that,

.. math::

    \alpha(\mathbf{R}) = g\big(m(\mathbf{R}), s(\mathbf{R})\big), \qquad
    g(m, s) = \Phi\!\left(\frac{m}{s}\right)
            + e^{m + s^2/2}\,\Phi\!\left(-s - \frac{m}{s}\right)

which is the standard integral
:math:`\mathbb{E}[\min(1,e^X)] = P(X \ge 0) + \int_{-\infty}^{0} e^x\,\mathcal{N}(x;m,s^2)\,dx`.

Note that pointwise

.. math::

    m(\mathbf{R}) + \tfrac{1}{2}s^2(\mathbf{R})
    = \frac{2\,\mathrm{dtvmc}^2}{3}\left[T_D(\mathbf{R}) - T_L(\mathbf{R})\right] \neq 0

so :math:`g` cannot be simplified at this stage. It becomes simplifiable only after stage II.

.. _vmc-stage-two:

Stage II: averaging along the chain
-----------------------------------

Integrating by parts against :math:`|\Psi|^2`, with the boundary term vanishing for a bound
state,

.. math::

    \mathbb{E}_\mathbf{R}[T_D] = \frac{1}{2}\int |\nabla\Psi|^2
                               = -\frac{1}{2}\int \Psi \nabla^2\Psi
                               = \mathbb{E}_\mathbf{R}[T_L]

so the three estimators are unbiased for the same quantity and differ only in variance:

.. math::

    \mathbb{E}_\mathbf{R}[T_L] = \mathbb{E}_\mathbf{R}[T_D]
    = \mathbb{E}_\mathbf{R}[T_M] \equiv \langle T \rangle

By the laws of total expectation and total variance,

.. math::

    \mu \equiv \mathbb{E}[X] = \mathbb{E}_\mathbf{R}[m]
    = -\frac{4}{3}\,\mathrm{dtvmc}^2 \langle T \rangle, \qquad
    \sigma^2 \equiv \mathrm{Var}[X]
    = \mathbb{E}_\mathbf{R}[s^2] + \mathrm{Var}_\mathbf{R}[m]
    = \frac{8}{3}\,\mathrm{dtvmc}^2 \langle T \rangle + O(\mathrm{dtvmc}^4)

the discarded piece being
:math:`(16\,\mathrm{dtvmc}^4/9)\,\mathrm{Var}_\mathbf{R}(T_M)`. Hence
:math:`\mu = -\sigma^2/2`, which is not an assumption but a consequence: stationarity of the
chain requires :math:`\mathbb{E}[e^X] = 1`, that is :math:`\mu + \sigma^2/2 = 0` for normal
:math:`X`, and the expansion reproduces it by itself. It appears only after stage II — stage I
does not give it.

The second relation is a sum rule:

.. math::

    \mathrm{Var}[X] = \frac{8}{3}\,\mathrm{dtvmc}^2 \langle T \rangle

and nothing else enters. Neither the geometry of the molecule, nor the distribution of nuclear
charge, nor the Jastrow factor and backflow appear other than through the single scalar
:math:`\langle T \rangle`. The reason is that the sum rule is a full trace,

.. math::

    \mathbb{E}_\mathbf{R}\left[\sum_i |\nabla_i \Lambda|^2\right]
    = \mathrm{Tr}\,\mathbb{E}_\mathbf{R}\left[\nabla_\alpha \Lambda \nabla_\beta \Lambda\right]
    = 2 \langle T \rangle

and an isotropic step sees the trace only. Every multipole of the density cancels identically,
so the law is blind to shape: linear, planar and cage-like molecules of the same composition
obey it alike. Anisotropy would enter only through a step that treats the Cartesian directions
differently and so probes the traceless part of
:math:`Q_{\alpha\beta} = \mathbb{E}_\mathbf{R}[\sum_i \nabla_\alpha \Lambda \nabla_\beta \Lambda]`;
no such step is implemented.

.. _vmc-interchange:

Interchanging the two stages
----------------------------

The exact acceptance is
:math:`A = \mathbb{E}_\mathbf{R}[g(m(\mathbf{R}), s(\mathbf{R}))]`. Replacing
:math:`T_D(\mathbf{R})` and :math:`T_M(\mathbf{R})` by their common mean
:math:`\langle T \rangle` *before* applying :math:`g` — that is, commuting
:math:`\mathbb{E}_\mathbf{R}` with a non-linear function — sends :math:`m \to -\sigma^2/2` and
:math:`s \to \sigma`, so the exponential becomes unity and both arguments collapse to
:math:`-\sigma/2`:

.. math::

    A(\sigma) \approx 2\,\Phi(-\sigma/2)

a one-parameter curve: acceptance depends on the step size through :math:`\sigma` only.
Setting :math:`A = 1/2` gives :math:`\sigma = 2s = 1.3489795` and, with the sum rule,

.. math::

    \mathrm{dtvmc}_{50}\,\sqrt{\langle T \rangle}
    = s\sqrt{3/2} = \sqrt{3}\,\mathrm{erfinv}(1/2) = 0.8260784

For a general target :math:`A` the constant is
:math:`\sqrt{3}\,\mathrm{erfinv}(1 - A)`; at the Roberts–Gelman–Gilks optimal acceptance
0.234 [22]_ it would be 1.4576. There is nothing fundamental about 0.826: it is fixed by three
conventions — a uniform cube rather than a Gaussian proposal, a 50 % target, and atomic
units. Nor is there any reason for it to equal one, the prefactor being dimensional,
:math:`[\mathrm{dtvmc}] = \sqrt{1/\mathrm{energy}}`.

.. _vmc-approximate-step-size:

approximate_step_size
---------------------

To use the result as an initial guess, :math:`\langle T \rangle` is needed before any sampling
has been done. For a neutral system the virial theorem gives
:math:`\langle T \rangle = |E|`, and the Thomas–Fermi expansion with its Scott and Dirac
corrections [24]_ [25]_ supplies the energy with no fitted constant at all:

.. math::

    \langle T \rangle \approx \sum_a \left[
        0.7687\,Z_a^{7/3} - 0.5\,Z_a^2 + 0.2699\,Z_a^{5/3} \right]

which reproduces Hartree–Fock energies to 1.8 % rms. The leading term alone is much worse than
it looks — 4.2 % rms, with the effective coefficient
:math:`|E| / \sum_a Z_a^{7/3}` drifting from 0.54 to 0.64 across the periodic table — and that
drift is exactly the Scott term.

The number of electrons deliberately does not appear: it equals :math:`\sum_a Z_a` only for a
neutral system, and substituting it costs 27 % on :math:`\mathrm{Be}^{2+}`. The remaining
correction is written in terms of the participation ratio of the Thomas–Fermi weight,

.. math::

    n_\mathrm{nuc} = \frac{\left(\sum_a Z_a^{7/3}\right)^2}{\sum_a Z_a^{14/3}}, \qquad
    \mathrm{dtvmc} = \sqrt{3}\,\mathrm{erfinv}(1/2)\,
    \frac{1 + 0.045 / n_\mathrm{nuc}}{\sqrt{\langle T \rangle}}

which is :math:`1` for a single heavy atom and the number of equivalent nuclei for a symmetric
molecule, and never divides by zero for a hydrogen-only system. Measured on the systems in
``examples/time_step/CBCS`` the 50 % acceptance point lands within 0.9 % rms of this guess for
atoms, ions, hydrides and hydrocarbons alike.

Only the last constant is fitted, and the next section explains what it stands for.

.. _vmc-cost-of-interchange:

Cost of the interchange
-----------------------

The interchange above is the only uncontrolled step of the derivation. Its sign follows
without any calculation: :math:`g` is convex in :math:`T_D`, so by Jensen's inequality the
spread of :math:`T_D` *raises* the acceptance and a larger step is needed. The correction is
positive by construction and is exactly the Jensen gap of commuting
:math:`\mathbb{E}_\mathbf{R}` with :math:`g`.

In terms of the dimensionless fluctuations
:math:`u = T_D/\langle T\rangle - 1` and :math:`v = T_L/\langle T\rangle - 1`, expanding the
exact acceptance to second order and solving :math:`A = 1/2` gives

.. math::

    \frac{\delta\,\mathrm{dtvmc}}{\mathrm{dtvmc}}
    = c_{DD}\,\mathbb{E}_\mathbf{R}[u^2]
    + c_{DL}\,\mathbb{E}_\mathbf{R}[uv]
    + c_{LL}\,\mathbb{E}_\mathbf{R}[v^2]

.. math::

    c_{DD} = \frac{1}{8} - \frac{s^2}{4} + \frac{s^3}{16\varphi(s)} = +0.071617, \qquad
    c_{DL} = \frac{s^2}{2} - \frac{s^3}{8\varphi(s)} = +0.106766, \qquad
    c_{LL} = -\frac{s^2}{8} + \frac{s^3}{16\varphi(s)} = +0.003484

with :math:`c_{DD} + c_{DL} + c_{LL} = (1 + s^2)/8`,
:math:`c_{DD} - c_{LL} = (1 - s^2)/8` and :math:`c_{DL} + 2c_{LL} = s^2/4`; the terms in
:math:`s^3/\varphi` cancel in the sum. Treating :math:`\nabla^2\Lambda` as non-fluctuating,
that is setting :math:`T_L(\mathbf{R}) = T_D(\mathbf{R})` pointwise, collapses this to a single
term,

.. math::

    \frac{\delta\,\mathrm{dtvmc}}{\mathrm{dtvmc}}
    = \frac{1 + s^2}{8}\,\frac{\mathrm{Var}_\mathbf{R}(T_D)}{\langle T \rangle^2}
    = 0.18187\,\mathrm{CV}^2(T_D)

and this is the form behind the fitted :math:`1 + 0.045/n_\mathrm{nuc}`: :math:`T_D` is
dominated by whichever electron is currently near a nucleus, since :math:`|\nabla\Lambda| \sim Z`
there, so :math:`n` equivalent nuclei contribute :math:`n` independent terms and
:math:`\mathrm{CV}^2(T_D)` falls as :math:`1/n`.

This one-constant form should not be trusted quantitatively. Rewritten in the variables Casino
prints, :math:`t = T_M/\langle T\rangle - 1 = (u + v)/2`, the same expression reads

.. math::

    \frac{\delta\,\mathrm{dtvmc}}{\mathrm{dtvmc}}
    = -0.03167\,\mathbb{E}_\mathbf{R}[u^2]
    + 0.19959\,\mathbb{E}_\mathbf{R}[ut]
    + 0.01395\,\mathbb{E}_\mathbf{R}[t^2]

The coefficient of :math:`\mathrm{Var}(T_D)` has turned negative and the covariance dominates,
because :math:`T_M = -\nabla^2\Lambda/4` diverges as :math:`Z/2r` at a nucleus — from
:math:`\Lambda \approx -Zr` follows :math:`\nabla^2\Lambda = -2Z/r` — whereas :math:`T_D` stays
bounded there, :math:`|\nabla\Lambda| \to Z`. The variances of the estimators are not
comparable: on helium the two error bars differ by an order of magnitude, so
:math:`\mathrm{Var}(T_M)/\mathrm{Var}(T_D)` is of order 70 and :math:`\mathrm{CV}(T_L)` is of
order unity, where a second-order expansion in :math:`v` no longer applies. A single fitted
constant absorbs :math:`\mathbb{E}_\mathbf{R}[uv]`, :math:`\mathbb{E}_\mathbf{R}[v^2]` and the
truncation of the series alike.

The interchange is avoidable in principle. The exact acceptance is the average of a known
function of two quantities that are already evaluated at every configuration, so accumulating
:math:`T_M` and :math:`T_D` over the equilibration walk — which runs at :math:`|\Psi|^2`
*before* the step is optimized — and solving :math:`A(\mathrm{dtvmc}) = 1/2` numerically would
remove the Thomas–Fermi estimate, the virial theorem and the fitted constant together, and
would account for the Jastrow factor, backflow, ions and pseudopotentials for free, since the
function actually being sampled is the one being measured.

.. _vmc-validity:

Range of validity
-----------------

Of stage I:

1. the expansion of :math:`X` is truncated at second order in
   :math:`\boldsymbol{\Delta}`, and at the 50 % point
   :math:`\mathrm{dtvmc}^2 \langle T\rangle \approx 0.68`, so the step is not small;
2. :math:`\Psi` is not analytic at a nuclear cusp, which is precisely where :math:`T_L`
   diverges, so the Taylor expansion formally fails there;
3. normality of the leading term rests on a central limit theorem over :math:`3N_e` terms. The
   uniform proposal has excess kurtosis :math:`-6/5` per component, giving a negative
   correction of order :math:`1/n_\mathrm{eff}` that competes with the positive one above.

Of stage II:

4. the :math:`\mathrm{Var}_\mathbf{R}[m]` contribution to :math:`\mathrm{Var}[X]` is dropped at
   order :math:`\mathrm{dtvmc}^4`;
5. the interchange is controlled only to second order and requires small
   :math:`\mathrm{CV}(u)` and :math:`\mathrm{CV}(v)`; the second is not satisfied in practice.

Outside both stages:

6. the virial theorem holds for an eigenstate, or for a trial function optimized with respect
   to a uniform scaling of all coordinates; the sum rule itself holds for any smooth
   :math:`\Psi`.

.. _vmc-ebes:

EBES
----

Only stage I changes. With :math:`\mathbf{R}` fixed *and* the electron :math:`i` chosen, the
sum runs over 3 components rather than :math:`3N_e`:

.. math::

    \mathrm{Var}_\Delta[X | \mathbf{R}, i]
    = \frac{4\,\mathrm{dtvmc}^2}{3}\,|\nabla_i \Lambda|^2

Averaging over the uniform choice of electron as well gives
:math:`\mathbb{E}_i \mathbb{E}_\mathbf{R}[|\nabla_i\Lambda|^2] = 2\langle T\rangle / N_e`, so

.. math::

    \mathrm{Var}[X] = \frac{8}{3}\,\mathrm{dtvmc}^2\,\frac{\langle T \rangle}{N_e},
    \qquad
    \mathrm{dtvmc}_{50}(\mathrm{EBES}) = \sqrt{N_e}\;\mathrm{dtvmc}_{50}(\mathrm{CBCS})

The EBES step is therefore set by the kinetic energy *per electron*, an intensive quantity: at
fixed composition it does not depend on the size of the system, and for neutral atoms it
scales as :math:`Z^{-2/3}`. Point 3 of the previous section becomes decisive, however —
:math:`X` is a sum of three terms, the central limit theorem does not apply, and the Gaussian
form of :math:`g` cannot be used at all.

.. _vmc-acceptance-ratio:

acceptance_ratio
----------------

Measures the fraction of accepted moves at the current step size. CBCS counts an accepted move
directly. EBES records a step as accepted when *at least one* electron moved, so the reported
fraction is :math:`1 - (1 - a)^{N_e}` for a per-electron probability :math:`a`; the latter is
recovered by inverting that expression. Without the inversion the measured quantity can never
exceed :math:`1/N_e` and the step-size fit is blind to its own target.

.. _vmc-optimize-step:

optimize_vmc_step
-----------------

The acceptance is measured on a grid of eleven step sizes spanning twice the value returned by
:ref:`approximate_step_size <vmc-approximate-step-size>`, averaged over all MPI ranks, and
fitted to

.. math::

    p(\mathrm{dtvmc}) = \frac{e^{a/\lambda} - 1}
    {e^{a/\lambda} + e^{\mathrm{dtvmc}/\lambda} - 2}

whose parameter :math:`a` is by construction the step size at 50 % acceptance and is adopted
as ``dtvmc``. The functional form is empirical, chosen because it is monotone, equals 1 at
zero step size and decays exponentially; the theory above predicts :math:`a`, not the shape of
the curve away from the target.

:meth:`casino.pycasino.Casino.vmc_step_graph` scans the same quantity over a wider range and
writes the raw acceptance curve. Its grid is fixed in units of
:math:`\mathrm{dtvmc} \times N_e` rather than in units of
:ref:`approximate_step_size <vmc-approximate-step-size>`, so that the measurements in
``examples/time_step`` stay unnormalized by the very law they are used to test.

References
----------

.. [22] G. O. Roberts, A. Gelman, and W. R. Gilks,
   *Weak convergence and optimal scaling of random walk Metropolis algorithms*,
   Ann. Appl. Probab. **7**, 110 (1997).

.. [23] R. M. Lee, G. J. Conduit, N. Nemec, P. López Ríos, and N. D. Drummond,
   *Strategies for improving the efficiency of quantum Monte Carlo calculations*,
   Phys. Rev. E **83**, 066706 (2011).

.. [24] E. H. Lieb,
   *Thomas-Fermi and related theories of atoms and molecules*,
   Rev. Mod. Phys. **53**, 603 (1981).

.. [25] J. M. C. Scott,
   *The binding energy of the Thomas-Fermi atom*,
   Philos. Mag. **43**, 859 (1952).
