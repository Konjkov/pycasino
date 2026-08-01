.. _vmc:

Variational Monte Carlo
=======================

Variational Monte Carlo is implemented in the :class:`casino.vmc.VMC` class. It samples the
probability density :math:`|\Psi(\mathbf{R})|^2` of a trial wave function with the Metropolis
algorithm and estimates the expectation value of the Hamiltonian as the mean of the local
energy over the resulting chain. The Hamiltonian enters only through that average: the walk
itself is driven by :math:`|\Psi|^2` alone.

Three sampling modes are named by ``vmc_method``, of which two are implemented:

- **EBES** (``vmc_method : 1``) — electrons are moved one at a time, each with its own
  accept/reject step;
- **DBDS** (``vmc_method : 2``) — one spin determinant is displaced at a time.
  :ref:`Not implemented <vmc-dbds>`, and selecting it silently freezes the walk;
- **CBCS** (``vmc_method : 3``) — all electrons are moved at once, single accept/reject.

The single free parameter of the walk is the step size, set by ``dtvmc``. The formulas below are
written in :math:`\tau`, the half-width of the cube Pycasino proposes from, which is not the same
number: ``dtvmc`` is the variance of one displacement component, so :math:`\mathtt{dtvmc} =
\tau^2/3`, and :ref:`the convention <vmc-dtvmc>` is what makes the step comparable between codes.
The step is too small when
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

The full average is :math:`\mathbb{E} = \mathbb{E}_\mathbf{R}\,\mathbb{E}_\Delta`, innermost
first, and that is the order the algorithm performs: the walker sits at :math:`\mathbf{R}` and
draws :math:`\boldsymbol{\Delta}`, and only the chain distributes :math:`\mathbf{R}` itself.
The order matters, because the acceptance probability is a non-linear function of the
configuration and the two stages do not commute. The derivation below commutes them anyway —
it averages the local kinetic energies over the chain and only then applies the non-linear
acceptance function, rather than the reverse — and the whole of
:ref:`the last section <vmc-cost-of-interchange>` is the price of doing so. That price is paid
by the *estimate* of the step size and by nothing else: the walk itself never commutes anything
and samples :math:`|\Psi|^2` exactly at any step size.

Three local kinetic energies are used. They are functions of :math:`\mathbf{R}`, hence
constants with respect to :math:`\mathbb{E}_\Delta`:

.. math::

    \begin{aligned}
    T_L(\mathbf{R}) &= -\frac{1}{2}\frac{\nabla^2\Psi}{\Psi} \\
    T_D(\mathbf{R}) &= \frac{1}{2}\,\mathbf{F}\cdot\mathbf{F}
                     = \frac{1}{2}\sum_i |\nabla_i\Lambda|^2 \\
    T_M(\mathbf{R}) &= \frac{1}{2}\left(T_L + T_D\right)
    \end{aligned}

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
:math:`[-\tau, \tau]`. The proposal is symmetric, so the Metropolis ratio
reduces to the density ratio alone:

.. math::

    \begin{aligned}
    a(\mathbf{R} \to \mathbf{R}') &= \min(1, e^X) \\
    X = \ln\frac{|\Psi(\mathbf{R}')|^2}{|\Psi(\mathbf{R})|^2} &= 2\left[\Lambda(\mathbf{R}') - \Lambda(\mathbf{R})\right]
    \end{aligned}

Everything reduces to the statistics of the single scalar :math:`X`. The mean acceptance is a
two-stage average, in the order the algorithm performs it — the walker sits at
:math:`\mathbf{R}` and draws :math:`\boldsymbol{\Delta}`, and only then is :math:`\mathbf{R}`
itself distributed according to :math:`|\Psi|^2`:

.. math::

    \begin{aligned}
    A(\tau) &= \mathbb{E}_\mathbf{R}\left[\alpha(\mathbf{R})\right] \\
    \alpha(\mathbf{R}) &= \mathbb{E}_\Delta\left[\min(1, e^X)\,\middle|\,\mathbf{R}\right]
    \end{aligned}

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
:math:`\mathbb{E}_\Delta[\delta_k \delta_l] = (\tau^2/3)\,\delta_{kl}` and
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
    = \frac{\tau^2}{3}\,\mathrm{Tr}\,\mathsf{H}
    = \frac{\tau^2}{3}\,\nabla^2\Lambda

and in the variance the cross term vanishes because it involves only third moments of a
symmetric distribution:

.. math::

    \begin{aligned}
    \mathrm{Var}_\Delta[X|\mathbf{R}]
    &= \mathrm{Var}_\Delta\!\left(2\boldsymbol{\Delta}\cdot\mathbf{F}\right)
     + \underbrace{\mathrm{Var}_\Delta\!\left(\boldsymbol{\Delta}^{\mathsf T}\mathsf{H}
       \boldsymbol{\Delta}\right)}_{O(\tau^4)}
     + \underbrace{2\,\mathrm{Cov}_\Delta}_{0} \\
    &= \frac{4\,\tau^2}{3}\,\mathbf{F}\cdot\mathbf{F} + O(\tau^4)
    \end{aligned}

The mean is therefore of second order in the step size while the variance survives at first
non-vanishing order, and both are kinetic energies:

.. math::

    \begin{aligned}
    m(\mathbf{R}) &\equiv \mathbb{E}_\Delta[X|\mathbf{R}]
    = -\frac{4\,\tau^2}{3}\,T_M(\mathbf{R}) \\
    s^2(\mathbf{R}) &\equiv \mathrm{Var}_\Delta[X|\mathbf{R}]
    = \frac{8\,\tau^2}{3}\,T_D(\mathbf{R})
    \end{aligned}

The asymmetry is essential: the mean sees the combination :math:`T_L + T_D`, the variance sees
:math:`T_D` alone, and the Laplacian form does not enter the variance at all.

At fixed :math:`\mathbf{R}` the leading part of :math:`X` is a sum of :math:`3N_e` independent
terms and is normal up to :math:`O(1/n_\mathrm{eff})`, where
:math:`n_\mathrm{eff} = (\sum_k F_k^2)^2 / \sum_k F_k^4` is the participation ratio of the
drift components. With that,

.. math::

    \begin{aligned}
    \alpha(\mathbf{R}) &= g\big(m(\mathbf{R}), s(\mathbf{R})\big) \\
    g(m, s) &= \Phi\!\left(\frac{m}{s}\right)
             + e^{m + s^2/2}\,\Phi\!\left(-s - \frac{m}{s}\right)
    \end{aligned}

which is the standard integral

.. math::

    \mathbb{E}[\min(1,e^X)] = P(X \ge 0)
    + \int_{-\infty}^{0} e^x\,\mathcal{N}(x; m, s^2)\,dx

Note that pointwise

.. math::

    m(\mathbf{R}) + \tfrac{1}{2}s^2(\mathbf{R})
    = \frac{2\,\tau^2}{3}\left[T_D(\mathbf{R}) - T_L(\mathbf{R})\right] \neq 0

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

    \begin{aligned}
    \mu \equiv \mathbb{E}[X] &= \mathbb{E}_\mathbf{R}[m]
    = -\frac{4}{3}\,\tau^2 \langle T \rangle \\
    \sigma^2 \equiv \mathrm{Var}[X]
    &= \mathbb{E}_\mathbf{R}[s^2] + \mathrm{Var}_\mathbf{R}[m]
    = \frac{8}{3}\,\tau^2 \langle T \rangle + O(\tau^4)
    \end{aligned}

the discarded piece being
:math:`(16\,\tau^4/9)\,\mathrm{Var}_\mathbf{R}(T_M)`. Hence
:math:`\mu = -\sigma^2/2`, which is not an assumption but a consequence: stationarity of the
chain requires :math:`\mathbb{E}[e^X] = 1`, that is :math:`\mu + \sigma^2/2 = 0` for normal
:math:`X`, and the expansion reproduces it by itself. It appears only after stage II — stage I
does not give it.

The second relation is a sum rule:

.. math::

    \mathrm{Var}[X] = \frac{8}{3}\,\tau^2 \langle T \rangle

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

    \tau_{50}\,\sqrt{\langle T \rangle}
    = s\sqrt{3/2} = \sqrt{3}\,\mathrm{erfinv}(1/2) = 0.8260784

For a general target :math:`A` the constant is
:math:`\sqrt{3}\,\mathrm{erfinv}(1 - A)`; at the Roberts–Gelman–Gilks optimal acceptance
0.234 [22]_ it would be 1.4576. There is nothing fundamental about 0.826: it is fixed by three
conventions — a uniform cube rather than a Gaussian proposal, a 50 % target, and atomic
units. Nor is there any reason for it to equal one, the prefactor being dimensional,
:math:`[\tau] = \sqrt{1/\mathrm{energy}}`.

.. _vmc-dtvmc:

Only the second moment of the proposal enters the sum rule, so the shape of the proposal drops
out once the step size is quoted as that moment. This is what ``dtvmc`` is: Casino draws each
component from a Gaussian of variance ``dtvmc``, Pycasino from a cube of half-width
:math:`\tau` and hence variance :math:`\tau^2/3`, and the two agree on

.. math::

    \mathtt{dtvmc} \cdot \langle T \rangle
    = \frac{1}{2}\left[\Phi^{-1}(3/4)\right]^2 = 0.227475

which is the form to compare codes in, and the one used below.

.. _vmc-approximate-step-size:

approximate_step_size
---------------------

To use the result as an initial guess, :math:`\langle T \rangle` is needed before any sampling
has been done. The virial theorem gives :math:`\langle T \rangle = |E|`, for an ion no less than
for a neutral atom since the net charge enters nowhere, and the Thomas–Fermi expansion with its
Scott and Dirac corrections [24]_ [25]_ supplies the energy with no fitted constant at all:

.. math::

    \langle T \rangle \approx \sum_a \left[
        0.7687\,Z_a^{7/3} - 0.5\,Z_a^2 + 0.2699\,Z_a^{5/3} \right]

which reproduces Hartree–Fock energies to 1.8 % rms. The leading term alone is much worse than
it looks — 4.2 % rms, with the effective coefficient
:math:`|E| / \sum_a Z_a^{7/3}` drifting from 0.54 to 0.64 across the periodic table — and that
drift is exactly the Scott term.

Neutrality is assumed by the expansion rather than by the virial theorem: its leading term
minimizes the Thomas–Fermi functional subject to :math:`\int\!\rho = Z`. The number of electrons
deliberately does not appear — substituting it for :math:`Z_a` would put
:math:`\mathrm{Be}^{2+}` 80 % low in :math:`\langle T \rangle` and 123 % high in :math:`\tau` —
so what the formula returns for an ion is the neutral atom carrying the same nucleus:
:math:`-1.0\,\%` on :math:`\mathrm{Li}^+` and :math:`+4.7\,\%` on :math:`\mathrm{Be}^{2+}`,
:math:`+0.5\,\%` and :math:`-2.3\,\%` in :math:`\tau`. The two errors partly cancel, the
expansion running low on a neutral atom while a cation sits below its neutral parent.

The expansion does extend to ions, and the extension is worth stating because it is not the
obvious one. Thomas–Fermi theory scales as :math:`E = -Z^{7/3}e(\lambda)` in
:math:`\lambda = N/Z` [24]_, the Scott term survives untouched — it comes from
:math:`r \sim 1/Z`, where no valence electron screens anything — and only the exchange part of
the :math:`Z^{5/3}` coefficient follows the density. But there is no term linear in the net
charge, because the neutral Thomas–Fermi atom has zero chemical potential: :math:`\partial
E/\partial N` vanishes at :math:`N = Z`, hence :math:`e'(1) = 0`. Solving the equation with the
ion boundary condition confirms it, :math:`e(1) - e(\lambda)` falling as
:math:`(q/Z)^{2.45}` at :math:`q/Z = 0.02` and approaching the classical :math:`(q/Z)^{7/3}`
from above. Carried out exactly the correction works: it moves :math:`\mathrm{Li}^+` to
:math:`-4.3\,\%` and :math:`\mathrm{Be}^{2+}` to :math:`-4.1\,\%`, which is where the same
expansion already sits on *neutral* lithium and beryllium, :math:`-3.6\,\%` and
:math:`-2.3\,\%`. The ion-specific error is genuinely removed and what is left is the series
being 2 to 4 % low at :math:`Z` of 3 and 4. That is also why it is not used here: the residual
it exposes is the one that was cancelling against it, so nothing is gained in :math:`\tau`,
while the price is a numerical solution of the Thomas–Fermi equation and a per-atom partition
of the electron count that a molecular ion does not admit.

Writing :math:`T_a` for the share of :math:`\langle T \rangle` carried by nucleus :math:`a`, the
remaining correction is expressed through its participation ratio,

.. math::

    \begin{aligned}
    n_\mathrm{nuc} &= \frac{\left(\sum_a T_a\right)^2}{\sum_a T_a^2} \\
    \tau &= \sqrt{3}\,\mathrm{erfinv}(1/2)\,
    \frac{1 + 0.045 / n_\mathrm{nuc}}{\sqrt{\langle T \rangle}}
    \end{aligned}

which is :math:`1` for a single heavy atom and the number of equivalent nuclei for a symmetric
molecule. Everything here follows from the sum rule and from tabulated atomic physics except the
:math:`0.045`, the only fitted number in the estimate; the next section explains what it stands
for. Measured on the systems in ``examples/time_step/CBCS`` the 50 % acceptance point lands
within 0.9 % rms of this guess for atoms, ions, hydrides and hydrocarbons alike.

The other two sampling modes take the same estimate through their own sum rule, since only the
block of coordinates that moves changes:

.. math::

    \tau(\mathrm{EBES}) = \sqrt{N_e}\;\tau, \qquad
    \tau(\mathrm{DBDS}) = \sqrt{2}\;\tau

The EBES factor is the Gaussian one, and :ref:`its central limit theorem fails <vmc-ebes>`, so
the guess is knowingly off by the non-Gaussian correction; that is left to
:ref:`optimize_vmc_step <vmc-optimize-step>`, whose fixed point does not depend on the model.

.. _vmc-cost-of-interchange:

Cost of the interchange
-----------------------

Suppose the trial function were exact. The zero-variance property fixes :math:`E_L` at the
eigenvalue and with it :math:`T_L = E - V` pointwise, so the better the wave function, the more
rigidly the Laplacian estimator is tied to the potential — which diverges at every nucleus.
Perfection does not remove the fluctuation the interchange discards; it prescribes it.

What the interchange needs is not an accurate wave function but a featureless one,
:math:`\mathrm{Var}(T_D) = 0`, a log-density of constant slope, and it is again the exact wave
function that forbids it. The Kato cusp fixes :math:`|\nabla_i\Lambda| \to Z_a` as electron
:math:`i` reaches nucleus :math:`a`, the asymptotic decay :math:`\Psi \sim e^{-\sqrt{2I}\,r}`,
with :math:`I` the ionization energy the departing electron leaves against,
fixes it at :math:`\sqrt{2I}` far out, so each electron's share of :math:`T_D` swings between
:math:`Z_a^2/2` and :math:`I` — on neon, 50 against 0.79 — whatever is done to the ansatz. The
correction below is therefore a property of the system and not a defect of the trial function,
which is why nothing about the latter enters it. Measured on the campaign, the coefficient of
variation :math:`\mathrm{CV}(T_D) = \sqrt{\mathrm{Var}_\mathbf{R}(T_D)}\,/\,\langle T \rangle`
grows with the nuclear charge, 0.10 on helium, 2.5 on neon, 3.1 on krypton, and falls when the same charge is spread over more nuclei, 1.3 on
:math:`\mathrm{C_2H_2}` against 0.5 on :math:`\mathrm{C_6H_6}`.

The same asymmetry is reported from the diffusion side by Umrigar, Nightingale and Runge [26]_,
who observe that the acceptance of a drift–diffusion move is considerably lower for electrons
close to a nucleus than for electrons far from any, and who raise it by replacing the Gaussian
proposal near a nucleus with one carrying the cusp. Their remedy does not transfer: it repairs a
Green function that is an approximation, whereas the Metropolis walk samples :math:`|\Psi|^2`
exactly at every step size, so here the same nonuniformity costs efficiency and not accuracy.
The observation does transfer, and it is the one being quantified above.

The interchange is thus the only uncontrolled step of the derivation. Its sign follows
without any calculation: :math:`g` is convex in :math:`T_D`, so by Jensen's inequality the
spread of :math:`T_D` *raises* the acceptance and a larger step is needed. The correction is
positive by construction and is exactly the Jensen gap of commuting
:math:`\mathbb{E}_\mathbf{R}` with :math:`g`.

In terms of the dimensionless fluctuations
:math:`u = T_D/\langle T\rangle - 1` and :math:`v = T_L/\langle T\rangle - 1`, expanding the
exact acceptance to second order and solving :math:`A = 1/2` gives

.. math::

    \frac{\delta\,\tau}{\tau}
    = c_{DD}\,\mathbb{E}_\mathbf{R}[u^2]
    + c_{DL}\,\mathbb{E}_\mathbf{R}[uv]
    + c_{LL}\,\mathbb{E}_\mathbf{R}[v^2]

.. math::

    \begin{aligned}
    c_{DD} &= \frac{1}{8} - \frac{s^2}{4} + \frac{s^3}{16\varphi(s)} = +0.071617 \\
    c_{DL} &= \frac{s^2}{2} - \frac{s^3}{8\varphi(s)} = +0.106766 \\
    c_{LL} &= -\frac{s^2}{8} + \frac{s^3}{16\varphi(s)} = +0.003484
    \end{aligned}

with :math:`c_{DD} + c_{DL} + c_{LL} = (1 + s^2)/8`,
:math:`c_{DD} - c_{LL} = (1 - s^2)/8` and :math:`c_{DL} + 2c_{LL} = s^2/4`; the terms in
:math:`s^3/\varphi` cancel in the sum. Treating :math:`\nabla^2\Lambda` as non-fluctuating,
that is setting :math:`T_L(\mathbf{R}) = T_D(\mathbf{R})` pointwise, collapses this to a single
term,

.. math::

    \frac{\delta\,\tau}{\tau}
    = \frac{1 + s^2}{8}\,\frac{\mathrm{Var}_\mathbf{R}(T_D)}{\langle T \rangle^2}
    = 0.18187\,\mathrm{CV}^2(T_D)

and this is the form behind the fitted :math:`1 + 0.045/n_\mathrm{nuc}`: :math:`T_D` is
dominated by whichever electron is currently near a nucleus, since :math:`|\nabla\Lambda| \sim Z`
there, so :math:`n` equivalent nuclei contribute :math:`n` independent terms and
:math:`\mathrm{CV}^2(T_D)` falls as :math:`1/n`.

This one-constant form should not be trusted quantitatively. Rewritten in the variables Casino
prints, :math:`t = T_M/\langle T\rangle - 1 = (u + v)/2`, the same expression reads

.. math::

    \frac{\delta\,\tau}{\tau}
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

.. admonition:: The interchange is avoidable in principle

   The exact acceptance is the average of a known function of two quantities that are already
   evaluated at every configuration, so accumulating :math:`T_M` and :math:`T_D` over the
   equilibration walk — which runs at :math:`|\Psi|^2` *before* the step is optimized — and
   solving :math:`A(\tau) = 1/2` numerically would remove the Thomas–Fermi estimate, the virial
   theorem and the fitted constant together, and would account for the Jastrow factor, backflow,
   ions and pseudopotentials for free, since the function actually being sampled is the one being
   measured.

.. _vmc-validity:

Range of validity
-----------------

Of stage I:

1. the expansion of :math:`X` is truncated at second order in
   :math:`\boldsymbol{\Delta}`, and at the 50 % point
   :math:`\tau^2 \langle T\rangle \approx 0.68`, so the step is not small;
2. :math:`\Psi` is not analytic at a nuclear cusp, which is precisely where :math:`T_L`
   diverges, so the Taylor expansion formally fails there;
3. normality of the leading term rests on a central limit theorem over :math:`3N_e` terms. The
   uniform proposal has excess kurtosis :math:`-6/5` per component, giving a negative
   correction of order :math:`1/n_\mathrm{eff}` that competes with the positive one above.

Of stage II:

4. the :math:`\mathrm{Var}_\mathbf{R}[m]` contribution to :math:`\mathrm{Var}[X]` is dropped at
   order :math:`\tau^4`;
5. the interchange is controlled only to second order and requires small
   :math:`\mathrm{CV}(u)` and :math:`\mathrm{CV}(v)`; the second is not satisfied in practice.

Outside both stages:

6. the virial theorem holds for an eigenstate, or for a trial function optimized with respect
   to a uniform scaling of all coordinates; the sum rule itself holds for any smooth
   :math:`\Psi`.

.. _vmc-pseudopotential:

Pseudopotentials and the electron gas
-------------------------------------

Neither stage of the derivation ever differentiates the potential: everything is built from
:math:`\nabla\Lambda` and :math:`\nabla\nabla\Lambda`, and :math:`V` enters only the local
energy, which the walk never sees. The sum rule
:math:`\mathrm{Var}[X] = (8/3)\,\tau^2\langle T \rangle` therefore holds verbatim for a
pseudopotential, local or non-local, and for a periodic system. What breaks is only point 6
above, the route from the Hamiltonian to a *number* for :math:`\langle T \rangle`:

- the virial theorem needs a potential homogeneous of degree :math:`-1` under a uniform
  scaling. A pseudopotential is not, so :math:`\langle T \rangle \neq |E|`; in the Casino runs
  under ``examples/ppotential_HF`` the ratio to the valence energy runs from 0.62 on carbon to
  0.81 on neon, with no plateau to extrapolate to;
- Thomas–Fermi is a semiclassical theory of many electrons and says nothing about the handful
  of valence electrons a pseudopotential leaves behind. Fed the pseudo charge it overshoots
  fourfold on carbon — :math:`13.2` against a measured :math:`3.3` — because it places those
  four electrons in a :math:`1s` shell.

The valence electrons are hydrogenic instead. With :math:`Z_a` the pseudo charge of nucleus
:math:`a`, :math:`n_a` the principal quantum number of its row, and Slater screening by the
other valence electrons,

.. math::

    \langle T \rangle \approx \sum_a \frac{Z_a (0.65\,Z_a + 0.35)^2}{2\,n_a^2}

This is exact on hydrogen and 11 % to 55 % high across the second row, a pseudo-orbital being
smoother than the hydrogenic one it is modelled on, which halves into a 5 % to 20 % low
estimate of :math:`\tau` — well inside the basin of
:ref:`optimize_vmc_step <vmc-optimize-step>`.

That the sum rule itself survives can be read off existing Casino output, in the
proposal-independent form. Taking every CBCS run under ``examples`` and multiplying the
optimized ``dtvmc`` by the ``KEI`` the same run reports:

.. list-table::
   :widths: 30 15 25 30
   :header-rows: 1
   :width: 100%

   * - Runs
     - Number
     - :math:`\langle T \rangle` (au)
     - :math:`\mathtt{dtvmc}\cdot\langle T \rangle`
   * - all-electron
     - 184
     - 2.4 … 2781
     - 0.261 ± 15 %
   * - pseudopotential
     - 72
     - 0.49 … 57
     - 0.293 ± 14 %

The product is constant over a factor of 1160 in :math:`\langle T \rangle`, with and without a
pseudopotential — which is the claim being tested, and the only one these runs settle. Both
medians sit above the Gaussian :math:`0.227475` by the
:ref:`Jensen gap <vmc-cost-of-interchange>`, :math:`+7\,\%` and :math:`+14\,\%` in
:math:`\tau` — the all-electron figure compatible with the :math:`4.5\,\%` fitted above, the
pseudopotential one larger as expected, since removing the core leaves only three to eight
electrons to average :math:`T_D` over. The :math:`15\,\%` scatter, however, is not
physics but the tolerance of Casino's own step-size optimizer, which stops as soon as the
acceptance is near enough to 50 %, so nothing finer — the :math:`1/n_\mathrm{nuc}` collapse in
particular — can be extracted from these runs. That is what the campaign described in
:ref:`optimize_vmc_step <vmc-optimize-step>` is for. There is also no pseudopotential
*molecule* to test it on: :math:`\mathrm{B_2H_6}` under ``examples/ppotential_HF`` carries a
pseudopotential on hydrogen only, and pseudo-hydrogen is hydrogen.

The homogeneous electron gas is the opposite limit and the cleanest test the law admits.
:math:`\langle T \rangle` is known in closed form, without a virial theorem or a fit,

.. math::

    \langle T \rangle = \frac{3}{10}
    \left(\frac{6\pi^2}{\Omega}\right)^{2/3}
    \left(N_\uparrow^{5/3} + N_\downarrow^{5/3}\right)
    = \frac{1.10495\,N_e}{r_s^2} \quad (\text{unpolarized})

so the prediction carries no adjustable constant at all: :math:`\mathtt{dtvmc} =
0.2058\,r_s^2 / N_e` in CBCS, and :math:`N_e` times that, :math:`0.2058\,r_s^2`, in
:ref:`EBES <vmc-ebes>`, where the kinetic energy enters per electron and the size of the
system drops out.

One condition attaches to that closed form. What the sum rule needs is
:math:`\langle T_D \rangle`, and :math:`\Lambda = \ln|\Psi|` knows the modulus only, so for a
complex wave function the physical kinetic energy exceeds it by the current term
:math:`\langle |\nabla\theta|^2 \rangle / 2`, which :math:`|\Psi|^2` cannot see. At the
:math:`\Gamma` point with a closed shell the determinant is real up to a global phase and the
two coincide, which is the case the number above describes. Under twisted boundary conditions —
the standard route to finite-size corrections — or with an open shell, the determinant carries a
current, :math:`3 k_F^2 N_e / 10` is then larger than what the sum rule is asking for, and the
step comes out too small. Nothing in the derivation breaks: it is the closed-form input that
stops being the right quantity, and the free-particle expression should be read as the
modulus-only kinetic energy wherever the two part company.

Translational invariance costs nothing here, which is worth saying because it looks as though it
should. Displacing every electron by the same vector leaves :math:`|\Psi|^2` exactly unchanged,
so of the :math:`3N_e` components a CBCS move proposes, three do not enter :math:`X` at all,
while EBES moves one electron and cannot produce such a displacement in the first place. Neither
formula changes, because neither counts degrees of freedom: both follow from
:math:`\sum_k F_k^2`, and the invariance is precisely the statement
:math:`\sum_i \nabla_i \Lambda = 0`, which puts no weight in that subspace to begin with. The
bookkeeping is automatic, and the same holds for :math:`n_\mathrm{eff}`, which counts
:math:`3N_e - 3` of its own accord. What the free subspace does cost is efficiency, at order
:math:`1/N_e`: the centre of mass random-walks with the full step and its displacement enters
the diffusion criterion below without ever having affected the acceptance. No atom or molecule
has this subspace, the clamped nuclei supplying the missing force.

The Jensen gap should be far smaller than in any
atom, since there is no nuclear cusp for :math:`T_D` to be dominated by and
:math:`\mathrm{CV}(T_D)` falls as :math:`1/\sqrt{N_e}`, so the interchange that costs 4.5 % on
a single atom should cost almost nothing here. Pycasino has no homogeneous-gas wave function,
so this is a prediction rather than a measurement; ``atom_kinetic_energy`` returns an empty
array for a system without nuclei and the guess falls back to unity, which
:ref:`optimize_vmc_step <vmc-optimize-step>` recovers from without help.

The gas is also the only system for which a published estimate of the step size exists to
check against. Section II.C of [23]_ sets the RMS distance an electron diffuses in one EBES
move equal to :math:`r_s`, assumes the acceptance is 50 %, and obtains
:math:`\mathtt{dtvmc} = 2 r_s^2/3 = 0.667\,r_s^2` — a factor 3.24 above the
:math:`0.2058\,r_s^2` the sum rule gives at that same 50 %. The estimate does not produce the
acceptance it assumes, which is one reason its authors call it far from optimal. But the 50 %
target is a convention too, and the alternative Casino implements as ``OPT_DTVMC`` 2 is to
maximize the diffusion constant, the mean square displacement per attempted move, which is
proportional to :math:`\mathtt{dtvmc}\cdot A`. Writing :math:`u = \sigma/2`, that is the
maximum of :math:`u^2\,\Phi(-u)`, stationary where

.. math::

    2\,\Phi(-u) = u\,\varphi(u), \qquad u = 1.1906, \qquad A = 0.2338

so the diffusion-optimal ``dtvmc`` is :math:`(u/s)^2 = 3.12` times the one the 50 % rule sets,
and the acceptance it wants is the Roberts–Gelman–Gilks 0.234 [22]_ — obtained here from the
sum rule and a definition of efficiency rather than from a scaling limit. Carried through,
:math:`\mathtt{dtvmc} = 0.641\,r_s^2`, within 4 % of :math:`2 r_s^2/3`. The published estimate
is therefore not the 50 % step it is derived as but, to that accuracy, the diffusion-optimal
one: taking :math:`r_s` as the length overshoots the 50 % step by 3.24, the target the
criterion should have used overshoots it by 3.12, and the two cancel. Both figures inherit the
reservation of the next section — three components move, not :math:`3N_e`, so the Gaussian
:math:`A(\sigma)` is at its weakest exactly where this comparison is made.

.. _vmc-ebes:

EBES
----

Only stage I changes. With :math:`\mathbf{R}` fixed *and* the electron :math:`i` chosen, the
sum runs over 3 components rather than :math:`3N_e`:

.. math::

    \mathrm{Var}_\Delta[X | \mathbf{R}, i]
    = \frac{4\,\tau^2}{3}\,|\nabla_i \Lambda|^2

Averaging over the uniform choice of electron as well gives
:math:`\mathbb{E}_i \mathbb{E}_\mathbf{R}[|\nabla_i\Lambda|^2] = 2\langle T\rangle / N_e`, so

.. math::

    \mathrm{Var}[X] = \frac{8}{3}\,\tau^2\,\frac{\langle T \rangle}{N_e},
    \qquad
    \tau_{50}(\mathrm{EBES}) = \sqrt{N_e}\;\tau_{50}(\mathrm{CBCS})

The EBES step is therefore set by the kinetic energy *per electron*, an intensive quantity: at
fixed composition it does not depend on the size of the system, and for neutral atoms it
scales as :math:`Z^{-2/3}`. Point 3 of the previous section applies in its sharpest form,
however: :math:`X` is a sum of three terms, not :math:`3N_e`, so there is no limit for the
central limit theorem to be taken in, and the effective count is smaller still — the drift of a
single electron near a nucleus is nearly radial, which drives
:math:`n_\mathrm{eff} = (\sum_k F_k^2)^2 / \sum_k F_k^4` from 3 towards 1, where :math:`X` is
one scaled uniform variable with bounded support and excess kurtosis :math:`-6/5`.

How much that costs depends entirely on where the acceptance is being asked for. The exact
:math:`\alpha` for the cube is available without any appeal to the theorem, :math:`X` being a
weighted Irwin–Hall variable whose density is piecewise quadratic, against which
:math:`\int \min(1,e^x)` integrates in closed form. Solved that way, the step the cube requires
differs from the Gaussian answer by :math:`+3.0\,\%` at :math:`n_\mathrm{eff} = 1` and under
:math:`0.5\,\%` at 2 or 3 — at the 50 % target. The bodies of two symmetric distributions of
equal variance agree, and the 50 % point sees only the body. The tails are another matter: at a
1 % target the same comparison reads :math:`-15.8\,\%`, :math:`-7.5\,\%` and
:math:`-3.7\,\%`, and the Gaussian form put at :math:`n_\mathrm{eff} = 1` predicts an
acceptance of 0.0007 where the truth is 0.01. So the Gaussian form remains serviceable for
setting the step and fails as a description of the acceptance curve away from it.

The clean repair is to stop proposing from a cube. A sum of Gaussians is Gaussian whatever the
number of summands, so a Gaussian displacement makes :math:`X` exactly normal at leading order
in EBES, with none of this reasoning needed; the bounded support that recommends the cube
elsewhere buys nothing when three components move.
:meth:`casino.vmc.VMC.random_step` dispatches the proposal by method, so the substitution is
local to one function.

.. _vmc-dbds:

DBDS
----

``vmc_method : 2`` selects determinant-by-determinant sampling, in which one spin determinant
is displaced at a time — all :math:`N_\uparrow` up-spin electrons together, then all
:math:`N_\downarrow` down-spin ones, each with its own accept/reject step. It sits between the
two implemented modes: EBES with a block size of one electron, CBCS with a block size of
:math:`N_e`.

**It is not implemented.** :meth:`casino.vmc.VMC.random_step` dispatches methods 1 and 3 only
and returns ``False`` for anything else, so the walker never moves and the run silently
produces the initial configuration repeated ``vmc_nstep`` times.
:ref:`approximate_step_size <vmc-approximate-step-size>` nevertheless still has a live branch
for it, returning the :math:`\sqrt{2}` of the sum rule below, which makes the mode look
supported.

Casino removed its own Method 2 as well: it "did not offer any advantage over the other
methods and was hard to support", and the one feature worth keeping — using acceptance
probabilities in the accumulation — was moved into Method 3.

The sum rule says what could be expected of it. A move of the :math:`\sigma` determinant is
the derivation of :ref:`stage I <vmc-stage-one>` with the sum restricted to
:math:`3N_\sigma` components, so

.. math::

    \mathrm{Var}[X] = \frac{8}{3}\,\tau^2 \langle T_\sigma \rangle, \qquad
    \langle T_\sigma \rangle = \frac{1}{2}\,\mathbb{E}_\mathbf{R}
    \left[\sum_{i \in \sigma} |\nabla_i\Lambda|^2\right]

For a closed-shell system the two spin channels carry half the kinetic energy each, so
:math:`\tau_{50}(\mathrm{DBDS}) = \sqrt{2}\,\tau_{50}(\mathrm{CBCS})` — a factor
:math:`\sqrt{2}`, against :math:`\sqrt{N_e}` for EBES. Three limitations follow.

- **The gain in step size is bounded by** :math:`\sqrt{2}` **whatever the system**, because
  there are only ever two determinants. It does not grow with :math:`N_e`, unlike the EBES
  gain.
- **An open-shell system cannot be served by one** ``dtvmc``. The optimal step of each channel
  is :math:`0.826/\sqrt{\langle T_\sigma \rangle}`, and the two differ whenever
  :math:`N_\uparrow \neq N_\downarrow`; a single step size puts one determinant off its 50 %
  target. Neither the input keyword nor
  :ref:`optimize_vmc_step <vmc-optimize-step>` has room for two values.
- **The saving DBDS aims at does not exist in Pycasino.** Its point is that moving one
  determinant requires updating only that determinant's inverse, not both. Both implemented
  modes here call ``wfn.value`` on the whole configuration for every proposal — EBES already
  pays the full cost per single-electron move — so there is no per-determinant update
  machinery for DBDS to exploit. Adding it is the prerequisite, and it would speed up EBES
  first.

What DBDS keeps, and EBES loses, is the central limit theorem: :math:`X` is a sum of
:math:`3N_\sigma` terms rather than 3, so the Gaussian form of :math:`g` and hence the
:math:`0.826` law remain usable. That is the only theoretical argument in its favour, and it
buys :math:`\sqrt{2}`.

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

Since :math:`\sigma \propto \tau`, one measurement already fixes the whole curve. Inverting
:math:`A = 2\Phi(-\sigma/2)` at the measured acceptance and reading it back at
:math:`A = 1/2` gives

.. math::

    \tau \;\longleftarrow\; \tau\,
    \frac{\Phi^{-1}(3/4)}{\Phi^{-1}\!\left(1 - A(\tau)/2\right)}

which lands on the target in a single shot whenever the Gaussian law is exact. Two properties
make this safe to iterate rather than fit:

- its **fixed point is** :math:`A = 1/2` **for any monotone acceptance curve**, since the two
  inverse normals cancel identically there. The law sets the rate of convergence, not the
  answer, so the map cannot converge to the wrong step size where the Gaussian fails — EBES,
  a one-electron system, a badly optimized wave function;
- it is a genuine Newton step in the variable the law is linear in, so it converges from far
  away. Replayed on the measured curves in ``examples/time_step/CBCS``, a 40 % error becomes
  2 %, then 0.2 %, then 0.02 %.

Three iterations are used. They cost a third of the eleven-point scan and empirical two-parameter
fit they replace, they start from the current step size rather than resetting to the guess — so a
re-optimization after a wave-function update is a single correction — and the residual is set by
the noise of the acceptance, :math:`\delta\tau/\tau = \delta A / 2\varphi(s)\,s = 2.33\,\delta A`,
rather than by the model.

:meth:`casino.pycasino.Casino.vmc_step_graph` measures the shape of the curve for the same
reason it is not fitted here. Its grid is laid out **equally in acceptance** — nineteen targets
from 0.95 to 0.05 — through the same inverse,
:math:`\tau = \tau_{50}\,\Phi^{-1}(1 - A/2)/\Phi^{-1}(3/4)`, anchored on the
:math:`\tau_{50}` the optimizer has just measured. A grid in step size instead spends most of
its points in the saturated tails, where the acceptance carries no information about
:math:`\tau_{50}`, and spends them at system-dependent places, so nothing is comparable between
files; a grid in acceptance gives every system — atom, molecule, pseudoatom, gas — the same
points in the only variable the law knows about. The step sizes are written out in atomic
units, so the data stay unnormalized by the law they are used to test.

What the columns test
~~~~~~~~~~~~~~~~~~~~~

The acceptance on its own cannot say which half of the law has failed, since it is produced by
the sum rule and the Gaussian shape acting together. The moments of :math:`X` can, because the
sum rule is a statement about the second moment alone and says nothing about shape, while
detailed balance constrains :math:`X` whatever its shape is. Each grid point is therefore
recorded as six numbers taken from one and the same walk, arranged so that no two of them test
the same statement.

``acc_ratio``
    the acceptance, averaged as :math:`\langle \min(1, e^X) \rangle` over the proposals rather
    than counted from the accept/reject decisions that follow them. The same mean with a smaller
    variance, the coin flip having been integrated out analytically.

``correction``
    the step size actually used, divided by the one the law would prescribe for the acceptance
    just measured,

    .. math::

        \text{correction} = \tau \left/ \left[
        \Phi^{-1}\!\left(1 - A/2\right)\sqrt{\frac{3}{2\langle T\rangle}}\,\right]\right.

    One means the law is exact at that point. This is the factor
    :ref:`approximate_step_size <vmc-approximate-step-size>` carries as
    :math:`1 + 0.045/n_\mathrm{nuc}`, expressed in the same units, so the file offers the number
    to be fitted rather than an energy to be converted into one; the value in the 50 % row is
    that constant measured on that system. In :ref:`EBES <vmc-ebes>` the denominator carries
    :math:`\sqrt{N_e}` besides, the sum rule of a one-electron move seeing the kinetic energy per
    electron.

``sum_rule``
    :math:`\mathrm{Var}(X) \big/ \frac{8}{3}\tau^2\langle T\rangle`, the sum rule by itself. It
    assumes nothing about the shape of the distribution and never refers to the acceptance. One
    in the limit :math:`\tau \to 0`, where the rule is exact; what it loses at finite
    :math:`\tau` is the :math:`O(\tau^2)` term that the leading order drops.

``gaussian``
    :math:`\langle X \rangle + \mathrm{Var}(X)/2`. A stationary walk with a symmetric proposal
    has :math:`\langle e^X\rangle = 1` exactly, and the cumulant expansion of that identity,
    :math:`\kappa_1 + \kappa_2/2 + \kappa_3/6 + \kappa_4/24 + \dots = 0`, makes this column equal
    to :math:`-(\kappa_3/6 + \kappa_4/24 + \dots)` — every cumulant above the second and nothing
    besides. It vanishes if and only if :math:`X` is Gaussian, and it does so as :math:`\tau^3`,
    being led by the third cumulant of a quantity linear in the displacement. Divided by
    :math:`\mathrm{Var}(X)^{3/2}` it becomes a skewness, which is the form in which systems may
    be compared; undivided it grows along the grid for dimensional reasons alone.

``exp_mean``
    :math:`\langle e^X \rangle`, which is one for any correct stationary walk with a symmetric
    proposal. It tests the implementation rather than the physics, and costs nothing to report.
    Rare large :math:`X` dominate the estimator, so it is informative only where the acceptance
    is not tiny.

``kurtosis``
    the excess kurtosis of :math:`X`. Being dimensionless already, it has a finite
    :math:`\tau \to 0` limit, and that limit is the shape of :math:`X` at leading order. Two
    mechanisms set it and they carry opposite signs. The proposal is a cube, so :math:`X` is a
    weighted sum of uniform variables and is *platykurtic*; and the variance of :math:`X`
    conditioned on the position is proportional to the local :math:`T_D`, so :math:`X` is also a
    scale mixture over the walk and is *leptokurtic*. Which of the two wins is a property of the
    wave function, not of the sampling: an atom with one occupied shell has almost no spread in
    the local kinetic energy and shows the cube, while a core and a valence shell differing by
    orders of magnitude in :math:`|\nabla\ln\Psi|^2` bury it. The variance of this estimator is
    set by the eighth moment, so wherever the mixture wins, the small step end of the column is
    a sign and not a number.

The middle two columns multiply back into the first:

.. math::

    \text{correction} = \frac{1}{\sqrt{\text{sum\_rule}}}\;\cdot\;
    \frac{\sqrt{\mathrm{Var}(X)}}{2\,\Phi^{-1}\!\left(1 - A/2\right)}

which is an identity and not an approximation. The first factor uses :math:`\langle T\rangle`
and never the acceptance; the second uses the acceptance and never :math:`\langle T\rangle`.
The failure of the law therefore separates cleanly into the part where the sum rule is breaking
down and the part where :math:`X` is not Gaussian, and the second factor — the true width of
:math:`X` against the width a Gaussian would need to reproduce the same acceptance — is above
one exactly when :math:`X` is leptokurtic.

The separation matters because the two factors are individually system-dependent and move in
opposite directions. Across the all-electron atoms measured so far, at 5 % acceptance, the first
factor spans 1.18 to 1.44 and the second 0.87 to 1.03, while their product stays within 3 %.
The near-universality of the raw ``correction`` curve is a cancellation between two effects, not
a single law, and quoting it without the decomposition would hide that.

The level of the whole test is fixed by :math:`\langle T \rangle`, which the same run measures
on the same walk and writes into the header, so no fitted constant enters anywhere.

Both estimators are measured, with error bars, and the two questions to ask of them come apart.
On **variance** neither dominates: :math:`\mathrm{Var}(T_L)/\mathrm{Var}(T_D)` measures 199 on
helium, 0.1 on krypton, 0.2 on neon and 0.03 on pseudo-carbon. On **whether the error bar can be
believed**, :math:`T_D` wins everywhere. Between :math:`10^4` and :math:`10^5` samples the mean
of :math:`T_L` drifts from 2.590 to 2.732 on helium and from 2557 to 2796 on krypton, many times
its own quoted bar, while :math:`T_D` moves from 2.840 to 2.853 and from 2908 to 2754, each time
within its. :math:`T_L` has a heavy right tail — it is the unbounded one — and a sample standard
deviation does not describe it.

So :math:`T_D` is the reference, its larger bar on heavy atoms being an honest one, and
:math:`T_L` is reported beside it because the two share a mean by parts: their agreement within
the quoted bars is the convergence check, and where it fails the file says so instead of quietly
reporting one number.

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

.. [26] C. J. Umrigar, M. P. Nightingale, and K. J. Runge,
   *A diffusion Monte Carlo algorithm with very small time-step errors*,
   J. Chem. Phys. **99**, 2865 (1993).
