.. _dmc:

Diffusion Monte Carlo
=====================

Diffusion Monte Carlo is implemented in the :class:`casino.dmc.DMC` class. It projects out
the ground state from a trial wave function by evolving an ensemble of walkers (electron
configurations) in imaginary time:

.. math::

    \Phi_0 = \lim_{t \to \infty} e^{-t(\hat H - E_T)} \Psi

where :math:`E_T` is the reference (trial) energy. With importance sampling the walkers are
distributed according to the mixed distribution :math:`f(\mathbf{r}, t) = \Psi(\mathbf{r})\Phi(\mathbf{r}, t)`,
which satisfies the Fokker–Planck-like equation with drift, diffusion and branching terms.
For fermions the nodal surface of :math:`\Phi` is constrained to that of :math:`\Psi`
(the fixed-node approximation [17]_): moves that change the sign of :math:`\Psi` are rejected.

For a short time step :math:`\tau` the importance-sampled Green's function factorizes into
a drift-diffusion part and a branching part:

.. math::

    G(\mathbf{r} \to \mathbf{r}', \tau) \approx
    \underbrace{(2\pi\tau)^{-3N_e/2} \exp\left[-\frac{(\mathbf{r}' - \mathbf{r} - \tau \mathbf{v}(\mathbf{r}))^2}{2\tau}\right]}_{G_D}
    \cdot
    \underbrace{\exp\left[\frac{\tau (S(\mathbf{r}) + S(\mathbf{r}'))}{2}\right]}_{G_B}

where :math:`\mathbf{v} = \nabla \ln \Psi` is the drift velocity and :math:`S` is the
branching energy. One DMC step consists of a drift-diffusion move for every walker,
a branching (birth/death) step, and a T-move step if nonlocal pseudopotentials are present.

Summary of Methods
------------------

DMC class has the following methods:

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :width: 100%

   * - Method
     - Description
   * - :ref:`random_walk <dmc-random-walk>`
     - perform a number of DMC steps, return the history of :math:`E_{best}`
   * - :ref:`random_step <dmc-random-step>`
     - one DMC step: drift-diffusion, branching, T-move, update of :math:`E_T`
   * - :ref:`drift_diffusion <dmc-drift-diffusion>`
     - EBES (method 1) or CBCS (method 2) drift-diffusion move of all walkers
   * - :ref:`limiting_velocity <dmc-limiting-velocity>`
     - drift velocity with UNR limiting applied
   * - :ref:`alimit_vector <dmc-limiting-velocity>`
     - position-dependent limiting parameter :math:`a(\mathbf{r}_i)`
   * - :ref:`branching_energy <dmc-branching-energy>`
     - ZSGMA-limited branching energy :math:`S(\mathbf{r})`
   * - :ref:`branching <dmc-branching>`
     - stochastic birth/death of walkers according to their weights
   * - :ref:`t_move <dmc-t-move>`
     - Casula T-move for nonlocal pseudopotentials
   * - :ref:`load_balancing <dmc-load-balancing>`
     - redistribute walkers evenly across MPI ranks

.. _dmc-drift-diffusion:

drift_diffusion
---------------

Each walker is propagated by the Langevin step

.. math::

    \mathbf{r}' = \mathbf{r} + \tau \bar{\mathbf{v}}(\mathbf{r}) + \boldsymbol{\eta},
    \qquad \boldsymbol{\eta} \sim \mathcal{N}(0, \tau)

where :math:`\bar{\mathbf{v}}` is the :ref:`limited <dmc-limiting-velocity>` drift velocity.
The short-time Green's function is not exact, so a Metropolis accept/reject step
enforces detailed balance:

.. math::

    p = \min\left(1, \frac{G_D(\mathbf{r}' \to \mathbf{r}) |\Psi(\mathbf{r}')|^2}
    {G_D(\mathbf{r} \to \mathbf{r}') |\Psi(\mathbf{r})|^2}\right)

Moves crossing the nodal surface (:math:`\mathrm{sign}\,\Psi(\mathbf{r}') \neq \mathrm{sign}\,\Psi(\mathbf{r})`)
are rejected. To prevent persistent (stuck) walkers, the acceptance probability of a walker
that has not moved for more than 50 steps (CBCS) or 20 steps (EBES) is multiplied by
:math:`1.1^{\text{age}}`.

Two sampling modes are implemented:

- **CBCS** (``dmc_method : 2``) — all electrons are moved at once, single accept/reject.
- **EBES** (``dmc_method : 1``) — electrons are moved one at a time, each with its own
  accept/reject; the acceptance probability entering :math:`\tau_{eff}` is averaged with
  weights :math:`|\Delta \mathbf{r}_i|^2` of the per-electron diffusion steps.

The acceptance probability defines the effective time step used in branching:

.. math::

    \tau_{eff} = \tau \langle p \rangle

**Modifications for bare nuclei** (``nucleus_gf_mods : T``, all-electron systems only [18]_):
the drift of an electron at distance :math:`z` from the nearest nucleus is decomposed
into components parallel and perpendicular to :math:`\hat{\mathbf{z}}`, the unit vector
from the nucleus to the electron. The drifted position is

.. math::

    z'' = \max(z + v_z \tau, 0), \qquad
    \mathbf{r}_{drift} = z'' \left(\hat{\mathbf{z}} + \frac{2 \mathbf{v}_\rho \tau}{z + z''}\right) + \mathbf{R}_I

so that the drift never carries an electron through the nucleus. The new position is then
sampled from a mixture of a Gaussian centred at :math:`\mathbf{r}_{drift}` and an exponential
centred at the nucleus:

.. math::

    q = \frac{1}{2}\,\mathrm{erfc}\left(\frac{z + v_z \tau}{\sqrt{2\tau}}\right), \qquad
    (1 - q)\, \mathcal{N}(\mathbf{r}_{drift}, \tau) + q\, \frac{\zeta^3}{\pi} e^{-2\zeta |\mathbf{r} - \mathbf{R}_I|},
    \qquad \zeta = \sqrt{Z^2 + 1/\tau}

which mimics the cusp in the Green's function near the nucleus.

.. _dmc-limiting-velocity:

limiting_velocity
-----------------

Close to the nodal surface the drift velocity diverges, causing large time-step errors.
The velocity is smoothly limited following Umrigar, Nightingale and Runge [18]_:

.. math::

    \bar{\mathbf{v}} = \mathbf{v}\,\frac{-1 + \sqrt{1 + 2 a v^2 \tau}}{a v^2 \tau}

where :math:`v^2 = \sum_i |\mathbf{v}_i|^2` is the total squared drift. For all-electron
systems with ``nucleus_gf_mods : T`` the parameter :math:`a` is position-dependent
(``alimit_vector``):

.. math::

    a(\mathbf{r}_i) = \frac{1 + \hat{\mathbf{v}}_i \cdot \hat{\mathbf{z}}_i}{2}
    + \frac{(Z z_i)^2}{10\,(4 + (Z z_i)^2)}

with :math:`z_i` the distance to the nearest nucleus; otherwise the constant ``alimit``
keyword (default 0.5) is used.

.. _dmc-branching-energy:

branching_energy
----------------

The branching energy of a walker is

.. math::

    S(\mathbf{r}) = (E_T - E_{best}) + \bar E_{cut}\,\frac{|\bar{\mathbf{v}}|}{|\mathbf{v}|}

where the ratio of the limited to the unlimited drift velocity suppresses branching near
the nodal surface [18]_, and the deviation of the local energy from the current best
estimate is limited according to Zen, Sorella, Gillan, Michaelides and Alfè [19]_:

.. math::

    \bar E_{cut} = \mathrm{sign}(E_{best} - E_L)\,
    \min\left(|E_{best} - E_L|,\ 0.2\sqrt{N_e/\tau}\right)

This scheme makes the time-step error size-consistent: the energy cutoff grows with the
number of electrons :math:`N_e`.

.. _dmc-branching:

branching
---------

After the drift-diffusion move each walker carries the weight

.. math::

    W = \exp\left[\tau_{eff}\,\frac{S(\mathbf{r}) + S(\mathbf{r}')}{2}\right]

(for a rejected move :math:`W = \exp[\tau_{eff} S(\mathbf{r})]`). The walker is then
replaced by :math:`M = \lfloor W + u \rfloor` copies, where :math:`u` is a uniform random
number on :math:`[0, 1]`; walkers with :math:`M = 0` are killed.

The reference energy :math:`E_T` is updated after every step to keep the total walker
population close to the target (``dmc_target_weight``) [18]_:

.. math::

    E_T = E_{best} - \ln\left(\frac{W_{tot}}{W_{target}}\right) \frac{\tau}{\tau_{eff}}

where :math:`E_{best}` is the population average of the local energy over all MPI ranks.

.. _dmc-t-move:

t_move
------

The nonlocal pseudopotential energy is evaluated with the localization approximation,
which makes the DMC energy non-variational and sensitive to the quality of :math:`\Psi`.
The T-move scheme of Casula [20]_ (``use_tmove : T``) restores the upper-bound property by
turning the sign-problem-free part of the nonlocal operator into an additional walker move:
after branching, each electron of each walker is offered a heat-bath move to one of the
angular quadrature points :math:`\mathbf{r}_q` (see :ref:`nonlocal_potential <nonlocal_potential>`)
with probability proportional to

.. math::

    -\tau\, V_{nl}(\mathbf{r}_q)\,\frac{\Psi(\mathbf{r}_q)}{\Psi(\mathbf{r})}

restricted to the points where this quantity is positive; with the complementary
probability the walker stays in place. Electrons are processed one after another
(the size-consistent version of the algorithm [21]_).

.. _dmc-load-balancing:

load_balancing
--------------

Branching makes the walker population fluctuate independently on every MPI rank. Every 500
steps the walkers are redistributed pairwise between ranks so that each rank carries the
same number of walkers. The load-balancing efficiency

.. math::

    \eta = \frac{\bar N_{walkers}}{\max_{rank} N_{walkers}}

is accumulated and reported at the end of the run.

.. _dmc-random-step:

random_step
-----------

One DMC step chains the stages described above:

1. :ref:`drift_diffusion <dmc-drift-diffusion>` — propagate and accept/reject every walker;
2. :ref:`branching <dmc-branching>` — duplicate/kill walkers according to their weights;
3. :ref:`t_move <dmc-t-move>` — T-move for pseudopotential systems;
4. update :math:`E_{best}`, :math:`\tau_{eff}` and :math:`E_T` (MPI-reduced over all ranks).

.. _dmc-random-walk:

random_walk
-----------

Runs ``steps`` DMC steps, calling :ref:`load_balancing <dmc-load-balancing>` every 500 steps,
and returns the per-step history of :math:`E_{best}` from which the DMC energy and its
statistical error are estimated by the reblocking analysis.

References
----------

.. [17] P. J. Reynolds, D. M. Ceperley, B. J. Alder, and W. A. Lester,
   *Fixed-node quantum Monte Carlo for molecules*,
   J. Chem. Phys. **77**, 5593 (1982).

.. [18] C. J. Umrigar, M. P. Nightingale, and K. J. Runge,
   *A diffusion Monte Carlo algorithm with very small time-step errors*,
   J. Chem. Phys. **99**, 2865 (1993).

.. [19] A. Zen, S. Sorella, M. J. Gillan, A. Michaelides, and D. Alfè,
   *Boosting the accuracy and speed of quantum Monte Carlo: Size consistency and time step*,
   Phys. Rev. B **93**, 241118(R) (2016).

.. [20] M. Casula,
   *Beyond the locality approximation in the standard diffusion Monte Carlo method*,
   Phys. Rev. B **74**, 161102(R) (2006).

.. [21] M. Casula, S. Moroni, S. Sorella, and C. Filippi,
   *Size-consistent variational approaches to nonlocal pseudopotentials: Standard and lattice
   regularized diffusion Monte Carlo methods revisited*,
   J. Chem. Phys. **132**, 154113 (2010).
