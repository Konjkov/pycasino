.. _backflow:

Backflow
========
Backflow displacement is the sum of homogeneous, isotropic electron–electron terms :math:`\eta(r_{ij})`, isotropic electron–nucleus terms
:math:`\mu(r_{iI})` centered on the nuclei, isotropic electron–electron–nucleus terms :math:`\Phi(r_{iI}, r_{jI}, r_{ij})`,
:math:`\Theta(r_{iI}, r_{jI}, r_{ij})`, also centered on the nuclei:

.. math::

    \mathbf{\xi} = \sum_{i \neq j}^{N_e} \eta(r_{ij})\mathbf{r}_{ij} + \sum_{I=1}^{N_I} \mu(r_{iI})\mathbf{r}_{iI} +
    \sum_{I=1}^{N_I}\sum_{i \neq j}^{N_e} \Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}

When all-electron atoms are present, the electron-nucleus Kato cusp conditions cannot be fulfilled unless the backflow displacement at the
nuclear position is zero. This can be obtained by applying smooth cutoffs around such atoms. An artificial multiplicative cutoff function
:math:`g(r_{iI})` is applied to all contributions to the backflow displacement of particle :math:`i` that do not depend on the distance
:math:`r_{iI}` to the all-electron atom :math:`I`. This includes the homogeneous backflow term displacement like :math:`\eta(r_{ij})\mathbf{r}_{ij}`
and the inhomogeneous contributions centered on other atom :math:`J \neq I` like :math:`\Phi(r_{iJ}, r_{jJ}, r_{ij})\mathbf{r}_{ij}` or
:math:`\Theta(r_{iJ}, r_{jJ}, r_{ij})\mathbf{r}_{ij}`.
The :math:`g(r_{iI})` function used is of the form:

.. math::

    g(r_{iI}) = \left(\frac{r_{iI}}{L_{gI}}\right)^2 \left[6 - 8 \left(\frac{r_{iI}}{L_{gI}}\right) + 3 \left(\frac{r_{iI}}{L_{gI}}\right)^2 \right]

Backflow part of wavefunction is represented by the :class:`casino.Backflow` class.

It must be initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.backflow import Backflow

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    backflow = Backflow(config)

.. _intermediate data:

To prevent code duplication, we need to prepare the necessary intermediate data::

    import numpy as np

    neu, ned = config.input.neu, config.input.ned
    ne = neu + ned
    r_e = np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
    atom_positions = config.wfn.atom_positions
    e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
    n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
    e_powers = backflow.ee_powers(e_vectors)
    n_powers = backflow.en_powers(n_vectors)
    r_ij = np.linalg.norm(e_vectors, axis=2)
    r_iI = np.linalg.norm(n_vectors, axis=2)
    ee_spin_mask = np.ones(shape=(ne, ne), dtype=int)
    ee_spin_mask[:neu, :neu] = 0
    ee_spin_mask[neu:, neu:] = 2
    en_spin_mask = np.zeros(shape=(ne,), dtype=int)
    en_spin_mask[neu:] = 1
    C = backflow.trunc

Origin of backflow
------------------

Optimization of a wavefunction within the DMC framework can be formally associated with an effect resembling backflow. While DMC does not modify the
functional form of the trial wavefunction, the projected probability density evolves into a well-defined effective function over configuration space,
constrained by the fixed nodal surface. This formal correspondence suggests that backflow-like correlations are implicitly discovered by the DMC dynamics,
providing a physical motivation for such ansätze as variational encodings of the correlations projection methods reveal. Consider the standard
Slater-Jastrow trial wavefunction:

.. math::

    \Psi = \exp(J(R)) \mathscr A \left( \prod_i^N \psi_i(r_i) \right)

where :math:`R = \{r_1,..., r_N\}` denotes the full configuration, :math:`\mathscr A` is the antisymmetrizer (forming a Slater determinant), and :math:`J(R)` is a symmetric
Jastrow factor ensuring proper cusp conditions.

The local energy is given by:

.. math::

    E_L(R) = - \frac{1}{2} \sum_i \left[ \nabla_i^2 J(R) + \frac{\nabla_i^2 \psi_i(r_i)}{\psi_i(r_i)} + \left(\nabla_i J(R) + \frac{\nabla_i \psi_i(r_i)}{\psi_i(r_i)} \right)^2 \right] + V(R)


Now, consider an improved trial wavefunction — one that has been optimized to achieve the lowest possible energy under the fixed-node constraint, as targeted in DMC calculations:

.. math::

    \Psi_T = \exp(J(R) + U(R)) \mathscr A \left( \prod_i^N \psi_i(r_i) \right)

Here, :math:`U(R)` represents an additional, variational correction to the Jastrow factor, designed to capture more complex correlation effects while preserving the nodal surface defined
by the Slater determinant. Crucially, the single-particle orbitals :math:`\psi_i(r_i)` remain unchanged from the original trial function, and no deformation is applied to their arguments.
As a result, the antisymmetrized product — and hence the nodes of the wavefunction — is identical to that of the original Slater determinant. This form is not the result of DMC projection,
but rather an optimized ansatz whose structure is informed by the correlation physics revealed through DMC simulations, such as the environment-dependent particle responses that go beyond
standard Jastrow descriptions. The local energy of this improved trial function is:

.. math::

    E_T(R) = - \frac{1}{2} \sum_i \left[ \nabla_i^2 (J(R) + U(R)) + \frac{\nabla_i^2 \psi_i(r_i)}{\psi_i(r_i)} + \left(\nabla_i (J(R) + U(R)) + \frac{\nabla_i \psi_i(r_i)}{\psi_i(r_i)} \right)^2 \right] + V(R)

The difference in local energy between the VMC and DMC-like wavefunctions is:

.. math::

    E_L - E_T = \frac{1}{2} \sum_i \left[ \nabla_i^2 U(R) + 2\nabla_i J(R) \nabla_i U(R) + 2\frac{\nabla_i \psi_i(r_i)}{\psi_i(r_i)} \nabla_i U(R) + (\nabla_i U(R))^2\right]

We now introduce a displacement vector field:

.. math::

    \xi_i(R) = - \nabla_i U(R)

This form enables orbital-level optimization — a necessary step toward a more complete and physically accurate description of the many-body wavefunction.
Substituting :math:`\xi_i(R)` into the energy difference, we obtain:

.. math::

    E_L(R) - E_T(R) = - \frac{1}{2} \sum_i \left( \nabla_i \cdot \xi_i + 2\nabla_i J(R) \cdot \xi_i + 2\frac{\nabla_i \psi_i(r_i)}{\psi_i(r_i)} \cdot \xi_i + \nabla_i U(R) \cdot \xi_i \right)

While the exact imaginary-time evolution in DMC combines diffusion, drift, and branching, the branching term alone governs the change in statistical weights of configurations.
Specifically, the weight of a walker at configuration :math:`R` evolves as :math:`w(R, t + dt) = w(R, t) \exp[-dt(E_L(R) - E_T(R))]`. This motivates the use of an effective amplitude
transformation:

.. math::

    \Psi(R, t + dt) = \Psi(R, t) \exp[-dt(E_L(R) - E_T(R))]

where :math:`\Psi` is interpreted not as the physical wavefunction, but as a proxy for the statistical weight in the DMC ensemble. This approximation captures how the branching
process selectively suppresses configurations where the trial wavefunction is energetically unfavorable, effectively reshaping the sampled distribution. This emergent bias mimics
the effect of a coordinate deformation — providing a physical rationale for incorporating such deformations explicitly via backflow in variational ansätze. Using the linear
approximation:

.. math::

    f(r_i + \xi_i dt) = f(r_i) + \nabla f(r_i) \cdot \xi_i dt + O(dt^2)

we shift the coordinates in the wavefunction: :math:`r_i \mapsto r_i + \xi_i dt`. This allows us to write:

.. math::

    \Psi(R, t + dt) = \Psi(R + \xi dt, t) \exp \left[ \frac{1}{2} \sum_i \left( \nabla_i \cdot \xi_i dt + \nabla_i U(R) \cdot \xi_i dt \right) \right] + O(dt^2)

Along this effective trajectory, the term :math:`\sum_i \nabla_i U(R) \cdot \xi_i dt` can be interpreted as the differential :math:`dU(R)`,
and :math:`\sum_i \nabla_i \cdot \xi_i dt` as the differential of the logarithm of the configuration-space volume, :math:`d\ln(V)`. Thus:

.. math::

    \Psi(R, t + dt) = \Psi(R + \xi dt, t) \exp \left[ \frac{d\ln(V(R)) + dU(R)}{2} \right] + O(dt^2)

This shows that the energy-driven weight change in DMC can be formally mapped onto a combined effect of:

- a coordinate transformation (backflow), represented by the shift :math:`r_i \mapsto r_i + \xi_i dt`
- and a Jacobian correction due to the deformation of configuration space, captured by :math:`d\ln(V)`.
- a nonlocal energy correction arising from the variation of the correlation potential :math:`U(R)` along the flow

Integrating from :math:`0` to :math:`\tau` , the finite-time transformation becomes:

.. math::

    \frac{\Psi(R, \tau)}{\rvert\Psi(R, \tau)\rvert} = \exp \left( \frac{U(R + \tilde \xi(\tau)) - U(R)}{2} \right) \frac{\Psi(R + \tilde \xi)}{\rvert\Psi(R + \tilde \xi)\rvert}

where :math:`\tilde \xi` is the integrated backflow displacement: :math:`\tilde \xi = \int_{0}^{\tau} \xi(R(s)) \, ds`

A common situation is when optimization starts with a pure Slater wave function, then :math:`J(R) \equiv 0`

Summary of Methods
------------------

Backflow class has a following methods:

.. list-table::
   :widths: 30 40 30
   :header-rows: 1
   :width: 100%

   * - Method
     - Output
     - Shape
   * - :ref:`eta_term <eta-term>`
     - :math:`\eta(r_{ij})\mathbf{r}_{ij}`
     - :math:`(3N_e,)`
   * - :ref:`mu_term <mu-term>`
     - :math:`\mu(r_{ij})\mathbf{r}_{ij}`
     - :math:`(3N_e,)`
   * - :ref:`phi_term <phi-term>`
     - :math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}`
     - :math:`(3N_e,)`
   * - :ref:`eta_term_gradient <eta-term-gradient>`
     - :math:`\nabla \eta(r_{ij})\mathbf{r}_{ij}`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`mu_term_gradient <mu-term-gradient>`
     - :math:`\nabla \mu(r_{ij})\mathbf{r}_{ij}`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`phi_term_gradient <phi-term-gradient>`
     - :math:`\nabla (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI})`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`eta_term_laplacian <eta-term-laplacian>`
     - :math:`\Delta \eta(r_{ij})\mathbf{r}_{ij}`
     - :math:`(3N_e,)`
   * - :ref:`mu_term_laplacian <mu-term-laplacian>`
     - :math:`\Delta \mu(r_{ij})\mathbf{r}_{ij}`
     - :math:`(3N_e,)`
   * - :ref:`phi_term_laplacian <phi-term-laplacian>`
     - :math:`\Delta (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI})`
     - :math:`(3N_e,)`

.. _eta-term:

eta-term
--------
:math:`\eta(r_{ij})\mathbf{r}_{ij}` consists of a complete power expansion in electron-electron distances :math:`r_{ij}`:

.. math::

    \eta(r_{ij}) = (1 - r_{ij}/L_\eta)^C\Theta(L_\eta - r_{ij}) \sum_{k=0}^{N_\eta}c_kr^k_{ij}

where :math:`\Theta` is the Heaviside function. Electron-electron Kato cusp conditions at :math:`r_{ij} = 0` satisfied by constraint
for spin-like electrons only:

.. math::

    L_\eta c_1 = C c_0

For certain electron coordinates, :math:`\eta` term can be obtained with :py:meth:`casino.Backflow.eta_term` method::

    backflow.eta_term(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_ij, backflow.eta_parameters.T)
    cutoff = np.maximum(1 - r_ij / backflow.eta_cutoff[0], 0) ** C
    np.fill_diagonal(cutoff, 0)
    eta = np.expand_dims(cutoff * np.choose(ee_spin_mask, poly, mode='wrap'), -1)
    np.sum(-e_vectors * eta, axis=0)

.. _mu-term:

mu-term
-------
:math:`\mu(r_{iI})\mathbf{r}_{iI}` term consists of a complete power expansion in electron-nucleus distances :math:`r_{iI}`:

.. math::

    \mu(r_{iI}) = (1 - r_{iI}/L_\mu)^C\Theta(L_\mu - r_{iI}) \sum_{k=0}^{N_\mu}d_kr^k_{iI}

where :math:`\Theta` is the Heaviside function. The electron-nucleus Kato cusp conditions at :math:`r_{iI} = 0` satisfied if

.. math::

    L_{\mu I} d_{1 I} = C d_{0 I}

for all atoms, and in addition,

.. math::

    d_{0 I} = 0

for all-electron atoms.

For certain electron coordinates, :math:`\mu` term can be obtained with :py:meth:`casino.Backflow.mu_term` method::

    backflow.mu_term(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_iI, backflow.mu_parameters[0].T)
    cutoff = np.maximum(1 - r_iI / backflow.mu_cutoff, 0) ** C
    n_vectors * np.expand_dims(cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap'), -1)

.. _phi-term:

phi-term
--------
:math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` and :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` terms describe two-electron
backflow displacements in terms of :math:`r_{ij}` , :math:`r_{iI}` , and :math:`r_{jI}` and vectors :math:`\mathbf{r}_{ij}` , :math:`\mathbf{r}_{iI}`:

.. math::

    \Phi(r_{iI}, r_{jI}, r_{ij}) = (1 - r_{iI}/L_{\Phi I})^C(1 - r_{jI}/L_{\Phi I})^C\Theta(L_{\Phi I} - r_{iI})\Theta(L_{\Phi I} - r_{jI})
    \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}}\phi_{lmnI}r_{iI}^kr_{jI}^lr_{ij}^m

.. math::

    \Theta(r_{iI}, r_{jI}, r_{ij}) = (1 - r_{iI}/L_{\Phi I})^C(1 - r_{jI}/L_{\Phi I})^C\Theta(L_{\Phi I} - r_{iI})\Theta(L_{\Phi I} - r_{jI})
    \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}}\theta_{lmnI}r_{iI}^kr_{jI}^lr_{ij}^m

where :math:`\Theta` is the Heaviside function. To ensure electron–electron Kato cusp conditions folowing :math:`3(N_{\Phi I}^{ee} + N_{\Phi I}^{en} + 1)`
constraints is applied:

.. math::

    \sum_{l,m}^{l+m=\alpha}(C\phi_{0lmI} - L_{\phi I}\phi_{1lmI}) = \sum_{k,m}^{k+m=\alpha}(C\phi_{k0mI} - L_{\phi I}\phi_{k1mI}) =
    \sum_{k,m}^{k+m=\alpha}(C\theta_{k0mI} - L_{\phi I}\theta_{k1mI}) = 0

another :math:`2N_{\Phi I}^{en} + 1` constraints from the electron-electron Kato cusp conditions:

.. math::

    \sum_{k,l}^{k+l=\alpha}\theta_{kl1I} = 0

and extra :math:`2N_{\Phi I}^{en} + 1` constraints for spin-like electrons:

.. math::

    \sum_{k,l}^{k+l=\alpha}\phi_{kl1I} = 0

for all-electron atoms there are :math:`4(N_{\Phi I}^{ee} + N_{\Phi I}^{en})+2` constraints on :math:`\phi_{klm}`

.. math::

    \sum_{l,m}^{l+m=\alpha}\phi_{0lmI} = \sum_{l,m}^{l+m=\alpha}m\phi_{0lmI} = \sum_{k,m}^{k+m=\alpha}\phi_{k0mI} = \sum_{k,m}^{k+m=\alpha}m\phi_{k0mI} = 0

for all-electron atoms there are :math:`3(N_{\Phi I}^{ee} + N_{\Phi I}^{en})+2` constraints on :math:`\theta_{klm}`

.. math::

    \sum_{l,m}^{l+m=\alpha}\theta_{0lmI} = \sum_{l,m}^{l+m=\alpha}m\theta_{0lmI} = \sum_{k,m}^{k+m=\alpha}m\theta_{k0mI} = 0

For certain electron coordinates, :math:`\Phi` and :math:`\Theta` terms can be obtained with :py:meth:`casino.Backflow.phi_term` method::

    backflow.phi_term(e_powers, n_powers, e_vectors, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
    r_ijI = np.tile(r_iI[0], (ne, 1))
    cutoff = np.maximum(1 - r_iI/backflow.phi_cutoff, 0) ** C
    phi_poly = polyval3d(r_ijI, r_ijI.T, r_ij, backflow.phi_parameters[0].T)
    theta_poly = polyval3d(r_ijI, r_ijI.T, r_ij, backflow.theta_parameters[0].T)
    phi = np.outer(cutoff[0], cutoff[0]) * np.choose(ee_spin_mask, phi_poly, mode='wrap')
    theta = np.outer(cutoff[0], cutoff[0]) * np.choose(ee_spin_mask, theta_poly, mode='wrap')
    np.fill_diagonal(theta, 0)
    np.sum(-e_vectors * np.expand_dims(phi, -1) + n_vectors * np.expand_dims(theta, -1), axis=0)

.. _eta-term-gradient:

eta-term gradient
-----------------

Considering that vector gradient of spherically symmetric vector function (in 3-D space) is:

.. math::

    \nabla (f(r)\mathbf{r}) = f'(r) \mathbf{\hat r} \otimes \mathbf{r} + f \cdot \mathbf{I}

There is only two non-zero terms of :math:`\eta(r_{ij})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_i} (\eta(r_{ij})\mathbf{r}_{ij}) = (1 - r_{ij}/L_\eta)^C\Theta(L_\eta - r_{ij})
    \sum_{k=0}^{N_\eta} \left[\left(\frac{k}{r_{ij}} - \frac{C}{L_\eta - r_{ij}}\right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} + \mathbf{I} \right] c_kr^k_{ij}

.. math::

    \nabla_{e_j} (\eta(r_{ij})\mathbf{r}_{ij}) = - \nabla_{e_i} (\eta(r_{ij})\mathbf{r}_{ij})

where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`

For certain electron coordinates, :math:`\eta` term gradient can be obtained with :py:meth:`casino.Backflow.eta_term_gradient` method::

    backflow.eta_term_gradient(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = backflow.eta_cutoff
    k = np.arange(backflow.eta_parameters.shape[1])
    cutoff = np.maximum(1 - r_ij / backflow.eta_cutoff[0], 0) ** C
    np.fill_diagonal(cutoff, 0)
    poly = polyval(r_ij, backflow.eta_parameters.T)
    poly_k = polyval(r_ij, (k * backflow.eta_parameters).T)
    unit_e_vectors = np.nan_to_num(e_vectors/np.expand_dims(r_ij, -1))
    t1 = cutoff * np.choose(ee_spin_mask, poly, mode='wrap')
    t2 = cutoff * np.choose(ee_spin_mask, poly * C / (r_ij - L) + poly_k / r_ij, mode='wrap')
    tt1 = np.einsum('ij,ab->aibj', np.eye(3), np.diag(np.sum(t1, axis=0)) - t1)
    np.fill_diagonal(t2, 0)
    tt2 = np.einsum('abi,abj,ab->aibj', e_vectors, unit_e_vectors, -t2)
    (tt1 + tt2).reshape(ne*3, ne*3)

.. _mu-term-gradient:

mu-term gradient
----------------
Considering that vector gradient of spherically symmetric vector function (in 3-D space) is:

.. math::

    \nabla (f(r)\mathbf{r}) = f'(r) \mathbf{\hat r} \otimes \mathbf{r} + f \cdot \mathbf{I}

There is only one non-zero term of :math:`\mu(r_{iI})` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} (\mu(r_{iI})\mathbf{r}_{iI}) = (1 - r_{iI}/L_\mu)^C\Theta(L_\mu - r_{iI})
    \sum_{k=0}^{N_\mu} \left[\left(\frac{k}{r_{iI}} - \frac{C}{L_\mu - r_{iI}}\right) \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{iI} + \mathbf{I}\right] d_kr^k_{ij}

where :math:`\mathbf{\hat r}_{iI}` is the unit vector in the direction of the :math:`\mathbf{r}_{iI}`

For certain electron coordinates, :math:`\mu` term gradient can be obtained with :py:meth:`casino.Backflow.mu_term_gradient` method::

    backflow.mu_term_gradient(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = backflow.mu_cutoff
    k = np.arange(backflow.mu_parameters[0].shape[1])
    cutoff = np.maximum(1 - r_iI / L, 0) ** C
    poly = polyval(r_iI, backflow.mu_parameters[0].T)
    poly_k = polyval(r_iI, (k * backflow.mu_parameters[0]).T)
    unit_n_vectors = n_vectors/np.expand_dims(r_iI, -1)
    t1 = cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap')
    t2 = cutoff[0] * np.choose(en_spin_mask, poly * C / (r_iI - L) + poly_k / r_iI, mode='wrap')
    tt1 = np.einsum('ij,ab,Ia->aibj', np.eye(3), np.eye(ne), t1)
    tt2 = np.einsum('Iai,Iaj,ab,Ia->aibj', n_vectors, unit_n_vectors, np.eye(ne), t2)
    (tt1 + tt2).reshape(ne*3, ne*3)

.. _phi-term-gradient:

phi-term gradient
-----------------

Considering that vector gradient of spherically symmetric vector function (in 3-D space) is:

.. math::

    \nabla (f(r)\mathbf{r}) = f'(r) \mathbf{\hat r} \otimes \mathbf{r} + f \cdot \mathbf{I}

There is only two non-zero terms of :math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` gradient, i.e. by :math:`i`-th:

.. math::

    \begin{align}
    & \nabla_{e_i} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = (1 - r_{iI}/L_{\Phi I})^C (1 - r_{jI}/L_{\Phi I})^C \Theta(L_{\Phi I} - r_{iI}) \Theta(L_{\Phi I} - r_{jI}) \\
    &  \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{k}{r_{iI}} - \frac{C}{L_{\Phi I} - r_{iI}} \right) \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{ij} + \left(\frac{m}{r_{ij}} \right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} + \mathbf{I} \right] \phi_{lmnI} r_{iI}^k r_{jI}^l r_{ij}^m\\
    \end{align}

or :math:`j`-th electron coordinates:

.. math::

    \begin{align}
    & \nabla_{e_j} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = (1 - r_{iI}/L_{\Phi I})^C (1 - r_{jI}/L_{\Phi I})^C \Theta(L_{\Phi I} - r_{iI}) \Theta(L_{\Phi I} - r_{jI}) \\
    &  \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}} \right) \mathbf{\hat r}_{jI} \otimes \mathbf{r}_{ij} - \left(\frac{m}{r_{ij}} \right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} - \mathbf{I} \right] \phi_{lmnI} r_{iI}^k r_{jI}^l r_{ij}^m\\
    \end{align}

There is only two non-zero terms of :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` gradient, i.e. by :math:`i`-th:

.. math::

    \begin{align}
    & \nabla_{e_i} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = (1 - r_{iI}/L_{\Phi I})^C (1 - r_{jI}/L_{\Phi I})^C \Theta(L_{\Phi I} - r_{iI}) \Theta(L_{\Phi I} - r_{jI}) \\
    & \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{k}{r_{iI}} -\frac{C}{L_{\Phi I} - r_{iI}}\right) \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{iI} + \left(\frac{m}{r_{ij}}\right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{iI} + \mathbf{I} \right]  \theta_{lmnI} r_{iI}^k r_{jI}^l r_{ij}^m\\
    \end{align}

or :math:`j`-th electron coordinates:

.. math::

    \begin{align}
    & \nabla_{e_j} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = (1 - r_{iI}/L_{\Phi I})^C (1 - r_{jI}/L_{\Phi I})^C \Theta(L_{\Phi I} - r_{iI}) \Theta(L_{\Phi I} - r_{jI}) \\
    & \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}}\right) \mathbf{\hat r}_{jI} \otimes \mathbf{r}_{iI} - \left(\frac{m}{r_{ij}}\right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{iI} \right]  \theta_{lmnI} r_{iI}^k r_{jI}^l r_{ij}^m\\
    \end{align}

where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`
and :math:`\mathbf{\hat r}_{iI}` is the unit vector in the direction of the :math:`\mathbf{r}_{iI}`

For certain electron coordinates, :math:`\phi` term gradient can be obtained with :py:meth:`casino.Backflow.phi_term_gradient` method::

    backflow.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d

.. _eta-term-laplacian:

eta-term laplacian
------------------

Considering that vector laplacian of spherically symmetric vector function (in 3-D space) is:

.. math::

    \Delta (f(r)\mathbf{r}) = \left(f''(r) + \frac{4}{r} f'(r)\right) \mathbf{r}

There is only two non-zero terms of :math:`\eta(r_{ij})\mathbf{r}_{ij}` laplacian, i.e. by :math:`i`-th  or :math:`j`-th electron coordinates:

.. math::

    \Delta_{e_i} (\eta(r_{ij})\mathbf{r}_{ij}) = (1 - r_{ij}/L_\eta)^C\Theta(L_\eta - r_{ij}) \mathbf{r}_{ij}\sum_{k=0}^{N_\eta} \left[\frac{C(C-1)}{(L_\eta - r_{ij})^2} - \frac{2C(k+2)}{r_{ij}(L_\eta - r_{ij})} + \frac{k(k+3)}{r_{ij}^2} \right] c_kr^k_{ij}

.. math::

    \Delta_{e_j} (\eta(r_{ij})\mathbf{r}_{ij}) = - \Delta_{e_i} (\eta(r_{ij})\mathbf{r}_{ij})

For certain electron coordinates, :math:`\eta` laplacian term can be obtained with :py:meth:`casino.Backflow.eta_term_laplacian` method::

    backflow.eta_term_laplacian(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval

.. _mu-term-laplacian:

mu-term laplacian
-----------------

Considering that vector laplacian of spherically symmetric vector function (in 3-D space) is:

.. math::

    \Delta (f(r)\mathbf{r}) = \left(f''(r) + \frac{4}{r} f'(r)\right) \mathbf{r}

There is only one non-zero term of :math:`\mu(r_{iI})\mathbf{r}_{iI}` laplacian, i.e. by :math:`i`-th electron coordinates:

.. math::

    \Delta_{e_i} (\mu(r_{iI})\mathbf{r}_{iI}) = (1 - r_{iI}/L_\mu)^C\Theta(L_\mu - r_{iI}) \mathbf{r}_{iI}\sum_{k=0}^{N_\mu} \left[\frac{C(C-1)}{(L_\mu - r_{iI})^2} - \frac{2C(k+2)}{r_{iI}(L_\mu - r_{iI})} + \frac{k(k+3)}{r_{iI}^2} \right]d_kr^k_{iI}

For certain electron coordinates, :math:`\mu` term laplacian can be obtained with :py:meth:`casino.Backflow.mu_term_laplacian` method::

    backflow.mu_term_laplacian(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval

.. _phi-term-laplacian:

phi-term laplacian
------------------

Considering that gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla f(r) = f'(r) \mathbf{\hat r}

and laplacian of spherically symmetric vector function (in 3-D space) is:

.. math::

    \Delta f(r) = f''(r) + \frac{2}{r} f'(r)

and :math:`\Phi` term addent is a product of constant :math:`\phi_{lmnI}r_{jI}^l` and three spherically symmetric functions :math:`f(r_{ij})`, :math:`g(r_{iI})`, :math:`\mathbf{r}_{ij}` so using:

.. math::

    \Delta (f(r_{ij})g(r_{iI})\mathbf{r}_{ij}) = \left(g\Delta f + 2\nabla f\nabla g + f\Delta g\right)r_{ij} + 2(g\nabla f + f\nabla g)

There is only two non-zero terms of :math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` laplacian, i.e. by :math:`i`-th  or :math:`j`-th electron coordinates:

.. math::

    \Delta_{e_i} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = (1 - r_{iI}/L_{\Phi I})^C (1 - r_{jI}/L_{\Phi I})^C \Theta(L_{\Phi I} - r_{iI}) \Theta(L_{\Phi I} - r_{jI})

There is only two non-zero terms of :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` laplacian, i.e. by :math:`i`-th  or :math:`j`-th electron coordinates:

.. math::

    \Delta_{e_i} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = (1 - r_{iI}/L_{\Phi I})^C (1 - r_{jI}/L_{\Phi I})^C \Theta(L_{\Phi I} - r_{iI}) \Theta(L_{\Phi I} - r_{jI})

For certain electron coordinates, :math:`\phi` term laplacian can be obtained with :py:meth:`casino.Backflow.phi_term_laplacian` method::

    backflow.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
