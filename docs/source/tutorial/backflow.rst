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

Backflow class has a following methods:

eta-term
--------
:math:`\eta(r_{ij})\mathbf{r}_{ij}` consists of a complete power expansion in electron-electron distances :math:`r_{ij}`:

.. math::

    \eta(r_{ij}) = (1 - r_{ij}/L_\eta)^C\Theta(L_\eta - r_{ij})\sum_{k=0}^{N_\eta}c_kr^k_{ij}

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

mu-term
-------
:math:`\mu(r_{iI})\mathbf{r}_{iI}` term consists of a complete power expansion in electron-nucleus distances :math:`r_{iI}`:

.. math::

    \mu(r_{iI}) = (1 - r_{iI}/L_\mu)^C\Theta(L_\mu - r_{iI})\sum_{k=0}^{N_\mu}d_kr^k_{iI}

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

phi-term
--------
:math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` and :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` terms describe two-electron
backflow displacements in terms of :math:`r_{ij}` , :math:`r_{iI}` , and :math:`r_{jI}` and vectors :math:`\mathbf{r}_{ij}` , :math:`\mathbf{r}_{iI}`:

.. math::

    \Phi(r_{iI}, r_{jI}, r_{ij}) = (1 - r_{ij}/L_{\Phi I})^C(1 - r_{iI}/L_{\Phi I})^C\Theta(L_{\Phi I} - r_{ij})\Theta(L_{\Phi I} - r_{iI})
    \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}}\phi_{lmnI}r_{iI}^kr_{jI}^lr_{ij}^m

.. math::

    \Theta(r_{iI}, r_{jI}, r_{ij}) = (1 - r_{ij}/L_{\Phi I})^C(1 - r_{iI}/L_{\Phi I})^C\Theta(L_{\Phi I} - r_{ij})\Theta(L_{\Phi I} - r_{iI})
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

eta-term gradient
-----------------

Considering that vector gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla (f(r)\mathbf{r}) =  \frac{\partial{f(r)}}{\partial{r}} \mathbf{\hat e}_r \otimes \mathbf{r} + f \cdot I

There is only two non-zero terms of :math:`\eta(r_{ij})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_i}(\eta(r_{ij})\mathbf{r}_{ij}) = (1 - r_{ij}/L_\eta)^C\Theta(L_\eta - r_{ij})
    \sum_{k=0}^{N_\eta} (I + \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} (C/(r_{ij} - L_\eta) + k/r_{ij}))c_kr^k_{ij}

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

mu-term gradient
----------------
Considering that vector gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla (f(r)\mathbf{r}) =  \frac{\partial{f(r)}}{\partial{r}} \mathbf{\hat e}_r \otimes \mathbf{r} + f \cdot I

There is only two non-zero terms of :math:`\mu(r_{iI})` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} \mu(r_{iI}) = (1 - r_{iI}/L_\eta)^C\Theta(L_\mu - r_{iI})
    \sum_{k=0}^{N_\mu} (I + \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{iI} (k/r_{iI} - C/(L_\mu - r_{iI})))d_kr^k_{ij}

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

phi-term gradient
-----------------

There is only two non-zero terms of :math:`\Phi(r_{iI}, r_{jI}, r_{ij})` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} \Phi(r_{iI}, r_{jI}, r_{ij}) =

.. math::

    \nabla_{e_i} \Theta(r_{iI}, r_{jI}, r_{ij}) =

For certain electron coordinates, :math:`\phi` term gradient can be obtained with :py:meth:`casino.Backflow.phi_term_gradient` method::

    backflow.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d

eta-term laplacian
------------------

For certain electron coordinates, :math:`\eta` laplacian term can be obtained with :py:meth:`casino.Backflow.eta_term_laplacian` method::

    backflow.eta_term_laplacian(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval

mu-term laplacian
-----------------

For certain electron coordinates, :math:`\mu` term laplacian can be obtained with :py:meth:`casino.Backflow.mu_term_laplacian` method::

    backflow.mu_term_laplacian(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval

phi-term laplacian
------------------

For certain electron coordinates, :math:`\phi` term laplacian can be obtained with :py:meth:`casino.Backflow.phi_term_laplacian` method::

    backflow.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
