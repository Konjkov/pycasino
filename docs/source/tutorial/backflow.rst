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

    g(r_{iI}) = \left(\frac{r_{iI}}{L_{gI}}\right)^2 \left[6 - 8 \frac{r_{iI}}{L_{gI}} + 3 \left(\frac{r_{iI}}{L_{gI}}\right)^2 \right]

Backflow part of wavefunction is represented by the :class:`casino.Backflow` class.

It must be initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.backflow import Backflow

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    backflow = Backflow(config)

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

    backflow.eta_term(e_powers, e_vectors)


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

    backflow.mu_term(n_powers, n_vectors)

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

    backflow.phi_term(e_powers, n_powers, e_vectors, n_vectors)

eta-term gradient
-----------------

For certain electron coordinates, :math:`\eta` term gradient can be obtained with :py:meth:`casino.Backflow.eta_term_gradient` method::

    backflow.eta_term_gradient(e_powers, e_vectors)

mu-term gradient
----------------

For certain electron coordinates, :math:`\mu` term gradient can be obtained with :py:meth:`casino.Backflow.mu_term_gradient` method::

    backflow.mu_term_gradient(n_powers, n_vectors)


phi-term gradient
-----------------

For certain electron coordinates, :math:`\phi` term gradient can be obtained with :py:meth:`casino.Backflow.phi_term_gradient` method::

    backflow.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

eta-term laplacian
------------------

For certain electron coordinates, :math:`\eta` laplacian term can be obtained with :py:meth:`casino.Backflow.eta_term_laplacian` method::

    backflow.eta_term_laplacian(e_powers, e_vectors)

mu-term laplacian
-----------------

For certain electron coordinates, :math:`\mu` term laplacian can be obtained with :py:meth:`casino.Backflow.mu_term_laplacian` method::

    backflow.mu_term_laplacian(n_powers, n_vectors)

phi-term laplacian
------------------

For certain electron coordinates, :math:`\phi` term laplacian can be obtained with :py:meth:`casino.Backflow.phi_term_laplacian` method::

    backflow.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)
