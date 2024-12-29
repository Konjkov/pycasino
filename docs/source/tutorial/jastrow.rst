.. _jastrow:

Jastrow
=======

Jastrow part of wavefunction is represented by the :class:`casino.Jastrow` class.

It must be initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.jastrow import Jastrow

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    jastrow = Jastrow(config)

.. _intermediate data:

To prevent code duplication, we need to prepare the necessary intermediate data::

    import numpy as np

    neu, ned = config.input.neu, config.input.ned
    ne = neu + ned
    r_e = np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
    atom_positions = config.wfn.atom_positions
    e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
    n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
    e_powers = jastrow.ee_powers(e_vectors)
    n_powers = jastrow.en_powers(n_vectors)
    r_ij = np.linalg.norm(e_vectors, axis=2)
    r_iI = np.linalg.norm(n_vectors, axis=2)
    ee_spin_mask = np.ones(shape=(ne, ne), dtype=int)
    ee_spin_mask[:neu, :neu] = 0
    ee_spin_mask[neu:, neu:] = 2
    en_spin_mask = np.zeros(shape=(ne,), dtype=int)
    en_spin_mask[neu:] = 1
    C = jastrow.trunc


Jastrow class has a following methods:

u_term
------

:math:`u(r_{ij})` term consists of a complete power expansion in electron-electron distances :math:`r_{ij}`

.. math::

    u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij})\sum_{l=0}^{N_u}\alpha_lr^l_{ij}

where :math:`\Theta` is the Heaviside function. This term goes to zero at the cutoff length :math:`L_u` with :math:`C - 1` continuous derivatives at.
In this expression C determines the behavior at the cutoff length. If :math:`C = 2`, the gradient of this term is continuous but the second derivative
and hence the local energy is discontinuous, and if :math:`C = 3` then both the gradient of this term and the local energy are continuous.
Electron-electron Kato cusp conditions at :math:`r_{ij} = 0` satisfied by constraint:

.. math::

    \alpha_1 = \left[\frac{\Gamma_{ij}}{(-L_u)^C} + \frac{\alpha_0C}{L_u}\right]

where :math:`\Gamma_{ij} = 1/2` if electrons :math:`i` and :math:`j` have opposite spins and :math:`\Gamma_{ij} = 1/4` if :math:`i` and :math:`j` have
the same spin.
For certain electron coordinates, :math:`u` term can be obtained with :py:meth:`casino.Jastrow.u_term` method::

    jastrow.u_term(e_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_ij, jastrow.u_parameters.T)
    cutoff = np.minimum(r_ij - jastrow.u_cutoff, 0) ** C
    np.sum(np.triu(cutoff * np.choose(ee_spin_mask, poly, mode='wrap'), 1))


chi_term
--------

:math:`\chi(r_{iI})` term consists of a complete power expansion in electron-nucleus distances :math:`r_{iI}`

.. math::

    \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI})\sum_{m=0}^{N_\chi}\beta_mr^m_{iI}

where :math:`\Theta` is the Heaviside function. This term goes to zero at the cutoff length :math:`L_{\chi I}`.
The term involving the ionic charge :math:`Z_I` enforces the electron–nucleus cusp condition:

.. math::

    \beta_1 = \left[\frac{-Z_I}{(-L_{\chi I})^C} + \frac{\beta_{0I}C}{L_{\chi I}}\right]

if the Slater part of the wave function satisfies the electron–nucleus cusp condition, or if a non-divergent
pseudopotential is used, then the Jastrow factor is required to be cuspless at the nuclei, i.e it should satisfy
the above equation with :math:`Z_I = 0`

For certain electron coordinates, :math:`\chi` term can be obtained with :py:meth:`casino.Jastrow.chi_term` method::

    jastrow.chi_term(n_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_iI, jastrow.chi_parameters[0].T)
    cutoff = np.minimum(r_iI - jastrow.chi_cutoff, 0) ** C
    np.sum(cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap'))


f_term
------

:math:`f(r_{ij}, r_{iI}, r_{jI})` term is the most general expansion of a function of :math:`r_{ij}` , :math:`r_{iI}` , and :math:`r_{jI}`
that is cuspless at the coalescence point and goes smoothly to zero when either :math:`r_{iI}` or :math:`r_{jI}` reach cutoff lengths:

.. math::

    f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI})
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

To ensure no electron–electron cusps folowing :math:`2N_{fI}^{eN} + 1` conditions is applied:

.. math::

    \sum_{l,m \ : \ l+m=k}\gamma_{lm1I} = 0

and to ensure no electron–nucleus cusps folowing :math:`N_{fI}^{eN} + N_{fI}^{ee} + 1` conditions is applied:

.. math::

    \sum_{l,m \ : \ l+m=k'}(C\gamma_{0mnI} - L_{fI}\gamma_{1mnI}) = 0

If desired, there are :math:`N_{fI}^{ee}` constraints imposed to prevent duplication of :math:`u` term :math:`(γ_{00nI} = 0 \ \forall n)`
and there are :math:`N_{fI}^{eI}` constraints imposed to prevent duplication of :math:`\chi` term :math:`(γ_{l00I} = 0 \ \forall l)`
also the Jastrow factor to be symmetric under electron exchanges it is required that :math:`\gamma_{lmnI} = \gamma_{mlnI} \ \forall I, m, l, n`.

For certain electron coordinates, :math:`f` term can be obtained with :py:meth:`casino.Jastrow.f_term` method::

    jastrow.f_term(e_powers, n_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
    r_ijI = np.tile(r_iI[0], (ne, 1))
    cutoff = np.minimum(r_iI - jastrow.f_cutoff, 0) ** C
    poly = polyval3d(r_ijI, r_ijI.T, r_ij, jastrow.f_parameters[0].T)
    np.sum(np.triu(np.outer(cutoff[0], cutoff[0]) * np.choose(ee_spin_mask, poly, mode='wrap'), 1))


u_term_gradient
---------------

Considering that gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla f =  \frac{\partial{f}}{\partial{r}} \mathbf{\hat e}_r

There is only two non-zero terms of :math:`u(r_{ij})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_i} u(r_{ij}) = -\nabla_{e_j} u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij})\mathbf{\hat r}_{ij}\sum_{l=0}^{N_u}(C/(r_{ij} - L_u) + l/r_{ij})\alpha_lr^l_{ij}

where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`

For certain electron coordinates, :math:`u` gradient term can be obtained with :py:meth:`casino.Jastrow.u_term_gradient` method::

    jastrow.u_term_gradient(e_powers, e_vectors)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    import numpy as np
    from numpy.polynomial.polynomial import polyval
    L = jastrow.u_cutoff
    l = np.arange(jastrow.u_parameters.shape[1])
    cutoff = np.minimum(r_ij - L, 0) ** C
    poly = polyval(r_ij, jastrow.u_parameters.T) * C / (r_ij - L)
    poly += polyval(r_ij, (l * jastrow.u_parameters).T) / r_ij
    g_ij = np.nan_to_num(cutoff * np.choose(ee_spin_mask, poly, mode='wrap') * e_vectors.T / r_ij)
    np.sum(g_ij, axis=1).T.ravel()


chi_term_gradient
-----------------

There is only one non-zero term of :math:`\chi(r_{iI})` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI})\mathbf{\hat r}_{iI}\sum_{m=0}^{N_\chi}(C/(r_{iI} - L_{\chi I}) + m/r_{iI})\beta_mr^m_{iI}

where :math:`\mathbf{\hat r}_{iI}` is the unit vector in the direction of the :math:`\mathbf{r}_{iI}`

For certain electron coordinates, :math:`\chi` term gradient can be obtained with :py:meth:`casino.Jastrow.chi_term_gradient` method::

    jastrow.chi_term_gradient(n_powers, n_vectors)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = jastrow.chi_cutoff
    cutoff = np.minimum(r_iI - L, 0) ** C
    r_iI = np.linalg.norm(n_vectors, axis=2)
    m = np.arange(jastrow.chi_parameters[0].shape[1])
    poly = polyval(r_iI, jastrow.chi_parameters[0].T) * C / (r_iI[0] - L[0])
    poly += polyval(r_iI, (m * jastrow.chi_parameters[0]).T) / r_iI[0]
    (cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap') * n_vectors[0].T / r_iI[0]).T.ravel()


f_term_gradient
---------------

There is only two non-zero terms of :math:`f(r_{ij}, r_{iI}, r_{jI})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    g_{ij} =  \mathbf{\hat r}_{ij} \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}(n/r_{ij})\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    g_{iI} = \mathbf{\hat r}_{iI} \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}(C/(r_{iI} - L_{fI}) + l / r_{iI})\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    g_{jI} = \mathbf{\hat r}_{jI} \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}(C/(r_{jI} - L_{fI}) + m / r_{jI})\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    \nabla_{e_i} f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI})(g_{iI} + g_{ij})

.. math::

    \nabla_{e_j} f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI})(g_{jI} - g_{ij})

For certain electron coordinates, :math:`f` term gradient can be obtained with :py:meth:`casino.Jastrow.f_term_gradient` method::

    jastrow.f_term_gradient(e_powers, n_powers, e_vectors, n_vectors)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
    n = np.expand_dims(np.arange(jastrow.f_parameters[0].shape[1]), axis=(1, 2))
    m = np.expand_dims(np.arange(jastrow.f_parameters[0].shape[2]), axis=1)
    l = np.arange(jastrow.f_parameters[0].shape[3])
    L = jastrow.f_cutoff
    cutoff = np.minimum(r_iI - L, 0) ** C
    r_ijI = np.tile(r_iI[0], (ne, 1))
    poly = polyval3d(r_ijI, r_ijI.T, r_ij, jastrow.f_parameters[0].T)
    poly_l = polyval3d(r_ijI, r_ijI.T, r_ij, (l * jastrow.f_parameters[0]).T)
    poly_m = polyval3d(r_ijI, r_ijI.T, r_ij, (m * jastrow.f_parameters[0]).T)
    poly_n = polyval3d(r_ijI, r_ijI.T, r_ij, (n * jastrow.f_parameters[0]).T)

    g_ijI = np.choose(ee_spin_mask, poly, mode='wrap') * C / (r_iI[0] - L[0])
    g_ijI += np.choose(ee_spin_mask, poly_l, mode='wrap') / r_iI[0]
    g_ijI = np.triu(g_ijI, 1) * np.expand_dims(n_vectors[0].T / r_iI[0], 1)

    g_jiI = np.choose(ee_spin_mask, poly, mode='wrap').T * C / (r_iI[0] - L[0])
    g_jiI += np.choose(ee_spin_mask, poly_m, mode='wrap').T / r_iI[0]
    g_jiI = np.tril(g_jiI, -1) * np.expand_dims(n_vectors[0].T / r_iI[0], 1)

    g_ij = np.nan_to_num(np.choose(ee_spin_mask, poly_n / r_ij, mode='wrap') * e_vectors.T / r_ij)

    np.sum(np.outer(cutoff[0], cutoff[0]) * (g_ijI + g_jiI + g_ij), axis=1).T


u_term_laplacian
----------------

Considering that Laplace operator of spherically symmetric function (in 3-D space) is:

.. math::

    \Delta f = \frac{\partial^2{f}}{\partial{r^2}} + \frac{2}{r} \frac{\partial{f}}{\partial{r}}

then :math:`u(r_{ij})` term laplacian:

.. math::

    \Delta u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij}) \times

.. math::

    \sum_{l=0}^{N_u}(C(C-1)/(r_{ij} - L_u)^2 + 2C(l+1)/r_{ij}(r_{ij} - L_u) + l(l+1)/r_{ij}^2)\alpha_lr^l_{ij}

For certain electron coordinates, :math:`u` term laplacian can be obtained with :py:meth:`casino.Jastrow.u_term_laplacian` method::

    jastrow.u_term_laplacian(e_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = jastrow.u_cutoff
    l = np.arange(jastrow.u_parameters.shape[1])
    l_1 = np.arange(1, jastrow.u_parameters.shape[1] + 1)
    cutoff = np.minimum(r_ij - jastrow.u_cutoff, 0) ** C
    poly =  polyval(r_ij, jastrow.u_parameters.T) * C * (C - 1) / (r_ij - L) ** 2
    poly += 2 * polyval(r_ij, (l_1 * jastrow.u_parameters).T) * C / r_ij / (r_ij - L)
    poly += polyval(r_ij, (l * l_1 * jastrow.u_parameters).T) / r_ij ** 2
    np.sum(np.triu(cutoff * np.choose(ee_spin_mask, poly, mode='wrap'), 1))


chi_term_laplacian
------------------

Considering that Laplace operator of spherically symmetric function (in 3-D space) is:

.. math::

    \Delta f = \frac{\partial^2{f}}{\partial{r^2}} + \frac{2}{r} \frac{\partial{f}}{\partial{r}}

then :math:`\chi(r_{iI})` term laplacian:

.. math::

    \Delta \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI}) \times

.. math::

    \sum_{l=0}^{N_\chi}(C(C-1)/(r_{iI} - L_{\chi I})^2 + 2C(m+1)/r_{iI}(r_{iI} - L_{\chi I}) + m(m+1)/r_{iI}^2)\beta_mr^m_{iI}

For certain electron coordinates, :math:`\chi` term laplacian can be obtained with :py:meth:`casino.Jastrow.chi_term_laplacian` method::

    jastrow.chi_term_laplacian(n_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = jastrow.chi_cutoff
    m = np.arange(jastrow.chi_parameters[0].shape[1])
    m_1 = np.arange(1, jastrow.chi_parameters[0].shape[1] + 1)
    cutoff = np.minimum(r_iI - L, 0) ** C
    poly = polyval(r_iI, jastrow.chi_parameters[0].T) * C * (C - 1) / (r_iI[0] - L[0]) ** 2
    poly += 2 * polyval(r_iI, (m_1 * jastrow.chi_parameters[0]).T) * С / r_iI / (r_iI[0] - L[0])
    poly += polyval(r_iI, (m + m_1 * jastrow.chi_parameters[0]).T) / r_iI ** 2
    np.sum(cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap'))


f_term_laplacian
----------------

Considering that Laplace operator of spherically symmetric function (in 3-D space) is:

.. math::

    \Delta f = \frac{\partial^2{f}}{\partial{r^2}} + \frac{2}{r} \frac{\partial{f}}{\partial{r}}

and :math:`f` term is a product of two spherically symmetric functions :math:`f(r_{iI})` and :math:`g(r_{ij})` so using:

.. math::

    \Delta_{e_i}(fg) = g \Delta_{e_i}f + 2 \nabla_{e_i}f \nabla_{e_i}g + f \Delta_{e_i}g


then :math:`f(r_{ij}, r_{iI}, r_{jI})` term laplacian:

.. math::

    l_1 = \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    (C/r_{iI}(r_{iI} - L_{fI}) + l/r_{iI}^2) \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    l_2 = \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    (n(n+1)/r_{ij}^2) \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    l_{dot} = \mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{iI}
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    (n/r_{ij}) (C/(r_{iI} - L_{fI}) + l/r_{iI}) \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n


.. math::

    \Delta f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI}) (l_1 + 2l_{dot} + l_2)

For certain electron coordinates, :math:`f` term laplacian can be obtained with :py:meth:`casino.Jastrow.f_term_laplacian` method::

    jastrow.f_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
    n = np.expand_dims(np.arange(jastrow.f_parameters[0].shape[1]), axis=(1, 2))
    m = np.expand_dims(np.arange(jastrow.f_parameters[0].shape[2]), axis=1)
    l = np.arange(jastrow.f_parameters[0].shape[3])
    L = jastrow.f_cutoff
    cutoff = np.minimum(r_iI - L, 0) ** C
    r_ijI = np.tile(r_iI[0], (ne, 1))
    poly = polyval3d(r_ijI, r_ijI.T, r_ij, jastrow.f_parameters[0].T)
    poly_l = polyval3d(r_ijI, r_ijI.T, r_ij, (l * jastrow.f_parameters[0]).T)
    poly_m = polyval3d(r_ijI, r_ijI.T, r_ij, (m * jastrow.f_parameters[0]).T)
    poly_n = polyval3d(r_ijI, r_ijI.T, r_ij, (n * jastrow.f_parameters[0]).T)
    poly_lm = polyval3d(r_ijI, r_ijI.T, r_ij, (l * m * jastrow.f_parameters[0]).T)
    poly_ln = polyval3d(r_ijI, r_ijI.T, r_ij, (l * n * jastrow.f_parameters[0]).T)

    l_1 = 0

    l_dot = np.choose(ee_spin_mask, poly_n, mode='wrap') * C / (r_iI[0] - L[0]) / r_ij
    l_dot += np.choose(ee_spin_mask, poly_ln, mode='wrap') / r_iI[0] / r_ij
    l_dot *= -np.einsum('aij,ai->ji', (e_vectors.T / r_ij), (n_vectors[0].T / r_iI[0]))

    l_2 = 0

    np.sum(np.outer(cutoff[0], cutoff[0]) * (l_1 + 2 * np.nan_to_num(l_dot) + l_2)
