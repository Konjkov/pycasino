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

    import numpy as np
    import numpy.ma as ma
    from numpy.polynomial.polynomial import polyval

    neu, ned = config.input.neu, config.input.ned
    ne = neu + ned
    r_e = np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
    e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
    e_powers = jastrow.ee_powers(e_vectors)
    jastrow.u_term(e_powers)

    r_ij = np.linalg.norm(e_vectors, axis=2)
    cutoff_mask = r_ij > jastrow.u_cutoff
    cutoff_mask[np.tril_indices_from(cutoff_mask)] = True
    spin_mask = np.ones(shape=(ne, ne), dtype=int)
    spin_mask[:neu, :neu] = 0
    spin_mask[neu:, neu:] = 2
    cutoff = (ma.masked_array(r_ij, cutoff_mask) - jastrow.u_cutoff) ** jastrow.trunc
    np.sum(cutoff * np.choose(spin_mask, polyval(r_ij, jastrow.u_parameters.T), mode='wrap'))


chi_term
--------

:math:`\chi(r_{iI})` term consists of a complete power expansion in electron-nucleus distances :math:`r_{iI}`

.. math::

    \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI})\sum_{m=0}^{N_\chi}\beta_mr^m_{iI}

where :math:`\Theta` is the Heaviside function. This term goes to zero at the cutoff length :math:`L_{\chi I}`.
The term involving the ionic charge :math:`Z_I` enforces the electron–nucleus cusp condition:

.. math::

    \beta_1 = \left[\frac{-Z_I}{(-L_{\chi I})^C} + \frac{\beta_{0I}C}{L_{\chi I}}\right]

For certain electron coordinates, :math:`\chi` term can be obtained with :py:meth:`casino.Jastrow.chi_term` method::

    import numpy as np
    import numpy.ma as ma

    neu, ned = config.input.neu, config.input.ned
    ne = neu + ned
    r_e = np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
    atom_positions = config.wfn.atom_positions
    n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
    n_powers = jastrow.en_powers(n_vectors)
    jastrow.chi_term(n_powers)

    r_iI = np.linalg.norm(n_vectors, axis=2)
    cutoff_mask = r_iI > jastrow.chi_cutoff[0]
    cutoff_mask[np.tril_indices_from(cutoff_mask)] = True
    spin_mask = np.zeros(shape=(ne,), dtype=int)
    spin_mask[neu:, neu:] = 1



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

    neu, ned = config.input.neu, config.input.ned
    ne = neu + ned
    r_e = np.random.uniform(-1, 1, ne * 3).reshape((ne, 3))
    atom_positions = config.wfn.atom_positions
    e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
    n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(atom_positions, 1)
    e_powers = jastrow.ee_powers(e_vectors)
    n_powers = jastrow.en_powers(n_vectors)
    jastrow.f_term(e_powers, n_powers)


u_term_gradient
---------------

There is only two non-zero terms of :math:`u(r_{ij})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_i} u(r_{ij}) = -\nabla_{e_j} u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij})\mathbf{\hat r}_{ij}\sum_{l=0}^{N_u}(C/(r_{ij} - L_u) + l/r_{ij})\alpha_lr^l_{ij}

where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`


chi_term_gradient
-----------------

There is only one non-zero term of :math:`\chi(r_{iI})` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI})\mathbf{\hat r}_{iI}\sum_{m=0}^{N_\chi}(C/(r_{iI} - L_{\chi I}) + m/r_{iI})\beta_mr^m_{iI}

where :math:`\mathbf{\hat r}_{iI}` is the unit vector in the direction of the :math:`\mathbf{r}_{iI}`


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

    dot = \mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{iI}
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}(n/r_{ij})\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}(C/(r_{iI} - L_{fI}) + l / r_{iI})\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n


.. math::

    \Delta f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI}) \times

.. math::

    (2dot)
