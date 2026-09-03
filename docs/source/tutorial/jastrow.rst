.. _jastrow:

Jastrow
=======

The Slater--Jastrow trial wave function is written as

.. math::

    \Psi(\mathbf{R}) = e^{J(\mathbf{R})}\Psi_S(\mathbf{R})

where :math:`\Psi_S` is the Slater part of the wave function and :math:`e^{J(\mathbf{R})}` is the Jastrow correlation factor.
The Jastrow factor is chosen to be everywhere positive and symmetric with respect to the exchange of identical particles,
so it keeps the symmetry and the nodal surface of :math:`\Psi_S` unchanged while introducing dynamical correlation into
the wave function. It is also used to enforce the Kato cusp conditions, which determine the behavior of the wave function
when two charged particles approach one another; satisfying them removes divergences in the local energy at particle
coalescence points and strongly reduces the statistical fluctuations of the energy. A good Jastrow factor typically allows
one to recover 80--90% or more of the correlation energy in VMC [1]_ [2]_.

The Jastrow factor implemented here follows the Drummond--Towler--Needs (DTN) form [1]_: it is the sum of homogeneous,
isotropic electron–electron terms :math:`u(r_{ij})`, isotropic electron–nucleus terms :math:`\chi(r_{iI})` centered on
the nuclei, and isotropic electron–electron–nucleus terms :math:`f(r_{ij}, r_{iI}, r_{jI})`, also centered on the nuclei:

.. math::

    J = \sum_{i>j}^{N_e} u(r_{ij}) + \sum_{I=1}^{N_I}\sum_{i=1}^{N_e} \chi(r_{iI}) + \sum_{I=1}^{N_I}\sum_{i>j}^{N_e} f(r_{ij}, r_{iI}, r_{jI})

Each term is a power expansion in the interparticle distances, cut off at a finite optimizable length. All the expansion
coefficients enter the Jastrow factor linearly, which greatly simplifies the optimization; the only nonlinear parameters
are the cutoff lengths. The parameters may depend on the spins of the electrons involved: separate sets of coefficients
can be used for parallel spin-up, parallel spin-down and antiparallel electron pairs, and for spin-up and spin-down
electrons in the electron–nucleus terms.

The three-body :math:`f` term deserves a historical remark: explicit triplet correlations of this kind were first used in
quantum Monte Carlo by Schmidt, Kalos, Lee, and Chester for liquid :math:`{}^4\mathrm{He}` [3]_, where they were shown to
account for most of the difference between pair-product Jastrow and exact results.

Cusp conditions
---------------

The Coulomb potential diverges when two charged particles coalesce, so the kinetic energy must diverge with the opposite
sign to keep the local energy finite. This is achieved by demanding that the wave function obeys the Kato cusp
conditions. For the spherical average :math:`\hat\Psi` of the wave function about the coalescence point,

.. math::

    \left(\frac{\partial\hat\Psi}{\partial r_{ij}}\right)_{r_{ij}=0} = \Gamma_{ij}\hat\Psi_{r_{ij}=0}

where :math:`\Gamma_{ij} = 1/2` for antiparallel-spin electrons, :math:`\Gamma_{ij} = 1/4` for parallel-spin electrons
(in that case the condition applies to the :math:`r_{ij}Y_{1m}` component of :math:`\Psi`), and
:math:`\Gamma_{iI} = -Z_I` for an electron approaching a nucleus of charge :math:`Z_I`.

It is common practice to impose the electron–nucleus cusp conditions on the Slater part of the wave function (or use a
non-divergent pseudopotential) and the electron–electron cusp conditions on the Jastrow factor, since typical forms of
:math:`\Psi_S` depend explicitly on the electron–nucleus distances but not on the electron–electron ones. Note that even
when the cusp conditions are satisfied, the local energy retains a point discontinuity at coalescence points because the
:math:`\mathcal{O}(r_{ij}^0)` term depends on the direction of approach [1]_; this is harmless in practice.

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

Summary of Methods
------------------

Jastrow class has a following methods:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1
   :width: 100%

   * - Method
     - Output
     - Shape
   * - :ref:`u_term <u-term>`
     - :math:`u(r_{ij})`
     - scalar
   * - :ref:`chi_term <chi-term>`
     - :math:`\chi(r_{iI})`
     - scalar
   * - :ref:`f_term <f-term>`
     - :math:`f(r_{ij}, r_{iI}, r_{jI})`
     - scalar
   * - :ref:`u_term_gradient <u-term-gradient>`
     - :math:`\nabla u(r_{ij})`
     - :math:`(3N_e,)`
   * - :ref:`chi_term_gradient <chi-term-gradient>`
     - :math:`\nabla \chi(r_{iI})`
     - :math:`(3N_e,)`
   * - :ref:`f_term_gradient <f-term-gradient>`
     - :math:`\nabla f(r_{ij}, r_{iI}, r_{jI})`
     - :math:`(3N_e,)`
   * - :ref:`u_term_laplacian <u-term-laplacian>`
     - :math:`\Delta u(r_{ij})`
     - scalar
   * - :ref:`chi_term_laplacian <chi-term-laplacian>`
     - :math:`\Delta \chi(r_{iI})`
     - scalar
   * - :ref:`f_term_laplacian <f-term-laplacian>`
     - :math:`\Delta f(r_{ij}, r_{iI}, r_{jI})`
     - scalar

.. _u-term:

u-term
------

:math:`u(r_{ij})` term consists of a complete power expansion in electron-electron distances :math:`r_{ij}`
up to order :math:`r_{ij}^{C+N_u}`:

.. math::

    u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij})\sum_{l=0}^{N_u}\alpha_lr^l_{ij}

where :math:`\Theta` is the Heaviside function. This term goes to zero at the cutoff length :math:`L_u` with
:math:`C - 1` continuous derivatives. The truncation order :math:`C` determines the behavior of the local energy at the
cutoff length: if :math:`C = 2`, the gradient of this term is continuous but the second derivative and hence the local
energy is discontinuous, and if :math:`C = 3` then both the gradient of this term and the local energy are continuous.
The price paid for a smoother local energy at :math:`C = 3` is a reduction in variational freedom; in practice
:math:`C = 3` is used for all-electron atoms while :math:`C = 2` often gives slightly lower energies [1]_.
The electron-electron Kato cusp condition at :math:`r_{ij} = 0` is satisfied by the constraint:

.. math::

    \alpha_1 = \left[\frac{\Gamma_{ij}}{(-L_u)^C} + \frac{\alpha_0C}{L_u}\right]

where :math:`\Gamma_{ij} = 1/2` if electrons :math:`i` and :math:`j` have opposite spins and :math:`\Gamma_{ij} = 1/4`
if :math:`i` and :math:`j` have the same spin. If the same :math:`u` function is used for all pairs of electrons then
only the antiparallel-spin cusp condition can be satisfied; using different functions for parallel and antiparallel
pairs allows both cusp conditions to hold. Since electrons of the same spin rarely come close to each other (their
coalescence is suppressed by the Pauli principle), the error introduced by violating the parallel-spin cusp condition
is small.

For certain electron coordinates, :math:`u` term can be obtained with :py:meth:`casino.Jastrow.u_term` method::

    jastrow.u_term(e_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_ij, jastrow.u_parameters.T)
    cutoff = np.minimum(r_ij - jastrow.u_cutoff, 0) ** C
    np.sum(np.triu(cutoff * np.choose(ee_spin_mask, poly, mode='wrap'), 1))

.. _chi-term:

chi-term
--------

:math:`\chi(r_{iI})` term consists of a complete power expansion in electron-nucleus distances :math:`r_{iI}`:

.. math::

    \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI})\sum_{m=0}^{N_\chi}\beta_mr^m_{iI}

where :math:`\Theta` is the Heaviside function. This term goes to zero at the cutoff length :math:`L_{\chi I}` with
:math:`C - 1` continuous derivatives. Separate sets of coefficients :math:`\beta_{mI}` are used for each inequivalent
nucleus; for equivalent ions :math:`I` and :math:`J` it may be assumed that :math:`\beta_{mI} = \beta_{mJ}`.
The term involving the ionic charge :math:`Z_I` enforces the electron–nucleus Kato cusp condition:

.. math::

    \beta_1 = \left[\frac{-Z_I}{(-L_{\chi I})^C} + \frac{\beta_{0I}C}{L_{\chi I}}\right]

if the Slater part of the wave function satisfies the electron–nucleus cusp condition, or if a non-divergent
pseudopotential is used, then the Jastrow factor is required to be cuspless at the nuclei, i.e it should satisfy
the above equation with :math:`Z_I = 0`. It is clearly preferable to use orbitals that satisfy the electron–nucleus
cusp condition rather than to enforce it through :math:`\chi`: the latter requires a very large number of
:math:`\chi` parameters and still gives significantly poorer variational energies [1]_.

For certain electron coordinates, :math:`\chi` term can be obtained with :py:meth:`casino.Jastrow.chi_term` method::

    jastrow.chi_term(n_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_iI, jastrow.chi_parameters[0].T)
    cutoff = np.minimum(r_iI - jastrow.chi_cutoff, 0) ** C
    np.sum(cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap'))

.. _f-term:

f-term
------

:math:`f(r_{ij}, r_{iI}, r_{jI})` term is the most general expansion of a function of :math:`r_{ij}` , :math:`r_{iI}` , and :math:`r_{jI}`
that is cuspless at the coalescence point and goes smoothly to zero when either :math:`r_{iI}` or :math:`r_{jI}` reach cutoff lengths:

.. math::

    f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI})
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

The :math:`f` term describes inhomogeneous two-electron correlations that are spherically symmetric about nucleus
:math:`I`. Its importance for all-electron atoms is considerable: for the Ne atom less than 60% of the correlation
energy can be retrieved using only :math:`u` and :math:`\chi`, whereas more than 90% can be retrieved when :math:`f`
terms are used as well [1]_.

Various linear restrictions are placed on :math:`\gamma_{lmnI}`. For the Jastrow factor to be symmetric under electron
exchanges it is required that :math:`\gamma_{lmnI} = \gamma_{mlnI} \ \forall I, l, m, n`. The condition that the
:math:`f` term has no electron–electron cusp,

.. math::

    \left(\frac{\partial f}{\partial r_{ij}}\right)_{\substack{r_{ij}=0 \\ r_{iI}=r_{jI}}} = 0

leads to the following :math:`2N_{fI}^{eN} + 1` constraints (:math:`\forall k \in \{0,\ldots,2N_{fI}^{eN}\}`):

.. math::

    \sum_{l,m \,:\, l+m=k}\gamma_{lm1I} = 0

and the condition that it has no electron–nucleus cusp,

.. math::

    \left(\frac{\partial f}{\partial r_{iI}}\right)_{\substack{r_{iI}=0 \\ r_{ij}=r_{jI}}} = 0

leads to the following :math:`N_{fI}^{eN} + N_{fI}^{ee} + 1` constraints (:math:`\forall k' \in \{0,\ldots,N_{fI}^{eN} + N_{fI}^{ee}\}`):

.. math::

    \sum_{m,n \,:\, m+n=k'}(C\gamma_{0mnI} - L_{fI}\gamma_{1mnI}) = 0

If desired, :math:`N_{fI}^{ee} + 1` constraints :math:`(\gamma_{00nI} = 0 \ \forall n)` may be imposed to prevent
duplication of the :math:`u` term and :math:`N_{fI}^{eN} + 1` constraints :math:`(\gamma_{l00I} = 0 \ \forall l)` to
prevent duplication of the :math:`\chi` term. Note, however, that the terms of :math:`f` with :math:`l = m = 0` do not
exactly duplicate :math:`u`: they are electron–electron terms local to ion :math:`I`, and this extra variational
freedom may be used to describe correlations on two different length scales. Since all constraints are linear in
:math:`\gamma_{lmnI}`, they are imposed by Gaussian elimination: a subset of the parameters is expressed through the
remaining independent ones, which are optimized directly [1]_.

For certain electron coordinates, :math:`f` term can be obtained with :py:meth:`casino.Jastrow.f_term` method::

    jastrow.f_term(e_powers, n_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval3d
    r_ijI = np.tile(r_iI[0], (ne, 1))
    cutoff = np.minimum(r_iI - jastrow.f_cutoff, 0) ** C
    poly = polyval3d(r_ijI, r_ijI.T, r_ij, jastrow.f_parameters[0].T)
    np.sum(np.triu(np.outer(cutoff[0], cutoff[0]) * np.choose(ee_spin_mask, poly, mode='wrap'), 1))

.. _u-term-gradient:

u-term gradient
---------------

Considering that gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla f(r) =  f'(r) \mathbf{\hat r}

There are only two non-zero terms of :math:`u(r_{ij})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_i} u(r_{ij}) = -\nabla_{e_j} u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij})\mathbf{\hat r}_{ij}\sum_{l=0}^{N_u}\left(\frac{C}{r_{ij} - L_u} + \frac{l}{r_{ij}}\right)\alpha_lr^l_{ij}

where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`

For certain electron coordinates, :math:`u` term gradient can be obtained with :py:meth:`casino.Jastrow.u_term_gradient` method::

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

.. _chi-term-gradient:

chi-term gradient
-----------------

Considering that gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla f(r) =  f'(r) \mathbf{\hat r}

There is only one non-zero term of :math:`\chi(r_{iI})` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI})\mathbf{\hat r}_{iI}\sum_{m=0}^{N_\chi}\left(\frac{C}{r_{iI} - L_{\chi I}} + \frac{m}{r_{iI}}\right)\beta_mr^m_{iI}

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

.. _f-term-gradient:

f-term gradient
---------------

Considering that gradient of spherically symmetric function (in 3-D space) is:

.. math::

    \nabla f(r) =  f'(r) \mathbf{\hat r}

There are only two non-zero terms of :math:`f(r_{ij}, r_{iI}, r_{jI})` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    g_{ij} =  \mathbf{\hat r}_{ij} \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}\left(\frac{n}{r_{ij}}\right)\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    g_{iI} = \mathbf{\hat r}_{iI} \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}\left(\frac{C}{r_{iI} - L_{fI}} + \frac{l}{r_{iI}}\right)\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    g_{jI} = \mathbf{\hat r}_{jI} \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}\left(\frac{C}{r_{jI} - L_{fI}} + \frac{m}{r_{jI}}\right)\gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

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

.. _u-term-laplacian:

u-term laplacian
----------------

Considering that Laplace operator of spherically symmetric function (in 3-D space) is:

.. math::

    \Delta f(r) = f''(r) + \frac{2}{r} f'(r)

There are only two non-zero terms of :math:`u(r_{ij})` laplacian, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \Delta_{e_i} u(r_{ij}) = \Delta_{e_j} u(r_{ij}) = (r_{ij} - L_u)^C\Theta(L_u - r_{ij}) \sum_{l=0}^{N_u}\left(\frac{C(C-1)}{(r_{ij} - L_u)^2} + \frac{2C(l+1)}{r_{ij}(r_{ij} - L_u)} + \frac{l(l+1)}{r_{ij}^2}\right)\alpha_lr^l_{ij}

For certain electron coordinates, :math:`u` term laplacian can be obtained with :py:meth:`casino.Jastrow.u_term_laplacian` method::

    jastrow.u_term_laplacian(e_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = jastrow.u_cutoff
    l = np.arange(jastrow.u_parameters.shape[1])
    cutoff = np.minimum(r_ij - jastrow.u_cutoff, 0) ** C
    poly =  polyval(r_ij, jastrow.u_parameters.T) * C * (C - 1) / (r_ij - L) ** 2
    poly += 2 * polyval(r_ij, ((l + 1) * jastrow.u_parameters).T) * C / r_ij / (r_ij - L)
    poly += polyval(r_ij, (l * (l + 1) * jastrow.u_parameters).T) / r_ij ** 2
    2 * np.sum(np.triu(cutoff * np.choose(ee_spin_mask, poly, mode='wrap'), 1))

.. _chi-term-laplacian:

chi-term laplacian
------------------

Considering that Laplace operator of spherically symmetric function (in 3-D space) is:

.. math::

    \Delta f(r) = f''(r) + \frac{2}{r} f'(r)

then :math:`\chi(r_{iI})` term laplacian:

.. math::

    \Delta_{e_i} \chi(r_{iI}) = (r_{iI} - L_{\chi I})^C\Theta(L_{\chi I} - r_{iI}) \sum_{m=0}^{N_\chi}\left(\frac{C(C-1)}{(r_{iI} - L_{\chi I})^2} + \frac{2C(m+1)}{r_{iI}(r_{iI} - L_{\chi I})} + \frac{m(m+1)}{r_{iI}^2}\right)\beta_mr^m_{iI}

For certain electron coordinates, :math:`\chi` term laplacian can be obtained with :py:meth:`casino.Jastrow.chi_term_laplacian` method::

    jastrow.chi_term_laplacian(n_powers)

this is equivalent to (continues :ref:`from <intermediate data>`)::

    from numpy.polynomial.polynomial import polyval
    L = jastrow.chi_cutoff
    m = np.arange(jastrow.chi_parameters[0].shape[1])
    cutoff = np.minimum(r_iI - L, 0) ** C
    poly = polyval(r_iI, jastrow.chi_parameters[0].T) * C * (C - 1) / (r_iI[0] - L[0]) ** 2
    poly += 2 * polyval(r_iI, ((m + 1) * jastrow.chi_parameters[0]).T) * C / r_iI / (r_iI[0] - L[0])
    poly += polyval(r_iI, (m * (m + 1) * jastrow.chi_parameters[0]).T) / r_iI ** 2
    np.sum(cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap'))

.. _f-term-laplacian:

f-term laplacian
----------------

Considering that Laplace operator of spherically symmetric function (in 3-D space) is:

.. math::

    \Delta f(r) = f''(r) + \frac{2}{r} f'(r)

and :math:`f` term is a product of two spherically symmetric functions :math:`g(r_{ij})` and :math:`h(r_{iI})` so using:

.. math::

    \Delta_{e_i}(g(r_{ij})h(r_{iI})) = h(r_{iI})\Delta_{e_i}g(r_{ij}) + 2\nabla_{e_i}g(r_{ij})\cdot\nabla_{e_i}h(r_{iI}) + g(r_{ij})\Delta_{e_i}h(r_{iI})

There are only two non-zero terms of :math:`f(r_{ij}, r_{iI}, r_{jI})` laplacian, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    l_{iI} = \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    \left(\frac{C(C-1)}{(r_{iI} - L_{fI})^2} + \frac{2C(l+1)}{r_{iI}(r_{iI} - L_{fI})} + \frac{l(l+1)}{r_{iI}^2}\right)
    \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    l_{jI} = \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    \left(\frac{C(C-1)}{(r_{jI} - L_{fI})^2} + \frac{2C(m+1)}{r_{jI}(r_{jI} - L_{fI})} + \frac{m(m+1)}{r_{jI}^2}\right)
    \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    l_{dot,i} = \mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{iI}
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    \left(\frac{C}{(r_{iI} - L_{fI})} + \frac{l}{r_{iI}}\right) \frac{n}{r_{ij}} \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    l_{dot,j} = \mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{jI}
    \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    \left(\frac{C}{(r_{jI} - L_{fI})} + \frac{m}{r_{jI}}\right) \frac{n}{r_{ij}} \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    l_{ij} = \sum_{l=0}^{N_{fI}^{eN}}\sum_{m=0}^{N_{fI}^{eN}}\sum_{n=0}^{N_{fI}^{ee}}
    \frac{n(n+1)}{r_{ij}^2} \gamma_{lmnI}r_{iI}^lr_{jI}^mr_{ij}^n

.. math::

    \Delta_{e_i} f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI}) (l_{iI} + 2l_{dot,i} + l_{ij})

.. math::

    \Delta_{e_j} f(r_{ij}, r_{iI}, r_{jI}) = (r_{iI} - L_{fI})^C(r_{jI} - L_{fI})^C \Theta(L_{fI} - r_{iI})\Theta(L_{fI} - r_{jI}) (l_{jI} - 2l_{dot,j} + l_{ij})

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
    poly_l = polyval3d(r_ijI, r_ijI.T, r_ij, ((l + 1) * jastrow.f_parameters[0]).T)
    poly_ll = polyval3d(r_ijI, r_ijI.T, r_ij, (l * (l + 1) * jastrow.f_parameters[0]).T)
    poly_n = polyval3d(r_ijI, r_ijI.T, r_ij, (n * jastrow.f_parameters[0]).T)
    poly_nn = polyval3d(r_ijI, r_ijI.T, r_ij, (n * (n + 1) * jastrow.f_parameters[0]).T)
    poly_lm = polyval3d(r_ijI, r_ijI.T, r_ij, (l * m * jastrow.f_parameters[0]).T)
    poly_ln = polyval3d(r_ijI, r_ijI.T, r_ij, (l * n * jastrow.f_parameters[0]).T)

    l_1 = np.choose(ee_spin_mask, poly, mode='wrap') * C * (C - 1) / (r_iI[0] - L[0]) ** 2
    l_1 += 2 * np.choose(ee_spin_mask, poly_l, mode='wrap') * C / (r_iI[0] - L[0]) / r_iI[0]
    l_1 += np.choose(ee_spin_mask, poly_ll, mode='wrap') / r_iI[0] ** 2

    l_dot = np.choose(ee_spin_mask, poly_n, mode='wrap') * C / (r_iI[0] - L[0]) / r_ij
    l_dot += np.choose(ee_spin_mask, poly_ln, mode='wrap') / r_iI[0] / r_ij
    l_dot *= -np.einsum('aij,ai->ji', (e_vectors.T / r_ij), (n_vectors[0].T / r_iI[0]))

    l_2 = np.choose(ee_spin_mask, poly_nn, mode='wrap') / r_ij ** 2

    np.sum(np.outer(cutoff[0], cutoff[0]) * np.nan_to_num(l_1 + 2 * l_dot + l_2))

References
----------

.. [1] N. D. Drummond, M. D. Towler, and R. J. Needs,
   *Jastrow correlation factor for atoms, molecules, and solids*,
   Phys. Rev. B **70**, 235119 (2004).

.. [2] P. López Ríos, P. Seth, N. D. Drummond, and R. J. Needs,
   *Framework for constructing generic Jastrow correlation factors*,
   Phys. Rev. E **86**, 036703 (2012).

.. [3] K. E. Schmidt, M. A. Kalos, M. A. Lee, and G. V. Chester,
   *Variational Monte Carlo Calculations of Liquid* :math:`{}^4He` *with Three-Body Correlations*,
   Phys. Rev. Lett. **45**, 573 (1980).
