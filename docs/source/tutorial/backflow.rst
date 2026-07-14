.. _backflow:

Backflow
========
The Slater-Jastrow (SJ) wave function

.. math::

    \Psi^{SJ}_T(\mathbf{R}) = e^{J(\mathbf{R})}\Psi_S(\mathbf{R})

consists of a Slater part :math:`\Psi_S` (a determinant or a sum of determinants), which defines the nodes of the wave function,
multiplied by the Jastrow correlation factor :math:`e^{J}`. Backflow correlations are introduced by substituting a set of collective
coordinates :math:`\mathbf{X} = \{\mathbf{x}_i\}` for the actual electron coordinates :math:`\mathbf{R} = \{\mathbf{r}_i\}` in the
Slater part only,

.. math::

    \Psi^{BF}_T(\mathbf{R}) = e^{J(\mathbf{R})}\Psi_S(\mathbf{X}), \qquad
    \mathbf{x}_i = \mathbf{r}_i + \mathbf{\xi}_i(\mathbf{R})

where :math:`\mathbf{\xi}_i` is the backflow displacement of electron :math:`i`, which depends on the configuration of the whole
system. While the Jastrow factor keeps electrons away from one another, it does not alter the nodal surface; backflow does, and can
therefore be used to reduce the fixed-node error in DMC (with an important topological limitation; see
:ref:`backflow-nodes`). The concept of backflow goes back to the Feynman--Cohen treatment of excitations in liquid
:math:`{}^4He` [5]_; in QMC it was first applied to electronic systems in studies of the homogeneous electron gas [6]_ [7]_.

Following López Ríos et al. [4]_, the backflow displacement is the sum of a homogeneous, isotropic
electron–electron term :math:`\eta(r_{ij})`, an isotropic electron–nucleus term :math:`\mu(r_{iI})` centered on the nuclei and an
isotropic electron–electron–nucleus term :math:`\Phi(r_{iI}, r_{jI}, r_{ij})`, :math:`\Theta(r_{iI}, r_{jI}, r_{ij})`, also centered
on the nuclei:

.. math::

    \mathbf{\xi}_i = \sum_{j \neq i}^{N_e} \eta(r_{ij})\mathbf{r}_{ij} + \sum_{I=1}^{N_I} \mu(r_{iI})\mathbf{r}_{iI} +
    \sum_{j \neq i}^{N_e}\sum_{I=1}^{N_I} \left[\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}\right]

The electron–electron displacement is taken along :math:`\mathbf{r}_{ij}` because, for a pair of electrons, this is the only
inequivalent direction. The combination :math:`\Phi\mathbf{r}_{ij} + \Theta\mathbf{r}_{iI}` is capable of spanning the plane defined
by :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` and :math:`\mathbf{r}_I`, so no component along :math:`\mathbf{r}_{jI}` is needed.
The displacement can be extended beyond this plane by the :math:`\kappa` term directed along its normal; see the
:ref:`kappa-term <kappa-term>` section.

Each of the functions :math:`\eta`, :math:`\mu`, :math:`\Phi`, :math:`\Theta` is a natural power expansion cut off smoothly at some
radius :math:`L` by the cutoff function

.. math::

    f(r; L) = \left(1 - \frac{r}{L}\right)^C H(L - r)

where :math:`H` is the Heaviside function, the cutoff length :math:`L` is an optimizable parameter and :math:`C` is the truncation
order. The :math:`C`-th derivative of the wave function is discontinuous at :math:`r = L`, so the Laplacian used in the kinetic energy
is discontinuous there for :math:`C < 3`; only :math:`C = 2` and :math:`C = 3` are used in practice. The parameters of :math:`\eta`,
:math:`\Phi` and :math:`\Theta` are allowed to depend on the spins of the electron pair, and those of :math:`\mu` on the spin of the
electron.

When all-electron atoms are present, the electron-nucleus Kato cusp conditions [8]_ cannot be fulfilled unless the total backflow
displacement at the nuclear position is zero. This can be obtained by applying smooth cutoffs around such atoms. An artificial
multiplicative cutoff function :math:`g(r_{iI})` is applied to all contributions to the backflow displacement of particle :math:`i`
that do not depend on the distance :math:`r_{iI}` to the all-electron atom :math:`I`. This includes the homogeneous displacement
:math:`\eta(r_{ij})\mathbf{r}_{ij}` and the inhomogeneous contributions centered on other atoms :math:`J \neq I`, such as
:math:`\mu(r_{iJ})\mathbf{r}_{iJ}`, :math:`\Phi(r_{iJ}, r_{jJ}, r_{ij})\mathbf{r}_{ij}` and
:math:`\Theta(r_{iJ}, r_{jJ}, r_{ij})\mathbf{r}_{iJ}`.
The function :math:`g(r_{iI})` must go to zero together with its first derivative at :math:`r_{iI} = 0`, so that the cusp conditions
are fulfilled; it must become unity when :math:`r_{iI} \geq L_{gI}`, and its first two derivatives must be continuous at
:math:`r_{iI} = L_{gI}` for the local energy to be well defined. The simplest polynomial obeying these conditions is

.. math::

    g(r_{iI}) = \left(\frac{r_{iI}}{L_{gI}}\right)^2 \left[6 - 8 \left(\frac{r_{iI}}{L_{gI}}\right) + 3 \left(\frac{r_{iI}}{L_{gI}}\right)^2 \right]

Although it is possible to optimize the cutoff length :math:`L_{gI}`, a fixed value of about 1 a.u. is normally used.

The backflow part of the wave function is represented by the :class:`casino.Backflow` class.

It must be initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.backflow import Backflow

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    backflow = Backflow(config)

.. _backflow-intermediate-data:

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

.. _backflow-nodes:

Backflow and nodal surfaces
---------------------------
The backflow transformation :math:`\mathbf{x}_i = \mathbf{r}_i + \mathbf{\xi}_i(\mathbf{R})` with a continuous displacement field and
a non-singular Jacobian (required for the kinetic energy to be finite) is a homeomorphism of the configuration space. The nodes of
:math:`\Psi_S(\mathbf{X}(\mathbf{R}))` are the preimage of the nodes of :math:`\Psi_S`, and a homeomorphism preserves topology:
backflow can smoothly deform the nodal surface, but it cannot change the number of nodal pockets.

This limitation matters when the nodal surface of the starting determinant is topologically wrong. The beryllium atom is the classic
example. The Hartree-Fock wave function factorizes, :math:`\Psi_{HF} = D^\uparrow D^\downarrow`, so its node is the union of the two
surfaces :math:`D^\uparrow = 0` and :math:`D^\downarrow = 0`, which intersect on a manifold of codimension 2. Near the intersection
:math:`\Psi_{HF} \sim uv` (with :math:`u = D^\uparrow`, :math:`v = D^\downarrow`), giving four nodal pockets, whereas the exact
ground state has only two [9]_. Note that if an exact eigenfunction
did possess such a nodal crossing, the two surfaces would have to intersect at right angles, since the leading behavior of the wave
function at a crossing is a second-degree harmonic polynomial.

Opening the crossing requires adding to the product a term that does not vanish on the intersection manifold itself:

.. math::

    \Psi = D^\uparrow D^\downarrow - \varepsilon g, \qquad g \neq 0 \text{ on } \{u = v = 0\}

so that locally :math:`uv \to uv - \varepsilon`: the crossing is resolved into two smooth hyperbolic sheets and the four pockets
merge into two. The :math:`2p^2` determinants of a CASSCF(2,4) expansion provide exactly such a term for beryllium; geminal
(AGP) [10]_ and Pfaffian [11]_ wave functions embed it into the antisymmetric form itself (see :ref:`pfaffian`). A backflow transformation could only
achieve this if the displacement field were discontinuous precisely on the intersection manifold, which would produce a divergent
local energy there and is considered a tremendously difficult task ([4]_, Sec. V C).

In short, a multideterminant or pairing wave function performs the topological surgery on the nodes, while backflow refines the
shape of a topologically correct nodal surface; for systems like beryllium both are needed, in that order.

Summary of Methods
------------------

The Backflow class has the following methods:

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
     - :math:`\mu(r_{iI})\mathbf{r}_{iI}`
     - :math:`(3N_e,)`
   * - :ref:`phi_term <phi-term>`
     - :math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}`
     - :math:`(3N_e,)`
   * - :ref:`eta_term_gradient <eta-term-gradient>`
     - :math:`\nabla (\eta(r_{ij})\mathbf{r}_{ij})`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`mu_term_gradient <mu-term-gradient>`
     - :math:`\nabla (\mu(r_{iI})\mathbf{r}_{iI})`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`phi_term_gradient <phi-term-gradient>`
     - :math:`\nabla (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI})`
     - :math:`(3N_e, 3N_e)`
   * - :ref:`eta_term_laplacian <eta-term-laplacian>`
     - :math:`\Delta (\eta(r_{ij})\mathbf{r}_{ij})`
     - :math:`(3N_e,)`
   * - :ref:`mu_term_laplacian <mu-term-laplacian>`
     - :math:`\Delta (\mu(r_{iI})\mathbf{r}_{iI})`
     - :math:`(3N_e,)`
   * - :ref:`phi_term_laplacian <phi-term-laplacian>`
     - :math:`\Delta (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij} + \Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI})`
     - :math:`(3N_e,)`

.. _eta-term:

eta-term
--------
:math:`\eta(r_{ij})\mathbf{r}_{ij}` describes the homogeneous backflow displacement; :math:`\eta` is a natural power expansion in the
electron-electron distance :math:`r_{ij}`:

.. math::

    \eta(r_{ij}) = f(r_{ij}; L_\eta) \sum_{k=0}^{N_\eta}c_kr^k_{ij}

where :math:`f` is the cutoff function defined above, and the parameters :math:`c_k` are allowed to depend on the spin pair
(:math:`\uparrow\uparrow`, :math:`\uparrow\downarrow` and :math:`\downarrow\downarrow`). To satisfy the electron-electron Kato cusp
conditions at :math:`r_{ij} = 0`, the total backflow displacement must have a well-defined gradient when :math:`r_{ij} \to 0` if
:math:`i` and :math:`j` are distinguishable particles, and zero gradient if they are indistinguishable. The :math:`\eta` term is
therefore constrained for like-spin electron pairs only:

.. math::

    L_\eta c_1 = C c_0

while the anti-parallel-spin :math:`\eta` function may have a nonzero derivative at :math:`r_{ij} = 0`.

For certain electron coordinates, :math:`\eta` term can be obtained with :py:meth:`casino.Backflow.eta_term` method::

    backflow.eta_term(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_ij, backflow.eta_parameters.T)
    cutoff = np.maximum(1 - r_ij / backflow.eta_cutoff[0], 0) ** C
    np.fill_diagonal(cutoff, 0)
    eta = np.expand_dims(cutoff * np.choose(ee_spin_mask, poly, mode='wrap'), -1)
    np.sum(-e_vectors * eta, axis=0)

.. _mu-term:

mu-term
-------
:math:`\mu(r_{iI})\mathbf{r}_{iI}` term describes the electron-nucleus backflow displacement centered on the nucleus :math:`I`;
:math:`\mu` is a natural power expansion in the electron-nucleus distance :math:`r_{iI}`:

.. math::

    \mu(r_{iI}) = f(r_{iI}; L_\mu) \sum_{k=0}^{N_\mu}d_kr^k_{iI}

To satisfy the electron-nucleus Kato cusp conditions, the total backflow displacement must have a well-defined gradient when
:math:`r_{iI} \to 0`, and must be zero at :math:`r_{iI} = 0` if :math:`I` is an all-electron atom. This is satisfied if

.. math::

    L_{\mu I} d_{1 I} = C d_{0 I}

for all atoms, and in addition,

.. math::

    d_{0 I} = 0

for all-electron atoms.

For certain electron coordinates, :math:`\mu` term can be obtained with :py:meth:`casino.Backflow.mu_term` method::

    backflow.mu_term(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval
    poly = polyval(r_iI, backflow.mu_parameters[0].T)
    cutoff = np.maximum(1 - r_iI / backflow.mu_cutoff, 0) ** C
    n_vectors * np.expand_dims(cutoff[0] * np.choose(en_spin_mask, poly, mode='wrap'), -1)

.. _phi-term:

phi-term
--------
:math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` and :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` terms describe
two-electron backflow displacements in the presence of a nearby nucleus, in terms of the distances :math:`r_{ij}`, :math:`r_{iI}`,
:math:`r_{jI}` and the vectors :math:`\mathbf{r}_{ij}`, :math:`\mathbf{r}_{iI}`:

.. math::

    \Phi(r_{iI}, r_{jI}, r_{ij}) = \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}} \Phi_{klmI}

.. math::

    \Theta(r_{iI}, r_{jI}, r_{ij}) = \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}} \Theta_{klmI}

where

.. math::

    \Phi_{klmI} = f(r_{iI}; L_{\Phi I})f(r_{jI}; L_{\Phi I})\phi_{klmI}r_{iI}^kr_{jI}^lr_{ij}^m

.. math::

    \Theta_{klmI} = f(r_{iI}; L_{\Phi I})f(r_{jI}; L_{\Phi I})\theta_{klmI}r_{iI}^kr_{jI}^lr_{ij}^m

and the parameters :math:`\phi_{klmI}`, :math:`\theta_{klmI}` are allowed to depend on the spin pair. To ensure the electron-nucleus
Kato cusp conditions the following :math:`3(N_{\Phi I}^{ee} + N_{\Phi I}^{eN} + 1)` constraints are applied:

.. math::

    \sum_{l,m}^{l+m=\alpha}(C\phi_{0lmI} - L_{\Phi I}\phi_{1lmI}) = \sum_{k,m}^{k+m=\alpha}(C\phi_{k0mI} - L_{\Phi I}\phi_{k1mI}) =
    \sum_{k,m}^{k+m=\alpha}(C\theta_{k0mI} - L_{\Phi I}\theta_{k1mI}) = 0 \quad \forall \alpha

another :math:`2N_{\Phi I}^{eN} + 1` constraints follow from the electron-electron Kato cusp conditions:

.. math::

    \sum_{k,l}^{k+l=\alpha}\theta_{kl1I} = 0 \quad \forall \alpha

and extra :math:`2N_{\Phi I}^{eN} + 1` constraints are applied for like-spin electron pairs:

.. math::

    \sum_{k,l}^{k+l=\alpha}\phi_{kl1I} = 0 \quad \forall \alpha

For all-electron atoms, where the backflow displacement must vanish at the nucleus, there are :math:`4(N_{\Phi I}^{ee} + N_{\Phi I}^{eN})+2`
additional constraints on :math:`\phi_{klm}`

.. math::

    \sum_{l,m}^{l+m=\alpha}\phi_{0lmI} = \sum_{l,m}^{l+m=\alpha}m\phi_{0lmI} = \sum_{k,m}^{k+m=\alpha}\phi_{k0mI} = \sum_{k,m}^{k+m=\alpha}m\phi_{k0mI} = 0 \quad \forall \alpha

and :math:`3(N_{\Phi I}^{ee} + N_{\Phi I}^{eN})+2` additional constraints on :math:`\theta_{klm}`

.. math::

    \sum_{l,m}^{l+m=\alpha}\theta_{0lmI} = \sum_{l,m}^{l+m=\alpha}m\theta_{0lmI} = \sum_{k,m}^{k+m=\alpha}m\theta_{k0mI} = 0 \quad \forall \alpha

These constraints form an indeterminate system of homogeneous linear equations, so a subset of the parameters can be expressed
in terms of the remaining free parameters by Gaussian elimination.

The electron-electron-nucleus term may optionally be constrained to be irrotational, so that the backflow displacement derives from a
scalar backflow potential :math:`Y(\mathbf{R})` as :math:`\mathbf{\xi}_i = \nabla_i Y` (the :math:`\eta` and :math:`\mu` terms satisfy
this by construction). From :math:`\nabla_i \times \mathbf{\xi}_i = 0` it follows that

.. math::

    r_{ij}\frac{\partial}{\partial r_{iI}}\Phi(r_{iI}, r_{jI}, r_{ij}) = r_{iI}\frac{\partial}{\partial r_{ij}}\Theta(r_{iI}, r_{jI}, r_{ij})

which for :math:`C > 0` results in the equations

.. math::

    (C+k)\phi_{k,l,m-1,I} - L_{\Phi I}(k+1)\phi_{k+1,l,m-1,I} - (m+1)\theta_{k-2,l,m+1,I} + L_{\Phi I}(m+1)\theta_{k-1,l,m+1,I} = 0

while for :math:`C = 0`,

.. math::

    (k+1)\phi_{k+1,l,m-1,I} - (m+1)\theta_{k-1,l,m+1,I} = 0

where :math:`0 \leq k \leq N_{\Phi I}^{eN}+2`, :math:`0 \leq l \leq N_{\Phi I}^{eN}`, :math:`0 \leq m \leq N_{\Phi I}^{ee}+1`, and
parameters with indices out of the allowed range are taken to be zero. These constraints reduce the number of free
electron-electron-nucleus parameters by more than half, as one would expect, because an equivalent displacement could be obtained by
parameterizing the single scalar field :math:`Y` instead of the two functions :math:`\Phi` and :math:`\Theta`.

For certain electron coordinates, :math:`\Phi` and :math:`\Theta` terms can be obtained with :py:meth:`casino.Backflow.phi_term` method::

    backflow.phi_term(e_powers, n_powers, e_vectors, n_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval3d
    r_ijI = np.tile(r_iI[0], (ne, 1))
    cutoff = np.maximum(1 - r_iI/backflow.phi_cutoff, 0) ** C
    phi_poly = polyval3d(r_ijI, r_ijI.T, r_ij, backflow.phi_parameters[0].T)
    theta_poly = polyval3d(r_ijI, r_ijI.T, r_ij, backflow.theta_parameters[0].T)
    phi = np.outer(cutoff[0], cutoff[0]) * np.choose(ee_spin_mask, phi_poly, mode='wrap')
    theta = np.outer(cutoff[0], cutoff[0]) * np.choose(ee_spin_mask, theta_poly, mode='wrap')
    np.fill_diagonal(theta, 0)
    np.sum(-e_vectors * np.expand_dims(phi, -1) + n_vectors * np.expand_dims(theta, -1), axis=0)

.. _kappa-term:

kappa-term
----------
The combination :math:`\Phi\mathbf{r}_{ij} + \Theta\mathbf{r}_{iI}` spans the plane defined by :math:`\mathbf{r}_i`,
:math:`\mathbf{r}_j` and :math:`\mathbf{r}_I`. The electron-electron-nucleus backflow displacement can be extended beyond this plane
by the :math:`\kappa` term directed along the only direction perpendicular to it:

.. math::

    \mathbf{\xi}_i^{\perp} = \sum_{j \neq i}^{N_e}\sum_{I=1}^{N_I} \kappa(r_{iI}, r_{jI}, r_{ij})\,(\mathbf{r}_{iI} \times \mathbf{r}_{ij})

where :math:`\kappa` is a natural power expansion of the same form as :math:`\Phi` and :math:`\Theta`:

.. math::

    \kappa(r_{iI}, r_{jI}, r_{ij}) = \sum_{k=0}^{N_{\kappa I}^{eN}}\sum_{l=0}^{N_{\kappa I}^{eN}}\sum_{m=0}^{N_{\kappa I}^{ee}}
    f(r_{iI}; L_{\kappa I})f(r_{jI}; L_{\kappa I})\kappa_{klmI}r_{iI}^kr_{jI}^lr_{ij}^m

with the parameters :math:`\kappa_{klmI}` allowed to depend on the spin pair. Note that, unlike a component along
:math:`\mathbf{r}_{jI}`, which would be redundant (:math:`\mathbf{r}_{jI} = \mathbf{r}_{iI} - \mathbf{r}_{ij}`, so it can be absorbed
exactly into :math:`\Phi` and :math:`\Theta`), the :math:`\kappa` term genuinely enlarges the variational freedom.

The :math:`\kappa` term has the following properties:

- :math:`\mathbf{r}_{iI} \times \mathbf{r}_{ij}` is a pseudovector, so the :math:`\kappa` displacement breaks parity: mirror-image
  configurations receive mirror-broken displacements. For any system possessing an improper symmetry operation (all atoms, and any
  molecule with a mirror plane or an inversion center) the energy is exactly even in the :math:`\kappa` parameters,
  :math:`E(\kappa) = E(-\kappa)`, so the energy gradient vanishes identically at :math:`\kappa = 0` and the term can contribute at
  first order only for chiral arrangements of the nuclei.
- It is incompatible with the irrotational constraint: the gradient :math:`\nabla_i Y` of any scalar function of the interparticle
  distances lies in the plane, so :math:`\nabla_i \times \mathbf{\xi}_i = 0` forbids an out-of-plane component.
- Since :math:`|\mathbf{r}_{iI} \times \mathbf{r}_{ij}| \leq r_{iI}r_{ij}`, the displacement automatically vanishes both at the
  nucleus and at the electron-electron coalescence point, and the vector :math:`\mathbf{r}_{iI} \times \mathbf{r}_{ij}` itself is
  bilinear in the electron coordinates and therefore smooth. The Kato cusp conditions thus impose fewer constraints than for
  :math:`\Phi` and :math:`\Theta`; in particular, no additional constraints are needed at all-electron atoms. The zero-gradient
  condition at the coalescence of like-spin electrons requires :math:`\kappa(r_{iI}, r_{iI}, 0) = 0`, i.e.

  .. math::

      \sum_{k,l}^{k+l=\alpha}\kappa_{kl0I} = 0 \quad \forall \alpha

For a system with a symmetry axis (unit vector :math:`\hat{\mathbf{z}}`, e.g. a linear molecule or an atom in an external axial
field), the parity problem can be avoided altogether: multiplying by the pseudoscalar
:math:`P = \hat{\mathbf{z}} \cdot (\mathbf{r}_{iI} \times \mathbf{r}_{ij})` turns the displacement into a true vector,

.. math::

    \mathbf{\xi}_i^{\perp} = \sum_{j \neq i}^{N_e}\sum_{I=1}^{N_I} \kappa(r_{iI}, r_{jI}, r_{ij})
    \left[\hat{\mathbf{z}} \cdot (\mathbf{r}_{iI} \times \mathbf{r}_{ij})\right](\mathbf{r}_{iI} \times \mathbf{r}_{ij})

Under a rotation about the axis this expression is covariant, and under a reflection :math:`\sigma_v` in a plane containing the axis
the pseudoscalar :math:`P` and the cross product change sign together, so the displacement transforms as a true vector and the energy
is no longer even in the :math:`\kappa` parameters. For a linear molecule this construction is equivalent to taking the second
nucleus as the fourth particle, since :math:`\hat{\mathbf{z}} \propto \mathbf{r}_{IJ}`. Furthermore,
:math:`P\,(\mathbf{r}_{iI} \times \mathbf{r}_{ij})` is a smooth (quartic) polynomial in the electron coordinates which vanishes
quadratically both at :math:`r_{ij} \to 0` and at :math:`r_{iI} \to 0`, so the displacement and its gradient vanish automatically at
all coalescence points: in this form the Kato cusp conditions impose no constraints at all on the :math:`\kappa_{klmI}` parameters,
and only the cutoff is needed. Note, however, that if the point group contains an operation reversing the axis (:math:`\sigma_h` or
inversion, as in :math:`D_{\infty h}`), :math:`P` is even under it while the cross product is not, so the displacement again fails to
transform as a true vector and :math:`E(\kappa) = E(-\kappa)` is restored; such systems additionally require an axis-odd factor, e.g.
odd powers of the signed projections :math:`\hat{\mathbf{z}} \cdot \mathbf{r}_{iI}` in the expansion.

The :math:`\kappa` term is not yet implemented in the :class:`casino.Backflow` class.

.. _eta-term-gradient:

eta-term gradient
-----------------

Considering that vector gradient of spherically symmetric vector function (in 3-D space) is:

.. math::

    \nabla (f(r)\mathbf{r}) = f'(r) \mathbf{\hat r} \otimes \mathbf{r} + f \cdot \mathbf{I}

There are only two non-zero terms of :math:`\eta(r_{ij})\mathbf{r}_{ij}` gradient, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_i} (\eta(r_{ij})\mathbf{r}_{ij}) = f(r_{ij}; L_\eta)
    \sum_{k=0}^{N_\eta} \left[\left(\frac{k}{r_{ij}} - \frac{C}{L_\eta - r_{ij}}\right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} + \mathbf{I} \right] c_kr^k_{ij}

.. math::

    \nabla_{e_j} (\eta(r_{ij})\mathbf{r}_{ij}) = - \nabla_{e_i} (\eta(r_{ij})\mathbf{r}_{ij})

where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`

For certain electron coordinates, :math:`\eta` term gradient can be obtained with :py:meth:`casino.Backflow.eta_term_gradient` method::

    backflow.eta_term_gradient(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

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

There is only one non-zero term of :math:`\mu(r_{iI})\mathbf{r}_{iI}` gradient, i.e. by :math:`i`-th electron coordinates:

.. math::

    \nabla_{e_i} (\mu(r_{iI})\mathbf{r}_{iI}) = f(r_{iI}; L_\mu)
    \sum_{k=0}^{N_\mu} \left[\left(\frac{k}{r_{iI}} - \frac{C}{L_\mu - r_{iI}}\right) \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{iI} + \mathbf{I}\right] d_kr^k_{iI}

where :math:`\mathbf{\hat r}_{iI}` is the unit vector in the direction of the :math:`\mathbf{r}_{iI}`

For certain electron coordinates, :math:`\mu` term gradient can be obtained with :py:meth:`casino.Backflow.mu_term_gradient` method::

    backflow.mu_term_gradient(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

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

There are only two non-zero terms of :math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` gradient, i.e. by :math:`i`-th:

.. math::

    \nabla_{e_i} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{k}{r_{iI}} - \frac{C}{L_{\Phi I} - r_{iI}} \right) \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{ij} + \left(\frac{m}{r_{ij}} \right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} + \mathbf{I} \right] \Phi_{klmI}

or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_j} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}} \right) \mathbf{\hat r}_{jI} \otimes \mathbf{r}_{ij} - \left(\frac{m}{r_{ij}} \right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{ij} - \mathbf{I} \right] \Phi_{klmI}

There are only two non-zero terms of :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` gradient, i.e. by :math:`i`-th:

.. math::

    \nabla_{e_i} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{k}{r_{iI}} -\frac{C}{L_{\Phi I} - r_{iI}}\right) \mathbf{\hat r}_{iI} \otimes \mathbf{r}_{iI} + \left(\frac{m}{r_{ij}}\right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{iI} + \mathbf{I} \right]  \Theta_{klmI}

or :math:`j`-th electron coordinates:

.. math::

    \nabla_{e_j} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = \sum_{k=0}^{N_{\Phi I}^{eN}} \sum_{l=0}^{N_{\Phi I}^{eN}} \sum_{m=0}^{N_{\Phi I}^{ee}} \left[\left(\frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}}\right) \mathbf{\hat r}_{jI} \otimes \mathbf{r}_{iI} - \left(\frac{m}{r_{ij}}\right) \mathbf{\hat r}_{ij} \otimes \mathbf{r}_{iI} \right]  \Theta_{klmI}

with :math:`\Phi_{klmI}` and :math:`\Theta_{klmI}` as defined in the :ref:`phi-term <phi-term>` section,
where :math:`\mathbf{\hat r}_{ij}` is the unit vector in the direction of the :math:`\mathbf{r}_{ij}`
and :math:`\mathbf{\hat r}_{iI}` is the unit vector in the direction of the :math:`\mathbf{r}_{iI}`

For certain electron coordinates, :math:`\phi` term gradient can be obtained with :py:meth:`casino.Backflow.phi_term_gradient` method::

    backflow.phi_term_gradient(e_powers, n_powers, e_vectors, n_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval3d

.. _eta-term-laplacian:

eta-term laplacian
------------------

Considering that vector laplacian of spherically symmetric vector function (in 3-D space) is:

.. math::

    \Delta (f(r)\mathbf{r}) = \left(f''(r) + \frac{4}{r} f'(r)\right) \mathbf{r}

There are only two non-zero terms of :math:`\eta(r_{ij})\mathbf{r}_{ij}` laplacian, i.e. by :math:`i`-th or :math:`j`-th electron coordinates:

.. math::

    \Delta_{e_i} (\eta(r_{ij})\mathbf{r}_{ij}) = f(r_{ij}; L_\eta) \mathbf{r}_{ij}\sum_{k=0}^{N_\eta} \left[\frac{C(C-1)}{(L_\eta - r_{ij})^2} - \frac{2C(k+2)}{r_{ij}(L_\eta - r_{ij})} + \frac{k(k+3)}{r_{ij}^2} \right] c_kr^k_{ij}

.. math::

    \Delta_{e_j} (\eta(r_{ij})\mathbf{r}_{ij}) = - \Delta_{e_i} (\eta(r_{ij})\mathbf{r}_{ij})

For certain electron coordinates, :math:`\eta` laplacian term can be obtained with :py:meth:`casino.Backflow.eta_term_laplacian` method::

    backflow.eta_term_laplacian(e_powers, e_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval

.. _mu-term-laplacian:

mu-term laplacian
-----------------

Considering that vector laplacian of spherically symmetric vector function (in 3-D space) is:

.. math::

    \Delta (f(r)\mathbf{r}) = \left(f''(r) + \frac{4}{r} f'(r)\right) \mathbf{r}

There is only one non-zero term of :math:`\mu(r_{iI})\mathbf{r}_{iI}` laplacian, i.e. by :math:`i`-th electron coordinates:

.. math::

    \Delta_{e_i} (\mu(r_{iI})\mathbf{r}_{iI}) = f(r_{iI}; L_\mu) \mathbf{r}_{iI}\sum_{k=0}^{N_\mu} \left[\frac{C(C-1)}{(L_\mu - r_{iI})^2} - \frac{2C(k+2)}{r_{iI}(L_\mu - r_{iI})} + \frac{k(k+3)}{r_{iI}^2} \right]d_kr^k_{iI}

For certain electron coordinates, :math:`\mu` term laplacian can be obtained with :py:meth:`casino.Backflow.mu_term_laplacian` method::

    backflow.mu_term_laplacian(n_powers, n_vectors)[1]

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval

.. _phi-term-laplacian:

phi-term laplacian
------------------

Considering that gradient of spherically symmetric scalar function (in 3-D space) is:

.. math::

    \nabla f(r) = f'(r) \mathbf{\hat r}

and laplacian of spherically symmetric scalar function (in 3-D space) is:

.. math::

    \Delta f(r) = f''(r) + \frac{2}{r} f'(r)

Each addend of the :math:`\Phi` term is a product of :math:`p=r_{ij}^m`, :math:`q=f(r_{iI}; L_{\Phi I})r_{iI}^k`,
:math:`s=f(r_{jI}; L_{\Phi I})r_{jI}^l`, :math:`\phi_{klmI}` and :math:`\mathbf{r}_{ij}` so using:

.. math::

    \frac{\Delta_{e_i} \Phi}{\Phi} = \frac{\Delta_{e_i} (pq\mathbf{r}_{ij})}{pq} = \left(\frac{\Delta_{e_i} p}{p} + 2\frac{\nabla_{e_i} p}{p} \cdot \frac{\nabla_{e_i} q}{q} + \frac{\Delta_{e_i} q}{q}\right)\mathbf{r}_{ij} + 2\left(\frac{\nabla_{e_i} p}{p} + \frac{\nabla_{e_i} q}{q}\right) \nabla_{e_i} \mathbf{r}_{ij}

.. math::

    \frac{\Delta_{e_j} \Phi}{\Phi} = \frac{\Delta_{e_j} (ps\mathbf{r}_{ij})}{ps} = \left(\frac{\Delta_{e_j} p}{p} + 2\frac{\nabla_{e_j} p}{p} \cdot \frac{\nabla_{e_j} s}{s} + \frac{\Delta_{e_j} s}{s}\right)\mathbf{r}_{ij} + 2\left(\frac{\nabla_{e_j} p}{p} + \frac{\nabla_{e_j} s}{s}\right) \nabla_{e_j} \mathbf{r}_{ij}

There are only two non-zero terms of :math:`\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}` laplacian, i.e. by :math:`i`-th:

.. math::

    l_{kmI} = \frac{\Delta_{e_i} p}{p} + \frac{\Delta_{e_i} q}{q} = \left( \frac{m(m+1)}{r_{ij}^2}  + \frac{k(k+1)}{r_{iI}^2} + \frac{C(C-1)}{(L_{\Phi I} - r_{iI})^2} - \frac{2C(k+1)}{r_{iI}(L_{\Phi I} - r_{iI})} \right)

.. math::

    d_{kmI} = \frac{\nabla_{e_i} p}{p} \cdot \frac{\nabla_{e_i} q}{q} = \frac{m}{r_{ij}} \left( \frac{k}{r_{iI}} - \frac{C}{L_{\Phi I} - r_{iI}} \right) (\mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{iI})

.. math::

    g_{kmI} = \frac{\nabla_{e_i} p}{p} + \frac{\nabla_{e_i} q}{q} = \frac{m}{r_{ij}} \mathbf{\hat r}_{ij} + \left( \frac{k}{r_{iI}} - \frac{C}{L_{\Phi I} - r_{iI}} \right)\mathbf{\hat r}_{iI}

.. math::

    \Delta_{e_i} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}} (l_{kmI}\mathbf{r}_{ij} + 2d_{kmI}\mathbf{r}_{ij} + 2g_{kmI}) \Phi_{klmI}

or :math:`j`-th electron coordinates:

.. math::

    l_{lmI} = \frac{\Delta_{e_j} p}{p} + \frac{\Delta_{e_j} s}{s} = \left( \frac{m(m+1)}{r_{ij}^2} + \frac{l(l+1)}{r_{jI}^2} + \frac{C(C-1)}{(L_{\Phi I} - r_{jI})^2} - \frac{2C(l+1)}{r_{jI}(L_{\Phi I} - r_{jI})} \right)

.. math::

    d_{lmI} = \frac{\nabla_{e_j} p}{p} \cdot \frac{\nabla_{e_j} s}{s} = - \frac{m}{r_{ij}} \left( \frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}} \right) (\mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{jI})

.. math::

    g_{lmI} = \left(\frac{\nabla_{e_j} p}{p} + \frac{\nabla_{e_j} s}{s}\right) \nabla_{e_j} \mathbf{r}_{ij} = \frac{m}{r_{ij}} \mathbf{\hat r}_{ij} - \left( \frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}} \right)\mathbf{\hat r}_{jI}

.. math::

    \Delta_{e_j} (\Phi(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{ij}) = \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}} (l_{lmI}\mathbf{r}_{ij} + 2d_{lmI}\mathbf{r}_{ij} + 2g_{lmI}) \Phi_{klmI}

Each addend of the :math:`\Theta` term is a product of :math:`p=r_{ij}^m`, :math:`q=f(r_{iI}; L_{\Phi I})r_{iI}^k`,
:math:`s=f(r_{jI}; L_{\Phi I})r_{jI}^l`, :math:`\theta_{klmI}` and :math:`\mathbf{r}_{iI}` so using:

.. math::

    \frac{\Delta_{e_i} \Theta}{\Theta} = \frac{\Delta_{e_i} (pq\mathbf{r}_{iI})}{pq} = \left(\frac{\Delta_{e_i} p}{p} + 2\frac{\nabla_{e_i} p}{p} \cdot \frac{\nabla_{e_i} q}{q} + \frac{\Delta_{e_i} q}{q}\right)\mathbf{r}_{iI} + 2\left(\frac{\nabla_{e_i} p}{p} + \frac{\nabla_{e_i} q}{q}\right) \nabla_{e_i} \mathbf{r}_{iI}

.. math::

    \frac{\Delta_{e_j} \Theta}{\Theta} = \frac{\Delta_{e_j} (ps\mathbf{r}_{iI})}{ps} = \left(\frac{\Delta_{e_j} p}{p} + 2\frac{\nabla_{e_j} p}{p} \cdot \frac{\nabla_{e_j} s}{s} + \frac{\Delta_{e_j} s}{s}\right)\mathbf{r}_{iI} + 2\left(\frac{\nabla_{e_j} p}{p} + \frac{\nabla_{e_j} s}{s}\right) \nabla_{e_j} \mathbf{r}_{iI}

There are only two non-zero terms of :math:`\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}` laplacian, i.e. by :math:`i`-th:

.. math::

    l_{kmI} = \left( \frac{m(m+1)}{r_{ij}^2} + \frac{k(k+1)}{r_{iI}^2} + \frac{C(C-1)}{(L_{\Phi I} - r_{iI})^2} - \frac{2C(k+1)}{r_{iI}(L_{\Phi I} - r_{iI})} \right)

.. math::

    d_{kmI} = \frac{m}{r_{ij}} \left( \frac{k}{r_{iI}} - \frac{C}{L_{\Phi I} - r_{iI}} \right) (\mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{iI})

.. math::

    g_{kmI} = \frac{m}{r_{ij}} \mathbf{\hat r}_{ij} + \left( \frac{k}{r_{iI}} - \frac{C}{L_{\Phi I} - r_{iI}} \right)\mathbf{\hat r}_{iI}

.. math::

    \Delta_{e_i} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}} (l_{kmI}\mathbf{r}_{iI} + 2d_{kmI}\mathbf{r}_{iI} + 2g_{kmI}) \Theta_{klmI}

or :math:`j`-th electron coordinates (since :math:`\nabla_{e_j} \mathbf{r}_{iI} = 0` there is no :math:`g`-like term):

.. math::

    l_{lmI} = \left( \frac{m(m+1)}{r_{ij}^2} + \frac{l(l+1)}{r_{jI}^2} + \frac{C(C-1)}{(L_{\Phi I} - r_{jI})^2} - \frac{2C(l+1)}{r_{jI}(L_{\Phi I} - r_{jI})} \right)

.. math::

    d_{lmI} =  - \frac{m}{r_{ij}} \left( \frac{l}{r_{jI}} - \frac{C}{L_{\Phi I} - r_{jI}} \right) (\mathbf{\hat r}_{ij} \cdot \mathbf{\hat r}_{jI})

.. math::

    \Delta_{e_j} (\Theta(r_{iI}, r_{jI}, r_{ij})\mathbf{r}_{iI}) = \sum_{k=0}^{N_{\Phi I}^{eN}}\sum_{l=0}^{N_{\Phi I}^{eN}}\sum_{m=0}^{N_{\Phi I}^{ee}} (l_{lmI}\mathbf{r}_{iI} + 2d_{lmI}\mathbf{r}_{iI}) \Theta_{klmI}

with :math:`\Phi_{klmI}` and :math:`\Theta_{klmI}` as defined in the :ref:`phi-term <phi-term>` section.

For certain electron coordinates, :math:`\phi` term laplacian can be obtained with :py:meth:`casino.Backflow.phi_term_laplacian` method::

    backflow.phi_term_laplacian(e_powers, n_powers, e_vectors, n_vectors)

this is equivalent to (continues :ref:`from <backflow-intermediate-data>`)::

    from numpy.polynomial.polynomial import polyval3d

References
----------

.. [4] P. López Ríos, A. Ma, N. D. Drummond, M. D. Towler, and R. J. Needs,
   *Inhomogeneous backflow transformations in quantum Monte Carlo calculations*,
   Phys. Rev. E **74**, 066701 (2006).

.. [5] R. P. Feynman and M. Cohen,
   *Energy Spectrum of the Excitations in Liquid Helium*,
   Phys. Rev. **102**, 1189 (1956).

.. [6] Y. Kwon, D. M. Ceperley, and R. M. Martin,
   *Effects of backflow correlation in the three-dimensional electron gas: Quantum Monte Carlo study*,
   Phys. Rev. B **58**, 6800 (1998).

.. [7] M. Holzmann, D. M. Ceperley, C. Pierleoni, and K. Esler,
   *Backflow correlations for the electron gas and metallic hydrogen*,
   Phys. Rev. E **68**, 046707 (2003).

.. [8] T. Kato,
   *On the eigenfunctions of many-particle systems in quantum mechanics*,
   Commun. Pure Appl. Math. **10**, 151 (1957).

.. [9] D. Bressanini, G. Morosi, and S. Tarasco,
   *An investigation of nodal structures and the construction of trial wave functions*,
   J. Chem. Phys. **123**, 204109 (2005).

.. [10] M. Casula and S. Sorella,
   *Geminal wave functions with Jastrow correlation: A first application to atoms*,
   J. Chem. Phys. **119**, 6500 (2003).

.. [11] M. Bajdich, L. Mitas, G. Drobný, L. K. Wagner, and K. E. Schmidt,
   *Pfaffian Pairing Wave Functions in Electronic-Structure Quantum Monte Carlo Simulations*,
   Phys. Rev. Lett. **96**, 130201 (2006).
