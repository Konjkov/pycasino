.. _cusp:

Cusp correction
===============

When Gaussian basis sets are used, molecular orbitals generally fail to satisfy the
electron–nucleus cusp condition at nuclear positions. The :class:`casino.Cusp` class implements
the correction scheme of Ma, Towler, Drummond and Needs [12]_.

Physical motivation
-------------------

The exact wavefunction must satisfy the Kato cusp condition [8]_ at every nucleus:

.. math::

    \left.\frac{\partial \langle\Psi\rangle}{\partial r_{iI}}\right|_{r_{iI}=0}
    = -Z_I \left.\langle\Psi\rangle\right|_{r_{iI}=0}

where :math:`Z_I` is the nuclear charge, :math:`r_{iI}` is the electron–nucleus distance and
:math:`\langle\Psi\rangle` is the spherical average of the wavefunction about the nuclear
position (the spherical average was later shown to be redundant [15]_). For a determinant of
orbitals it is sufficient that every orbital which is nonzero at a given nucleus obeys the
cusp condition there [12]_. Gaussian functions have zero derivative at the nucleus (they are
smooth at the origin), so a linear combination of Gaussians cannot satisfy this condition
directly: the local energy then diverges as :math:`-Z_I/r_{iI}` when an electron approaches
a nucleus and shows wild oscillations close to it, which increase the VMC variance and can
lead to severe bias or even numerical instabilities in DMC [12]_.

The Kato condition relates only the value and the first radial derivative of the wavefunction
at the coalescence point; higher-order coalescence conditions, constraining the second and
higher derivatives, are also known [13]_ [14]_ [16]_.

Correction scheme
-----------------

Each molecular orbital :math:`\psi` expanded in a Gaussian basis can be written as:

.. math::

    \psi = \phi + \eta

where :math:`\phi` is the contribution from s-type Gaussians centered on the nucleus in
question and :math:`\eta` contains all other contributions. The corrected orbital is:

.. math::

    \tilde{\psi} = \tilde{\phi} + \eta

Inside a cusp correction radius :math:`r_c`, the s-part is replaced by:

.. math::

    \tilde{\phi} = C + \text{sgn}[\tilde{\phi}(0)] \exp(p(r)) \equiv C + R(r)

where :math:`C` is a shift chosen so that :math:`\tilde{\phi} - C` is of one sign within
:math:`r_c`, and :math:`p(r)` is a quartic polynomial:

.. math::

    p(r) = \alpha_0 + \alpha_1 r + \alpha_2 r^2 + \alpha_3 r^3 + \alpha_4 r^4

Since :math:`\eta` is cuspless (it arises from Gaussians of nonzero angular momentum centered
on the nucleus in question and from tails of Gaussians centered on other sites), the cusp
condition for the corrected orbital reads [12]_:

.. math::

    \left.\frac{d\tilde{\phi}}{dr}\right|_{r=0} = -Z\left(\tilde{\phi}(0) + \eta(0)\right)

The five coefficients :math:`\alpha_i` are fixed by five constraints [12]_: continuity of
:math:`\tilde{\phi}` and of its first and second derivatives at :math:`r = r_c`, the cusp
condition at :math:`r = 0`, and a chosen value of :math:`\tilde{\phi}(0)`. With these
constraints the equations for :math:`\alpha_i` are solved analytically, and
:math:`\tilde{\phi}(0)` is then varied to minimise the maximum square deviation of the
effective one-electron local energy

.. math::

    E_L^s(r) = \tilde{\phi}^{-1}\left[-\frac{1}{2}\nabla^2 - \frac{Z_{\text{eff}}}{r}\right]
    \tilde{\phi}, \qquad
    Z_{\text{eff}} = Z\left(1 + \frac{\eta(0)}{C + R(0)}\right)

from an "ideal" curve :math:`E_L^{\text{ideal}}(r) = Z_{\text{eff}}^2(\beta_0 + \beta_1 r^2
+ \beta_2 r^3 + \ldots + \beta_7 r^7)`, whose coefficients :math:`\beta_1 \ldots \beta_7` were obtained by
fitting to the 1s orbital of the carbon atom (:math:`\beta_0` depends on the particular atom
and its environment) [12]_.

Choice of the correction radius
-------------------------------

The maximum possible cusp correction radius is :math:`r_{c,\max} = 1/Z`, further reduced to
:math:`0.9\,d/Z` when the nearest-neighbour distance :math:`d` is smaller than 1 bohr. The
actual :math:`r_c` for each orbital and nucleus is set to the largest radius less than
:math:`r_{c,\max}` at which the deviation of the effective one-electron local energy of the
uncorrected orbital from the ideal curve exceeds :math:`Z^2/c_c`, where :math:`c_c` is a
universal parameter with a default value of 50, set by the ``cusp_control`` keyword [12]_. It is then semi-optimized by varying it
by :math:`\pm 20\%` in nine steps and keeping the value which, together with its own optimum
:math:`\tilde{\phi}(0)`, minimises the maximum deviation from the ideal curve.

Both searches are carried out on a radial grid of spacing 0.0005 bohr, on which the nodes of
the s-part are also located by bisection. The local energy of the uncorrected orbital diverges
at a node, so grid points within 0.02 bohr of a node take no part in the fit, and :math:`r_c`
is moved inwards if it falls in such a region. Whenever a node remains inside :math:`r_c`, the
shift is set to twice the extremum of the s-part over the grid, which is what makes
:math:`\tilde{\phi} - C` of one sign there; otherwise :math:`C = 0`.

Enabling cusp correction
------------------------

Cusp correction is applied automatically when ``atom_basis_type : gaussian`` is set in the
``input`` file (it is disabled for Slater-type orbitals, which can satisfy the cusp condition
exactly).

To obtain the cusp polynomial coefficients from CASINO for comparison, add to your ``input``::

    cusp_info : T

and set ``POLYPRINT=.true.`` in ``gaussians.f90`` before compiling CASINO.

Initialisation
--------------

:class:`casino.Cusp` is created by ``casino.cusp.CuspFactory`` when cusp correction is enabled
and passed to :class:`casino.Slater`. It is not normally instantiated directly by the user.
The correction is applied transparently inside ``slater.value_matrix``,
``slater.gradient_matrix``, and higher derivatives.

Properties
----------

``orbital_sign``
    Array of shape ``(natom, norbitals)`` storing the sign of each corrected orbital at each
    nuclear position.

``shift``
    Array of shape ``(natom, norbitals)`` storing the constant shift :math:`C` for each
    orbital–nucleus pair.

``rc``
    Array of shape ``(natom, norbitals)`` storing the cusp correction radii.

``alpha``
    Array of shape ``(5, natom, norbitals)`` storing the quartic polynomial coefficients
    :math:`\alpha_0 \ldots \alpha_4`.

References
----------

.. [8] T. Kato,
   *On the eigenfunctions of many-particle systems in quantum mechanics*,
   Commun. Pure Appl. Math. **10**, 151 (1957).

.. [12] A. Ma, M. D. Towler, N. D. Drummond, and R. J. Needs,
   *Scheme for adding electron–nucleus cusps to Gaussian orbitals*,
   J. Chem. Phys. **122**, 224322 (2005).

.. [13] V. A. Rassolov and D. M. Chipman,
   *Behavior of electronic wave functions near cusps*,
   J. Chem. Phys. **104**, 9908 (1996).

.. [14] Y. I. Kurokawa, H. Nakashima, and H. Nakatsuji,
   *General coalescence conditions for the exact wave functions: higher-order relations
   for Coulombic and non-Coulombic systems*,
   Adv. Quantum Chem. **73**, 59 (2016).

.. [15] P. V. Tóth,
   *Boundary conditions for many-electron systems*,
   arXiv:1010.2700 [quant-ph] (2011).

.. [16] J. Karwowski and A. Savin,
   *Two-particle coalescence conditions revisited*,
   arXiv:2204.09897 [physics.chem-ph] (2022).
