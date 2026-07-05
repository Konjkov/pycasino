.. _cusp:

Cusp correction
===============

When Gaussian basis sets are used, molecular orbitals generally fail to satisfy the
electron–nucleus cusp condition at nuclear positions. The :class:`casino.Cusp` class implements
the correction scheme of `Ma, Towler, Drummond and Needs (J. Chem. Phys. 122, 224322, 2005)
<https://doi.org/10.1063/1.1940588>`_.

Physical motivation
-------------------

The exact wavefunction must satisfy the Kato cusp condition at every nucleus:

.. math::

    \left.\frac{\partial \Psi}{\partial r_{iI}}\right|_{r_{iI}=0} = -Z_I \Psi(r_{iI}=0)

where :math:`Z_I` is the nuclear charge and :math:`r_{iI}` is the electron–nucleus distance.
Gaussian functions have zero derivative at the nucleus (they are smooth at the origin), so
a linear combination of Gaussians cannot satisfy this condition directly.

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

    \tilde{\phi} = C + \text{sgn}[\tilde{\phi}(0)] \exp(p(r))

where :math:`C` is a shift chosen so that :math:`\tilde{\phi} - C` is of one sign within
:math:`r_c`, and :math:`p(r)` is a quartic polynomial:

.. math::

    p(r) = \alpha_0 + \alpha_1 r + \alpha_2 r^2 + \alpha_3 r^3 + \alpha_4 r^4

The five coefficients :math:`\alpha_i` are determined by requiring that :math:`\tilde{\psi}`
satisfies the cusp condition at :math:`r = 0`, and that :math:`\tilde{\psi}` and its first
three derivatives match :math:`\psi` at :math:`r = r_c`.

Enabling cusp correction
------------------------

Cusp correction is applied automatically when ``atom_basis_type : gaussian`` is set in the
``input`` file (it is disabled for Slater-type orbitals, which can satisfy the cusp condition
exactly). The correction radius :math:`r_c` is read from ``gwfn.data``.

To generate the required cusp polynomial coefficients from CASINO, add to your ``input``::

    cusp_info : T

and set ``POLYPRINT=.true.`` in ``gaussians.f90`` before compiling CASINO. Pycasino reads the
resulting ``cusp.data`` file automatically.

Initialisation
--------------

:class:`casino.Cusp` is constructed internally by :class:`casino.Slater` when cusp correction
is enabled. It is not normally instantiated directly by the user. The correction is applied
transparently inside ``slater.value_matrix``, ``slater.gradient_matrix``, and higher
derivatives.

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
