.. _wfn:

Wavefunction
============

Wavefunction in SJB Slater-Jastrow-backflow form is represented by the :class:`casino.Wfn` class.

.. math::

    \Psi(\mathbf{r}) = e^{J(\mathbf{r})}\Phi(\mathbf{r} - \xi(\mathbf{r}))

where :math:`\xi(\mathbf{r})` is the backflow displacement of electrons, which depends on the configuration of the whole system.

Summary of Methods
------------------

Slater class has a following methods:

.. list-table::
   :widths: 30 30 40
   :header-rows: 1
   :width: 100%

   * - Method
     - Output
     - Shape
   * - :ref:`value <value>`
     - :math:`\Psi(r)`
     - :math:`scalar`
   * - :ref:`energy <energy>`
     - :math:`\hat H \Psi(r) / \Psi(r)`
     - :math:`scalar`

.. _value:

value
-----

In quantum chemistry, molecular orbitals (MOs) are normally expanded in a set of atom-centered basis functions, or localized atomic orbitals (AOs):

.. _energy:

energy
------

In quantum chemistry, molecular orbitals (MOs) are normally expanded in a set of atom-centered basis functions, or localized atomic orbitals (AOs):
