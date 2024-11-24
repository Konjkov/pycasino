.. _wfn:

Wavefunction
============

Wavefunction in SJB Slater-Jastrow-backflow form is represented by the :class:`casino.Wfn` class.

.. math::

    \Psi(\mathbf{r}) = e^{J(\mathbf{r})}\Phi(\mathbf{r} - \xi(\mathbf{r}))

where \xi(\mathbf{r}) is the backflow displacement of electrons, which depends on the configuration of the whole system.
