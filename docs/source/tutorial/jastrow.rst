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

chi_term
--------

f_term
------

u_term_gradient
---------------

chi_term_gradient
-----------------

f_term_gradient
---------------

u_term_laplacian
----------------

chi_term_laplacian
------------------

f_term_laplacian
----------------
