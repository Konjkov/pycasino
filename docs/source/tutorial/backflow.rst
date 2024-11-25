.. _backflow:

Backflow
========

Backflow part of wavefunction is represented by the :class:`casino.Backflow` class.

It must be initialized from the configuration files::

    from casino.readers import CasinoConfig
    from casino.backflow import Backflow

    config_path = <path to a directory containing input file>
    config = CasinoConfig(config_path)
    config.read()
    backflow = Backflow(config)

Backflow class has a following methods:
