.. _installation:

:tocdepth: 2

Installation
============

Python version
--------------

Pycasino is compatible with Python 3.10 and higher.

Dependencies
------------

These packages will be installed automatically when installing Pycasino.

- `numba <https://github.com/numba/numba>`_ is an open source JIT compiler for subset of Python and NumPy functions.
- `numpy <https://github.com/numpy/numpy>`_ is fundamental package for scientific computing with Python.
- `scipy <https://github.com/scipy/scipy>`_ - is package for mathematics, science, and engineering.
- `mpi4py <https://github.com/mpi4py/mpi4py>`_ - Python bindings for the Message Passing Interface (`MPI <https://www.mpi-forum.org/>`_) standard.

Before installing mpi4py package, you need to install the system library::

    $ sudo apt install libopenmpi-dev

Virtual environments
--------------------

It is a good practice to separate the dependencies of different Python projects with the use of virtual environments
which is easier to do with `pipx <https://github.com/pypa/pipx>`_.

Install Pycasino
----------------

The latest official release of Pycasino can be installed from the Python Package Index `Casino <https://pypi.org/project/casino/>`_::

    $ pipx install casino

Developing
----------

In order to have access to the source code and stay up-to-date with the latest developments,
Pycasino can be installed directly from the `GitHub repository <https://github.com/Konjkov/pycasino>`_.

To install Pycasino from the Git repository run::

    $ git clone https://github.com/Konjkov/pycasino
    $ cd pycasino
    $ pip install -e .[dev]

Note that the ``-e`` option installs the repository in editable mode and the ``.[dev]`` specification includes the optional dependencies for development.

If `Pip <https://pip.pypa.io/en/stable/getting-started/>`_ complains about ``setup.py`` not being found, please update pip to the latest version.

