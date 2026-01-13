# Pycasino

The Pycasino is python implementation of the well-known [Casino](https://vallico.net/casinoqmc/) program.

[![PyPI version](https://badge.fury.io/py/casino.svg)](https://badge.fury.io/py/casino)
[![Build Status](https://github.com/Konjkov/pycasino/actions/workflows/tests.yml/badge.svg)](https://github.com/Konjkov/pycasino/actions)

## Installation

Install Python package from pip repository:

`pipx install casino`

Or clone the repository and install the from source:

`pipx install .`

Create virtualenv and run tests:

`python3 -m venv .venv`

`source .venv/bin/activate`

`pip install -e .[dev]`

`pytest --pyargs casino.tests`

## Documentation.
https://casinoqmc.readthedocs.io/en/latest/
