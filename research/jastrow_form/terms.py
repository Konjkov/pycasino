"""Evaluation of CASINO polynomial Jastrow and backflow terms on radial grids.

The expressions follow casino/jastrow.py and casino/backflow.py:
    u(r)   = (r - L)^C  sum_k alpha_k r^k                     e-e, r < L
    chi(r) = (r - L)^C  sum_k beta_k r^k                      e-n, r < L
    f      = (r1 - L)^C (r2 - L)^C sum gamma_lmn r1^l r2^m r12^n
    eta(r) = (1 - r/L)^C sum_k c_k r^k                        displacement eta(r_ij) r_ij
    mu(r)  = (1 - r/L)^C sum_k d_k r^k                        displacement mu(r_iI) r_iI
    Phi, Theta = (1 - r1/L)^C (1 - r2/L)^C sum p_klm r1^k r2^l r12^m
"""

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(HERE, '..', '..'))
PLOTS = os.path.join(HERE, 'plots')

ELEMENTS = ['', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar']
ELEMENTS += ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr']

# e-e cusp: du/dr(0) = 1/4 for parallel, 1/2 for antiparallel spins
U_SPIN_LABELS = {1: ['uu', 'ud'], 2: ['uu', 'ud', 'dd']}
U_CUSP = {'uu': 0.25, 'ud': 0.5, 'dd': 0.25}
CHI_SPIN_LABELS = {0: ['ud'], 1: ['u', 'd']}


def load(kinds=None):
    with open(os.path.join(HERE, 'data', 'parameters.json')) as f:
        entries = json.load(f)
    if kinds is None:
        return entries
    return [entry for entry in entries if entry['kind'] in kinds]


def polynomial(parameters, r):
    res = np.zeros_like(r)
    for k, p in enumerate(parameters):
        res += p * r**k
    return res


def u_profile(term, trunc, spin, r):
    L = term['cutoff']
    parameters = term['parameters'][spin]
    return np.where(r < L, (r - L) ** trunc * polynomial(parameters, r), 0.0)


def chi_profile(term, trunc, spin, r):
    L = term['cutoff']
    parameters = term['parameters'][spin]
    return np.where(r < L, (r - L) ** trunc * polynomial(parameters, r), 0.0)


def f_value(term, trunc, spin, r1, r2, r12):
    L = term['cutoff']
    parameters = np.array(term['parameters'][spin])
    poly = np.zeros(np.broadcast(r1, r2, r12).shape)
    for n in range(parameters.shape[0]):
        for m in range(parameters.shape[1]):
            for l in range(parameters.shape[2]):
                poly = poly + parameters[n, m, l] * r1**l * r2**m * r12**n
    inside = (r1 < L) & (r2 < L)
    return np.where(inside, (r1 - L) ** trunc * (r2 - L) ** trunc * poly, 0.0)


def eta_profile(term, trunc, spin, r):
    cutoffs = term['cutoff']
    L = cutoffs[spin % len(cutoffs)]
    parameters = term['parameters'][spin]
    return np.where(r < L, (1 - r / L) ** trunc * polynomial(parameters, r), 0.0)


def mu_profile(term, trunc, spin, r):
    L = term['cutoff']
    parameters = term['parameters'][spin]
    return np.where(r < L, (1 - r / L) ** trunc * polynomial(parameters, r), 0.0)


def phi_value(term, trunc, spin, r1, r2, r12, name='phi'):
    """Phi (coefficient of r_ij) or Theta (coefficient of r_iI) for electron 1 at r1 from the nucleus."""
    L = term['cutoff']
    parameters = np.array(term[name][spin])
    poly = np.zeros(np.broadcast(r1, r2, r12).shape)
    for m in range(parameters.shape[0]):
        for l in range(parameters.shape[1]):
            for k in range(parameters.shape[2]):
                poly = poly + parameters[m, l, k] * r1**k * r2**l * r12**m
    inside = (r1 < L) & (r2 < L)
    return np.where(inside, (1 - r1 / L) ** trunc * (1 - r2 / L) ** trunc * poly, 0.0)


def species_label(entry, labels):
    numbers = sorted({entry['atom_numbers'][i] for i in labels})
    return '+'.join(ELEMENTS[z] for z in numbers)


def species_z(entry, labels):
    return entry['atom_numbers'][labels[0]]


def short_name(entry):
    return f'{entry["basis"]}:{entry["system"]}'
