#!/usr/bin/env python3
"""Spherically averaged HF electron density around every nucleus.

The density is used only as a weight when profiles are compared or fitted: the optimized
polynomials are meaningless where no electrons are found.  Orbitals are evaluated with the
PyCasino Slater class (no Monte Carlo is run).  For e-e terms the pair-distance weight is
estimated by drawing independent electron positions from a sum of the spherical densities
around the nuclei (no exchange-correlation hole), which is sufficient for a weight.
"""

import json
import os

import numpy as np

from collect import ROOT, read_config, last_stage
from terms import HERE, load

from casino.slater import Slater  # noqa: E402

GRID = np.concatenate([np.linspace(0, 1, 101)[:-1], np.linspace(1, 12, 221)])
N_DIRECTIONS = 302


def directions(n):
    """Fibonacci sphere: nearly uniform directions."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5**0.5) * i
    s = np.sqrt(1 - z**2)
    return np.stack([s * np.cos(phi), s * np.sin(phi), z], axis=1)


DIRECTIONS = directions(N_DIRECTIONS)


class Density:
    def __init__(self, path):
        config = read_config(path, f'correlation.out.{last_stage(path)}')
        self.slater = Slater(config, None)
        self.neu, self.ned = config.input.neu, config.input.ned
        self.positions = config.wfn.atom_positions
        self.up = config.mdet.permutation_up[0]
        self.down = config.mdet.permutation_down[0]
        log_norm = np.sum(np.log(np.arange(1, self.neu + 1))) + np.sum(np.log(np.arange(1, self.ned + 1)))
        self.norm = np.exp(-log_norm / (self.neu + self.ned) / 2)

    def __call__(self, point):
        n_vectors = np.repeat((point - self.positions)[:, None, :], self.neu + self.ned, axis=1)
        wfn_u, wfn_d = self.slater.value_matrix(n_vectors)
        value = 0.0
        if self.neu:
            value += np.sum(wfn_u[self.up, 0] ** 2)
        if self.ned:
            value += np.sum(wfn_d[self.down, 0] ** 2)
        return value / self.norm**2

    def radial(self):
        """Spherical average of the density around every nucleus."""
        res = []
        for atom in range(self.positions.shape[0]):
            rho = np.zeros_like(GRID)
            for i, r in enumerate(GRID):
                for direction in DIRECTIONS:
                    rho[i] += self(self.positions[atom] + r * direction) / N_DIRECTIONS
            res.append(rho)
        return res

    def sample(self, n_samples=40000, seed=0):
        """Metropolis sampling of the one-electron density with log-uniform step lengths."""
        rng = np.random.default_rng(seed)
        point = self.positions[0] + 0.5
        value = self(point)
        samples = []
        for step in range(2 * n_samples):
            scale = 10 ** rng.uniform(-2, 0.5)
            new_point = point + scale * rng.normal(size=3)
            new_value = self(new_point)
            if new_value > rng.uniform() * value:
                point, value = new_point, new_value
            if step >= n_samples:
                samples.append(point)
        return np.array(samples)


def pair_distance_weight(samples, seed=1):
    """Histogram of distances between independent density samples on GRID."""
    rng = np.random.default_rng(seed)
    first = samples[rng.permutation(len(samples))]
    second = samples[rng.permutation(len(samples))]
    r = np.linalg.norm(first - second, axis=1)
    edges = np.concatenate([[0], (GRID[1:] + GRID[:-1]) / 2, [GRID[-1]]])
    hist, _ = np.histogram(r, bins=edges)
    return hist / np.diff(edges) / len(r)


def main():
    densities = {}
    for entry in load(['Jastrow_emin']):
        if entry['basis'] == 'stowfn-scan':
            continue
        key = f'{entry["basis"]}:{entry["system"]}'
        density = Density(os.path.join(ROOT, entry['path']))
        rho = density.radial()
        pair = pair_distance_weight(density.sample())
        total = [float(np.trapezoid(4 * np.pi * GRID**2 * x, GRID)) for x in rho]
        print(key, 'electrons within 12 bohr of each nucleus:', np.round(total, 3))
        densities[key] = {'grid': GRID.tolist(), 'rho': [x.tolist() for x in rho], 'pair': pair.tolist()}
    with open(os.path.join(HERE, 'data', 'densities.json'), 'w') as f:
        json.dump(densities, f)


if __name__ == '__main__':
    main()
