"""The eigenvalue identity of Mitas & Annaberdiyev Eq. (20) on a four-electron node.

Noninteracting Be: hydrogenic 1s and 2s at Z = 4, two electrons of each spin, H = sum(-1/2 grad^2
- Z/r). Psi = D(r1, r2) * D(r3, r4) is then an exact eigenstate with

    E = 2 * (-Z^2/2) + 2 * (-Z^2/8) = -20

exactly, so with Phi = 1 the third term of Eq. (15) vanishes and E_kin^nda + E_pot^nda must come
out at -20. The one-electron 2p toy of test_nodal.py checks the estimator on a plane in three
dimensions; this checks it on a determinant node in twelve.
"""

import numpy as np

Z = 4.0
NWALK = 20000
NSTEP = 500
DECORR = 5
EQUIL = 200
EPSILON = np.geomspace(0.002, 0.05, 10)


def orbitals(r):
    """1s and 2s and their radial derivatives"""
    e1 = np.exp(-Z * r)
    e2 = np.exp(-Z * r / 2)
    return e1, (1 - Z * r / 2) * e2, -Z * e1, -Z / 2 * (2 - Z * r / 2) * e2


def wfn(pos):
    """value and the 3N-dimensional gradient norm of Psi
    :param pos: walker positions - array(nwalk, 4, 3)
    """
    r = np.sqrt((pos * pos).sum(-1))
    s, p, ds, dp = orbitals(r)
    # D(a, b) = s(a) p(b) - p(a) s(b) for each spin pair
    a, b = (0, 2), (1, 3)
    d = s[:, a] * p[:, b] - p[:, a] * s[:, b]
    value = d[:, 0] * d[:, 1]
    # dD/da and dD/db, radial, then the other pair's D as a factor
    grad_a = ds[:, a] * p[:, b] - dp[:, a] * s[:, b]
    grad_b = s[:, a] * dp[:, b] - p[:, a] * ds[:, b]
    other = d[:, ::-1]
    grad2 = ((grad_a * other) ** 2).sum(-1) + ((grad_b * other) ** 2).sum(-1)
    return value, np.sqrt(grad2), r


def walk(step):
    """Metropolis walk distributed as |Psi|, yielding sigma and the potential of every sample"""
    pos = np.random.normal(scale=1 / Z, size=(NWALK, 4, 3))
    value, grad, r = wfn(pos)
    sigma = np.empty(shape=(NSTEP, NWALK))
    potential = np.empty(shape=(NSTEP, NWALK))
    accepted = 0
    for i in range(-EQUIL, NSTEP):
        for _ in range(DECORR):
            new_pos = pos + np.random.normal(scale=step, size=pos.shape)
            new_value, new_grad, new_r = wfn(new_pos)
            cond = np.abs(new_value) > np.abs(value) * np.random.random(NWALK)
            pos = np.where(cond[:, None, None], new_pos, pos)
            value = np.where(cond, new_value, value)
            grad = np.where(cond, new_grad, grad)
            r = np.where(cond[:, None], new_r, r)
            if i >= 0:
                accepted += cond.mean()
        if i >= 0:
            sigma[i] = np.abs(value) / grad
            potential[i] = (-Z / r).sum(-1)
    print(f'acceptance {accepted / (NSTEP * DECORR):.3f}')
    return sigma.ravel(), potential.ravel()


np.random.seed(1)
sigma, potential = walk(0.6 / Z)
n = sigma.size
e_pot = potential.mean()
e_pot_sem = potential.std() / np.sqrt(n)
print(f'{n} samples, E_pot^nda = {e_pot:.6f} +/- {e_pot_sem:.6f}')
print(f'\n{"epsilon":>9} {"inside":>8}   {"E_kin^nda":>12}                {"E^nda":>10}   exact -20')
for eps in EPSILON:
    inside = (sigma < eps).sum()
    e_kin = inside / n / eps**2
    print(f'{eps:9.4f} {inside:8d}   {e_kin:12.6f} +/- {np.sqrt(inside) / n / eps**2:.6f}   {e_kin + e_pot:12.6f}')
