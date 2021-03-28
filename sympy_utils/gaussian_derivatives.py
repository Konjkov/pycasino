#!/usr/bin/env python3
from sympy import *
from spherical_harmonics import harmonics, x, y, z, alpha, r2, momentum_map


def gradient(momentum):
    """
    ∇(orb) = ∇(angular) * exp(-alpha*r2) - 2 * alpha * (x, y, z) * orb
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        orb = harmonic * exp(-alpha*r2)
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * exp(-alpha*r2) - 2 * alpha * x * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * exp(-alpha*r2) - 2 * alpha * y * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * exp(-alpha*r2) - 2 * alpha * z * orb))
        )
        print("gradient({})=[{}, {}, {}]".format(i, *res))


def hessian(momentum):
    """
    hess(orb) =
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        l = momentum_map[momentum]
        orb = harmonic * exp(-alpha*r2)
        res = (
            diff(orb, x, x),
            diff(orb, x, y),
            diff(orb, y, y),
            diff(orb, x, z),
            diff(orb, y, z),
            diff(orb, z, z)
        )
        print("hessian({})=[{}, {}, {}, {}, {}, {}]".format(i, *res))


def laplacian(momentum):
    """
    ∇²(orb) = 2*alpha * (2*alpha*r2 - 2*l - 3) * orb
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        l = momentum_map[momentum]
        orb = harmonic * exp(-alpha*r2)
        res = simplify(diff(orb, x, x) + diff(orb, y, y) + diff(orb, z, z) - (2 * alpha * (2 * alpha * r2 - 2 * l - 3)) * orb)
        print("laplacian({})={}".format(i, res))


if __name__ == "__main__":

    for m in 'spdfg':
        gradient(m)
        # hessian(m)
        laplacian(m)
