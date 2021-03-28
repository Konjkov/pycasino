#!/usr/bin/env python3
from sympy import *
from spherical_harmonics import harmonics, x, y, z, alpha, r2, momentum_map


def gradient(momentum):
    """
    ∇(orb) = ∇(angular) * exp(-alpha*r2) - 2 * alpha * (x, y, z) * orb
    """
    for harmonic in harmonics[momentum]:
        orb = harmonic * exp(-alpha*r2)
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * exp(-alpha*r2) - 2 * alpha * x * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * exp(-alpha*r2) - 2 * alpha * y * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * exp(-alpha*r2) - 2 * alpha * z * orb))
        )
        print("gradient({})=[{}, {}, {}]".format(momentum, *res))


def hessian(momentum):
    """
    hess(orb) =
    """
    for harmonic in harmonics[momentum]:
        l = momentum_map[momentum]
        orb = harmonic * exp(-alpha*r2)
        res = (
            simplify(diff(orb, x, x) - diff(diff(harmonic, x) * exp(-alpha*r2) - 2 * alpha * x * orb, x)),
            simplify(diff(orb, x, y) - diff(diff(harmonic, x) * exp(-alpha*r2) - 2 * alpha * x * orb, y)),
            simplify(diff(orb, y, y) - diff(diff(harmonic, y) * exp(-alpha*r2) - 2 * alpha * y * orb, y)),
            simplify(diff(orb, x, z) - diff(diff(harmonic, x) * exp(-alpha*r2) - 2 * alpha * x * orb, z)),
            simplify(diff(orb, y, z) - diff(diff(harmonic, y) * exp(-alpha*r2) - 2 * alpha * y * orb, z)),
            simplify(diff(orb, z, z) - diff(diff(harmonic, z) * exp(-alpha*r2) - 2 * alpha * z * orb, z))
        )
        print("hessian({})=[{}, {}, {}, {}, {}, {}]".format(momentum, *res))


def laplacian(momentum):
    """
    ∇²(orb) = 2*alpha * (2*alpha*r2 - 2*l - 3) * orb
    """
    for harmonic in harmonics[momentum]:
        l = momentum_map[momentum]
        orb = harmonic * exp(-alpha*r2)
        res = simplify(diff(orb, x, x) + diff(orb, y, y) + diff(orb, z, z) - (2 * alpha * (2 * alpha * r2 - 2 * l - 3)) * orb)
        print("laplacian({})={}".format(momentum, res))


if __name__ == "__main__":

    for m in 'spdfg':
        # gradient(m)
        hessian(m)
        # laplacian(m)
