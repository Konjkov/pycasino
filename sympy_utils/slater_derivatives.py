#!/usr/bin/env python3
from sympy import *

from spherical_harmonics import harmonics, x, y, z, alpha, r, r2, momentum_map


def gradient(momentum, n):
    """
    ∇(orb) = ∇(angular) * r**n * exp(-alpha*r) + (n/r2 - alpha/r) * (x, y, z) * orb
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        orb = harmonic * r**n * exp(-alpha*r)
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * r**n * exp(-alpha*r) + (n/r2 - alpha/r) * x * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * r**n * exp(-alpha*r) + (n/r2 - alpha/r) * y * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * r**n * exp(-alpha*r) + (n/r2 - alpha/r) * z * orb))
        )
        print("gradient({})=[{}, {}, {}]".format(i, *res))


def hessian(momentum):
    """
    hess(orb) =
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        l = momentum_map[momentum]
        orb = harmonic * r**n * exp(-alpha*r)
        res = (
            diff(orb, x, x),
            diff(orb, x, y),
            diff(orb, y, y),
            diff(orb, x, z),
            diff(orb, y, z),
            diff(orb, z, z)
        )
        print("hessian({})=[{}, {}, {}, {}, {}, {}]".format(i, *res))


def laplacian(momentum, n):
    """
    ∇(orb) = (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * orb
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        l = momentum_map[momentum]
        orb = harmonic * r**n * exp(-alpha*r)
        lap = simplify(diff(orb, x, x) + diff(orb, y, y) + diff(orb, z, z) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * orb)
        print("laplacian({})={}".format(i, lap))


if __name__ == "__main__":
    for n in range(4):
        for m in 'spdfg':
            gradient(m, n)
            laplacian(m, n)
