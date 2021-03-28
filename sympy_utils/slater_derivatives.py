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
            simplify(diff(orb, x) - (diff(harmonic, x) * r**n * exp(-alpha*r) + (n/r - alpha) * x/r * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * r**n * exp(-alpha*r) + (n/r - alpha) * y/r * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * r**n * exp(-alpha*r) + (n/r - alpha) * z/r * orb))
        )
        print("gradient({})=[{}, {}, {}]".format(i, *res))


def hessian(momentum, n):
    """
    hess(orb) =
    """
    for i, harmonic in enumerate(harmonics[momentum]):
        l = momentum_map[momentum]
        orb = harmonic * r**n * exp(-alpha*r)
        res = (
            diff(orb, x, x) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * x*x/r2 * orb,
            diff(orb, x, y) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * x*y/r2 * orb,
            diff(orb, y, y) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * y*y/r2 * orb,
            diff(orb, x, z) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * x*z/r2 * orb,
            diff(orb, y, z) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * y*z/r2 * orb,
            diff(orb, z, z) - (alpha**2 - 2*(l+n+1)/r*alpha + (2*l+n+1)*n/r2) * z*z/r2 * orb
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
