#!/usr/bin/env python3
from sympy import *

from spherical_harmonics import harmonics, x, y, z, alpha, r, r2, momentum_map


def gradient(momentum, n):
    """
    ∇(orb) = ∇(angular) * r**n * exp(-alpha*r) + (n/r2 - alpha/r) * (x, y, z) * orb
    """
    for harmonic in harmonics[momentum]:
        orb = harmonic * r**n * exp(-alpha*r)
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * r**n * exp(-alpha*r) + (n/r - alpha) * x/r * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * r**n * exp(-alpha*r) + (n/r - alpha) * y/r * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * r**n * exp(-alpha*r) + (n/r - alpha) * z/r * orb))
        )
        print("gradient({})=[{}, {}, {}]".format(momentum, *res))


def hessian(momentum, n):
    """
    hess(orb) =
    """
    for harmonic in harmonics[momentum]:
        l = momentum_map[momentum]
        orb = harmonic * r**n * exp(-alpha*r)
        res = (
            simplify(diff(orb, x, x) - (
                (diff(harmonic, x, x) + diff(harmonic, x) * (n/r - alpha) * 2*x/r) * r**n * exp(-alpha*r) +
                (diff((n/r - alpha) * x/r, x) + (n/r - alpha) * x/r * (n/r - alpha) * x/r) * orb
            )),
            simplify(diff(orb, x, y) - (
                (diff(harmonic, x, y) + diff(harmonic, x) * (n/r - alpha) * y/r + diff(harmonic, y) * (n/r - alpha) * x/r) * r**n * exp(-alpha*r) +
                (diff((n/r - alpha) * x/r, y) + (n/r - alpha) * x/r * (n/r - alpha) * y/r) * orb
            )),
            simplify(diff(orb, y, y) - (
                (diff(harmonic, y, y) + diff(harmonic, y) * (n/r - alpha) * 2*y/r) * r**n * exp(-alpha*r) +
                (diff((n/r - alpha) * y/r, y) + (n/r - alpha) * y/r * (n/r - alpha) * y/r) * orb
            )),
            simplify(diff(orb, x, z) - (
                (diff(harmonic, x, z) + diff(harmonic, x) * (n/r - alpha) * z/r + diff(harmonic, z) * (n/r - alpha) * x/r) * r**n * exp(-alpha*r) +
                (diff((n/r - alpha) * x/r, z) + (n/r - alpha) * x/r * (n/r - alpha) * z/r) * orb
            )),
            simplify(diff(orb, y, z) - (
                (diff(harmonic, y, z) + diff(harmonic, y) * (n/r - alpha) * z/r + diff(harmonic, z) * (n/r - alpha) * y/r) * r**n * exp(-alpha*r) +
                (diff((n/r - alpha) * y/r, z) + (n/r - alpha) * y/r * (n/r - alpha) * z/r) * orb
            )),
            simplify(diff(orb, z, z) - (
                (diff(harmonic, z, z) + diff(harmonic, z) * (n/r - alpha) * 2*z/r) * r**n * exp(-alpha*r) +
                (diff((n/r - alpha) * z/r, z) + (n/r - alpha) * z/r * (n/r - alpha) * z/r) * orb
            )),
        )
        print("hessian({})=[{}, {}, {}, {}, {}, {}]".format(momentum, *res))


def laplacian(momentum, n):
    """
    ∇(orb) = (alpha**2 - 2*alpha*(l+n+1)/r + (2*l+n+1)*n/r2) * orb
    """
    for harmonic in harmonics[momentum]:
        l = momentum_map[momentum]
        orb = harmonic * r**n * exp(-alpha*r)
        lap = simplify(diff(orb, x, x) + diff(orb, y, y) + diff(orb, z, z) - (alpha**2 - 2*(l+n+1)*alpha/r + (2*l+n+1)*n/r2) * orb)
        print("laplacian({})={}".format(momentum, lap))


if __name__ == "__main__":

    for n in range(4):
        for m in 'spdfg':
            # gradient(m, n)
            hessian(m, n)
            # laplacian(m, n)
