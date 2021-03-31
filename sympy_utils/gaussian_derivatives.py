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
    hess(orb) = (hess(angular) - 2 * alpha * A * orb + 2 * alpha * 2 * alpha * B * angular) * exp(-alpha*r2) - 2 * alpha * orb * I
    A = (x, y, z) x ∇(angular) + ∇(angular) x (x, y, z)
    B = (x, y, z) x (x, y, z)
    """
    for harmonic in harmonics[momentum]:
        orb = harmonic * exp(-alpha*r2)
        c = -2*alpha
        res = (
            simplify(diff(orb, x, x) - (
                (diff(harmonic, x, x) + c * (x*diff(harmonic, x) + x*diff(harmonic, x)) + c**2 * x*x*harmonic) * exp(-alpha*r2) + c * orb
            )),
            simplify(diff(orb, x, y) - (
                (diff(harmonic, x, y) + c * (y*diff(harmonic, x) + x*diff(harmonic, y)) + c**2 * x*y*harmonic) * exp(-alpha*r2))
            ),
            simplify(diff(orb, y, y) - (
                (diff(harmonic, y, y) + c * (y*diff(harmonic, y) + y*diff(harmonic, y)) + c**2 * y*y*harmonic) * exp(-alpha*r2) + c * orb
            )),
            simplify(diff(orb, x, z) - (
                (diff(harmonic, x, z) + c * (z*diff(harmonic, x) + x*diff(harmonic, z)) + c**2 * x*z*harmonic) * exp(-alpha*r2))
            ),
            simplify(diff(orb, y, z) - (
                (diff(harmonic, y, z) + c * (z*diff(harmonic, y) + y*diff(harmonic, z)) + c**2 * y*z*harmonic) * exp(-alpha*r2))
            ),
            simplify(diff(orb, z, z) - (
                (diff(harmonic, z, z) + c * (z*diff(harmonic, z) + z*diff(harmonic, z)) + c**2 * z*z*harmonic) * exp(-alpha*r2) + c * orb
            ))
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
        gradient(m)
        hessian(m)
        laplacian(m)
