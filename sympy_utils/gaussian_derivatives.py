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
        res = (
            simplify(diff(orb, x, x) - (
                (
                    diff(harmonic, x, x) -
                    2*alpha*x * diff(harmonic, x) -
                    2*alpha*x * diff(harmonic, x) +
                    2*alpha*x * 2*alpha*x * harmonic
                ) * exp(-alpha*r2) - 2 * alpha * orb
            )),
            simplify(diff(orb, x, y) - (
                (
                    diff(harmonic, x, y) -
                    2*alpha*y * diff(harmonic, x) -
                    2*alpha*x * diff(harmonic, y) +
                    2*alpha*x * 2*alpha*y * harmonic
                ) * exp(-alpha*r2))
            ),
            simplify(diff(orb, y, y) - (
                (
                    diff(harmonic, y, y) -
                    2*alpha*y * diff(harmonic, y) -
                    2*alpha*y * diff(harmonic, y) +
                    2*alpha*y * 2*alpha*y * harmonic
                ) * exp(-alpha*r2) - 2 * alpha * orb
            )),
            simplify(diff(orb, x, z) - (
                (
                    diff(harmonic, x, z) -
                    2*alpha*z * diff(harmonic, x) -
                    2*alpha*x * diff(harmonic, z) +
                    2*alpha*x * 2*alpha*z * harmonic
                ) * exp(-alpha*r2))
            ),
            simplify(diff(orb, y, z) - (
                (
                    diff(harmonic, y, z) -
                    2*alpha*z * diff(harmonic, y) -
                    2*alpha*y * diff(harmonic, z) +
                    2*alpha*y * 2*alpha*z * harmonic
                ) * exp(-alpha*r2))
            ),
            simplify(diff(orb, z, z) - (
                (
                    diff(harmonic, z, z) -
                    2*alpha*z * diff(harmonic, z) -
                    2*alpha*z * diff(harmonic, z) +
                    2*alpha*z * 2*alpha*z * harmonic
                ) * exp(-alpha*r2) - 2 * alpha * orb
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
