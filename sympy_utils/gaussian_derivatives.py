#!/usr/bin/env python3
from sympy import *
from spherical_harmonics import harmonics, x, y, z, alpha, r2, momentum_map


def gradient(momentum):
    """
    ∇(orb) = ∇(angular) * exp(-alpha*r2) - 2 * alpha * (x, y, z) * orb
    """
    for harmonic in harmonics[momentum]:
        orb = harmonic * exp(-alpha*r2)
        c = -2 * alpha
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * exp(-alpha*r2) + c * x * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * exp(-alpha*r2) + c * y * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * exp(-alpha*r2) + c * z * orb))
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
        c = -2 * alpha
        res = (
            simplify(diff(orb, x, x) - (
                (diff(harmonic, x, x) + c * (harmonic + 2*x*diff(harmonic, x)) + c**2 * x*x*harmonic) * exp(-alpha*r2)
            )),
            simplify(diff(orb, x, y) - (
                (diff(harmonic, x, y) + c * (y*diff(harmonic, x) + x*diff(harmonic, y)) + c**2 * x*y*harmonic) * exp(-alpha*r2))
            ),
            simplify(diff(orb, y, y) - (
                (diff(harmonic, y, y) + c * (harmonic + 2*y*diff(harmonic, y)) + c**2 * y*y*harmonic) * exp(-alpha*r2)
            )),
            simplify(diff(orb, x, z) - (
                (diff(harmonic, x, z) + c * (z*diff(harmonic, x) + x*diff(harmonic, z)) + c**2 * x*z*harmonic) * exp(-alpha*r2))
            ),
            simplify(diff(orb, y, z) - (
                (diff(harmonic, y, z) + c * (z*diff(harmonic, y) + y*diff(harmonic, z)) + c**2 * y*z*harmonic) * exp(-alpha*r2))
            ),
            simplify(diff(orb, z, z) - (
                (diff(harmonic, z, z) + c * (harmonic + 2*z*diff(harmonic, z)) + c**2 * z*z*harmonic) * exp(-alpha*r2)
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


def tressian(momentum):
    """
    tress(orb) = ...
    """
    for harmonic in harmonics[momentum]:
        orb = harmonic * exp(-alpha*r2)
        c = -2 * alpha
        res = (
            simplify(diff(orb, x, x, x) - (
                diff(harmonic, x, x, x) +
                3 * c * (diff(harmonic, x) + x * diff(harmonic, x, x)) +
                3 * c**2 * x * (harmonic + x * diff(harmonic, x)) +
                c**3 * x * x * x * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, x, x, y) - (
                diff(harmonic, x, x, y) +
                c * (diff(harmonic, y) + 2 * x * diff(harmonic, x, y) + y * diff(harmonic, x, x)) +
                c**2 * (y * harmonic + 2 * x * y * diff(harmonic, x) + x * x * diff(harmonic, y)) +
                c**3 * x * x * y * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, x, x, z) - (
                diff(harmonic, x, x, z) +
                c * (diff(harmonic, z) + 2 * x * diff(harmonic, x, z) + z * diff(harmonic, x, x)) +
                c ** 2 * (z * harmonic + 2 * x * z * diff(harmonic, x) + x * x * diff(harmonic, z)) +
                c ** 3 * x * x * z * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, x, y, y) - (
                diff(harmonic, x, y, y) +
                c * (diff(harmonic, x) + 2 * y * diff(harmonic, x, y) + x * diff(harmonic, y, y)) +
                c ** 2 * (x * harmonic + 2 * x * y * diff(harmonic, y) + y * y * diff(harmonic, x)) +
                c ** 3 * x * y * y * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, x, y, z) - (
                diff(harmonic, x, y, z) +
                c * (z * diff(harmonic, x, y) + y * diff(harmonic, x, z) + x * diff(harmonic, y, z)) +
                c ** 2 * (z * y * diff(harmonic, x) + z * x * diff(harmonic, y) + x * y * diff(harmonic, z)) +
                c ** 3 * x * y * z * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, x, z, z) - (
                diff(harmonic, x, z, z) +
                c * (diff(harmonic, x) + 2 * z * diff(harmonic, x, z) + x * diff(harmonic, z, z)) +
                c ** 2 * (x * harmonic + 2 * x * z * diff(harmonic, z) + z * z * diff(harmonic, x)) +
                c ** 3 * x * z * z * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, y, y, y) - (
                diff(harmonic, y, y, y) +
                3 * c * (diff(harmonic, y) + y * diff(harmonic, y, y)) +
                3 * c ** 2 * y * (harmonic + y * diff(harmonic, y)) +
                c ** 3 * y * y * y * harmonic
              ) * exp(-alpha*r2)),
            simplify(diff(orb, y, y, z) - (
                diff(harmonic, y, y, z) +
                c * (diff(harmonic, z) + 2 * y * diff(harmonic, y, z) + z * diff(harmonic, y, y)) +
                c ** 2 * (z * harmonic + 2 * y * z * diff(harmonic, y) + y * y * diff(harmonic, z)) +
                c ** 3 * y * y * z * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, y, z, z) - (
                diff(harmonic, y, z, z) +
                c * (diff(harmonic, y) + 2 * z * diff(harmonic, y, z) + y * diff(harmonic, z, z)) +
                c ** 2 * (y * harmonic + 2 * y * z * diff(harmonic, z) + z * z * diff(harmonic, y)) +
                c ** 3 * y * z * z * harmonic
            ) * exp(-alpha*r2)),
            simplify(diff(orb, z, z, z) - (
                diff(harmonic, z, z, z) +
                3 * c * (diff(harmonic, z) + z * diff(harmonic, z, z)) +
                3 * c ** 2 * z * (harmonic + z * diff(harmonic, z)) +
                c ** 3 * z * z * z * harmonic
            ) * exp(-alpha*r2)),
        )
        print("tressian({})=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]".format(momentum, *res))


if __name__ == "__main__":

    for m in 'spdfg':
        gradient(m)
        hessian(m)
        laplacian(m)
        tressian(m)
