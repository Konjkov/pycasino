#!/usr/bin/env python3
from sympy import *

from spherical_harmonics import harmonics, x, y, z, alpha, r, r2, momentum_map


def gradient(momentum, n):
    """
    ∇(orb) = ∇(angular) * r**n * exp(-alpha*r) + (n/r2 - alpha/r) * (x, y, z) * orb
    """
    for harmonic in harmonics[momentum]:
        minus_alpha_r = -alpha * r
        orb = harmonic * r**n * exp(minus_alpha_r)
        c = (minus_alpha_r + n)/r2
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * r**n * exp(minus_alpha_r) + c * x * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * r**n * exp(minus_alpha_r) + c * y * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * r**n * exp(minus_alpha_r) + c * z * orb))
        )
        print("gradient({}, {})=[{}, {}, {}]".format(momentum, n, *res))


def hessian(momentum, n):
    """
    hess(orb) = (hess(angular) - c * A * orb + (c + d) * B * angular) * exp(-alpha*r) - 2 * alpha * orb * I
    A = (x, y, z) x ∇(angular) + ∇(angular) x (x, y, z)
    B = (x, y, z) x (x, y, z)
    """
    for harmonic in harmonics[momentum]:
        minus_alpha_r = -alpha * r
        orb = harmonic * r**n * exp(minus_alpha_r)
        c = (minus_alpha_r + n)/r2
        d = c**2 - c/r2 - n/r2**2
        res = (
            simplify(diff(orb, x, x) - (
                (diff(harmonic, x, x) + c * (harmonic + 2*x*diff(harmonic, x)) + d * x*x*harmonic) * r**n * exp(-alpha*r)
            )),
            simplify(diff(orb, x, y) - (
                (diff(harmonic, x, y) + c * (y*diff(harmonic, x) + x*diff(harmonic, y)) + d * x*y*harmonic) * r**n * exp(-alpha*r)
            )),
            simplify(diff(orb, y, y) - (
                (diff(harmonic, y, y) + c * (harmonic + 2*y*diff(harmonic, y)) + d * y*y*harmonic) * r**n * exp(-alpha*r)
            )),
            simplify(diff(orb, x, z) - (
                (diff(harmonic, x, z) + c * (z*diff(harmonic, x) + x*diff(harmonic, z)) + d * x*z*harmonic) * r**n * exp(-alpha*r)
            )),
            simplify(diff(orb, y, z) - (
                (diff(harmonic, y, z) + c * (z*diff(harmonic, y) + y*diff(harmonic, z)) + d * y*z*harmonic) * r**n * exp(-alpha*r)
            )),
            simplify(diff(orb, z, z) - (
                (diff(harmonic, z, z) + c * (harmonic + 2*z*diff(harmonic, z)) + d * z*z*harmonic) * r**n * exp(-alpha*r)
            )),
        )
        print("hessian({}, {})=[{}, {}, {}, {}, {}, {}]".format(momentum, n, *res))


def laplacian(momentum, n):
    """
    ∇(orb) = (alpha**2 - 2*alpha*(l+n+1)/r + (2*l+n+1)*n/r2) * orb
    """
    for harmonic in harmonics[momentum]:
        l = momentum_map[momentum]
        minus_alpha_r = -alpha * r
        c = (minus_alpha_r + n)/r2
        d = c**2 + 2*(l+1)*c/r2 - n/r2**2
        orb = harmonic * r**n * exp(minus_alpha_r)
        lap = simplify(diff(orb, x, x) + diff(orb, y, y) + diff(orb, z, z) - d * r2 * orb)
        print("laplacian({}, {})={}".format(momentum, n, lap))


def tressian(momentum, n):
    """
    tress(orb) = ...
    """
    for harmonic in harmonics[momentum]:
        minus_alpha_r = -alpha * r
        orb = harmonic * r**n * exp(minus_alpha_r)
        c = (minus_alpha_r + n)/r2
        d = c**2 - c/r2 - n/r2**2
        e = c**3 - 3*c**2/r2 - 3*(n-1)*c/r2**2 + n*5/r2**3
        res = (
            simplify(diff(orb, x, x, x) - (
                diff(harmonic, x, x, x) +
                3 * c * (diff(harmonic, x) + x * diff(harmonic, x, x)) +
                3 * d * x * (harmonic + x * diff(harmonic, x)) +
                e * x * x * x * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, x, x, y) - (
                diff(harmonic, x, x, y) +
                c * (diff(harmonic, y) + 2 * x * diff(harmonic, x, y) + y * diff(harmonic, x, x)) +
                d * (y * harmonic + 2 * x * y * diff(harmonic, x) + x * x * diff(harmonic, y)) +
                e * x * x * y * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, x, x, z) - (
                diff(harmonic, x, x, z) +
                c * (diff(harmonic, z) + 2 * x * diff(harmonic, x, z) + z * diff(harmonic, x, x)) +
                d * (z * harmonic + 2 * x * z * diff(harmonic, x) + x * x * diff(harmonic, z)) +
                e * x * x * z * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, x, y, y) - (
                diff(harmonic, x, y, y) +
                c * (diff(harmonic, x) + 2 * y * diff(harmonic, x, y) + x * diff(harmonic, y, y)) +
                d * (x * harmonic + 2 * x * y * diff(harmonic, y) + y * y * diff(harmonic, x)) +
                e * x * y * y * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, x, y, z) - (
                diff(harmonic, x, y, z) +
                c * (z * diff(harmonic, x, y) + y * diff(harmonic, x, z) + x * diff(harmonic, y, z)) +
                d * (z * y * diff(harmonic, x) + z * x * diff(harmonic, y) + x * y * diff(harmonic, z)) +
                e * x * y * z * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, x, z, z) - (
                diff(harmonic, x, z, z) +
                c * (diff(harmonic, x) + 2 * z * diff(harmonic, x, z) + x * diff(harmonic, z, z)) +
                d * (x * harmonic + 2 * x * z * diff(harmonic, z) + z * z * diff(harmonic, x)) +
                e * x * z * z * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, y, y, y) - (
                diff(harmonic, y, y, y) +
                3 * c * (diff(harmonic, y) + y * diff(harmonic, y, y)) +
                3 * d * y * (harmonic + y * diff(harmonic, y)) +
                e * y * y * y * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, y, y, z) - (
                diff(harmonic, y, y, z) +
                c * (diff(harmonic, z) + 2 * y * diff(harmonic, y, z) + z * diff(harmonic, y, y)) +
                d * (z * harmonic + 2 * y * z * diff(harmonic, y) + y * y * diff(harmonic, z)) +
                e * y * y * z * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, y, z, z) - (
                diff(harmonic, y, z, z) +
                c * (diff(harmonic, y) + 2 * z * diff(harmonic, y, z) + y * diff(harmonic, z, z)) +
                d * (y * harmonic + 2 * y * z * diff(harmonic, z) + z * z * diff(harmonic, y)) +
                e * y * z * z * harmonic
            ) * r**n * exp(-alpha*r)),
            simplify(diff(orb, z, z, z) - (
                diff(harmonic, z, z, z) +
                3 * c * (diff(harmonic, z) + z * diff(harmonic, z, z)) +
                3 * d * z * (harmonic + z * diff(harmonic, z)) +
                e * z * z * z * harmonic
            ) * r**n * exp(-alpha*r)),
        )
        print("tressian({}, {})=[{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]".format(momentum, n, *res))


if __name__ == "__main__":

    for n in range(4):
        for m in 'spdfg':
            gradient(m, n)
            hessian(m, n)
            laplacian(m, n)
            tressian(m, n)
