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
        c = (minus_alpha_r + n)/r**2
        res = (
            simplify(diff(orb, x) - (diff(harmonic, x) * r**n * exp(minus_alpha_r) + c * x * orb)),
            simplify(diff(orb, y) - (diff(harmonic, y) * r**n * exp(minus_alpha_r) + c * y * orb)),
            simplify(diff(orb, z) - (diff(harmonic, z) * r**n * exp(minus_alpha_r) + c * z * orb))
        )
        print("gradient({}, {})=[{}, {}, {}]".format(momentum, n, *res))


def hessian(momentum, n):
    """
    hess(orb) =
    """
    for harmonic in harmonics[momentum]:
        minus_alpha_r = -alpha * r
        orb = harmonic * r**n * exp(minus_alpha_r)
        c = (minus_alpha_r + n)/r**2
        d = c**2 - (minus_alpha_r + 2*n)/r**4
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
        orb = harmonic * r**n * exp(minus_alpha_r)
        lap = simplify(diff(orb, x, x) + diff(orb, y, y) + diff(orb, z, z) - (minus_alpha_r**2 + 2*(l+n+1)*minus_alpha_r + (2*l+n+1)*n)/r2 * orb)
        print("laplacian({}, {})={}".format(momentum, n, lap))


def tressian(momentum, n):
    """
    tress(orb) =
    """
    for harmonic in harmonics[momentum]:
        minus_alpha_r = -alpha * r
        orb = harmonic * r**n * exp(minus_alpha_r)
        c = (minus_alpha_r + n)/r**2
        d = c**2 - (minus_alpha_r + 2*n)/r**4
        e = c * d - (2 * minus_alpha_r**2 + 3 * (2*n - 1)*minus_alpha_r + 4*n*(n-2)) / r**6
        res = (
            # simplify(diff(orb, x, x, x)),
            # simplify(diff(orb, x, x, y)),
            # simplify(diff(orb, x, x, z)),
            # simplify(diff(orb, x, y, y)),
            # simplify(2 * c**2*r**4 - (2 * minus_alpha_r**2 + 3 * (2*n - 1)*minus_alpha_r + 4*n*(n-2))),
            simplify(diff(orb, x, y, z) - (
                diff(harmonic, x, y, z) +
                c * (z * diff(harmonic, x, y) + y * diff(harmonic, x, z) + x * diff(harmonic, y, z)) +
                d * (z * y * diff(harmonic, x) + z * x * diff(harmonic, y) + x * y * diff(harmonic, z)) +
                e * x * y * z * harmonic
            ) * r**n * exp(-alpha*r)),
            # simplify(diff(orb, x, z, z)),
            # simplify(diff(orb, y, y, y)),
            # simplify(diff(orb, y, y, z)),
            # simplify(diff(orb, y, z, z)),
            # simplify(diff(orb, z, z, z)),
        )
        print("tressian({}, {})=[{}]".format(momentum, n, *res))
        # print("tressian({}, {})=[{}, {}, {}, {}, {}, {},  {}, {},  {}, {}]".format(momentum, n, *res))


if __name__ == "__main__":

    for n in range(4):
        for m in 'spdfg':
            # gradient(m, n)
            # hessian(m, n)
            # laplacian(m, n)
            tressian(m, n)
