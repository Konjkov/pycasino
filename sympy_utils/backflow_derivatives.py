#!/usr/bin/env python3
from sympy import *

x, y, z, r, L, C = symbols('x y z r L C')

poly = Function('f')

backflow = dict(
    mu=(1 - r/L)**C * poly(r),
    eta=(1 - r/L)**C * poly(r),
    phi=(1 - r/L)**C * poly(r),
)


def gradient(term):
    """
    ∇(r*f) = ∇(r)*f + r*∇(f) = e*f + r*∇(f)

    """
    res = diff(term, r)
    print("gradient={}".format(res))


def laplacian(term):
    """
    ∇²(r*f) = ∇²(r)*f + 2*∇(r)*∇(f) + r*∇²(f) = 2*e*∇(f) + r*∇²(f)
    """
    res = [
        simplify(2 * diff(term, r) + x * diff(r**2 * diff(term, r), r) / r**2),
        simplify(2 * diff(term, r) + y * diff(r**2 * diff(term, r), r) / r**2),
        simplify(2 * diff(term, r) + z * diff(r**2 * diff(term, r), r) / r**2)
    ]
    print("laplacian=[{}, {}, {}]".format(*res))


if __name__ == "__main__":
    gradient(backflow['mu'])
    laplacian(backflow['mu'])
