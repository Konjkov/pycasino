#!/usr/bin/env python3
from sympy import *

r, L, C = symbols('r L C')

poly = Function('f')

jastrow = dict(
    u=(r - L)**C * poly(r)
)


def gradient(term):
    res = diff(term, r)
    print("gradient={}".format(simplify(res)))


def laplacian(term):
    res = diff(r**2 * diff(term, r), r) / r**2
    print("laplacian={}".format(simplify(res)))


if __name__ == "__main__":
    gradient(jastrow['u'])
    laplacian(jastrow['u'])
