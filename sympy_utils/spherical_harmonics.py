#!/usr/bin/env python3
from sympy import *

x, y, z, x2, y2, z2, alpha = symbols('x y z x2 y2 z2 alpha')

r2 = x*x + y*y + z*z

substitutions = {x**2: x2, y**2: y2, z**2: z2}

r = sqrt(x*x + y*y + z*z)

momentum_map = dict(s=0, p=1, d=2, f=3, g=4)

# https://www.chemeurope.com/en/encyclopedia/Solid_harmonics.html
harmonics = dict()

harmonics['s'] = (
    1,     # (0,  0)
)

harmonics['p'] = (
    x,     # (1,  1)
    y,     # (1, -1)
    z,     # (1,  0)
)

harmonics['d'] = (
    (3 * z*z - r2) / 2,    # (2,  0)
    3 * z*x,               # (2,  1)
    3 * y*z,               # (2, -1)
    3 * (x*x - y*y),       # (2,  2)
    6 * x*y,               # (2, -2)
)

harmonics['f'] = (
    (5 * z*z*z - 3 * z*r2) / 2,    # (3,  0)
    3 * (5 * x*z*z - x*r2) / 2,    # (3,  1)
    3 * (5 * y*z*z - y*r2) / 2,    # (3, -1)
    15 * (x*x*z - y*y*z),          # (3,  2)
    30 * x*y*z,                    # (3, -2)
    15 * (x*x*x - 3 * x*y*y),      # (3,  3)
    15 * (3 * x*x*y - y*y*y),      # (3, -3)
)

harmonics['g'] = (
    (35 * z*z*z*z - 30 * z*z*r2 + 3 * r2 * r2) / 8,           # (4,  0)
    5 * (7 * z*z*z*x - 3 * x*z*r2) / 2,                       # (4,  1)
    5 * (7 * z*z*z*y - 3 * y*z*r2) / 2,                       # (4, -1)
    15 * (7 * (x*x*z*z - y*y*z*z) - (x*x*r2 - y*y*r2)) / 2,   # (4,  2)
    15 * (7 * z*z*x*y - x*y*r2),                              # (4, -2)
    105 * (x*x*x*z - 3 * y*y*x*z),                            # (4,  3)
    105 * (3 * x*x*y*z - y*y*y*z),                            # (4, -3)
    105 * (x*x*x*x - 6 * x*x*y*y + y*y*y*y),                  # (4,  4)
    420 * (x*x*x*y - y*y*y*x),                                # (4, -4)
)


def derivatives_1():
    """first derivatives"""
    for momentum in 'spdfg':
        for harmonic in harmonics[momentum]:
            res = (
                simplify(diff(harmonic, x)).subs(substitutions),
                simplify(diff(harmonic, y)).subs(substitutions),
                simplify(diff(harmonic, z)).subs(substitutions),
            )
            print('{}, {}, {},'.format(*res))


def derivatives_2():
    """second derivatives"""
    for momentum in 'spdfg':
        for harmonic in harmonics[momentum]:
            res = (
                simplify(diff(harmonic, x, x)).subs(substitutions),
                simplify(diff(harmonic, x, y)).subs(substitutions),
                simplify(diff(harmonic, x, z)).subs(substitutions),
                simplify(diff(harmonic, y, y)).subs(substitutions),
                simplify(diff(harmonic, y, z)).subs(substitutions),
                simplify(diff(harmonic, z, z)).subs(substitutions),
            )
            print('{}, {}, {}, {}, {}, {},'.format(*res))


def derivatives_3():
    """third derivatives"""
    for momentum in 'spdfg':
        for harmonic in harmonics[momentum]:
            res = (
                simplify(diff(harmonic, x, x, x)),
                simplify(diff(harmonic, x, x, y)),
                simplify(diff(harmonic, x, x, z)),
                simplify(diff(harmonic, x, y, y)),
                simplify(diff(harmonic, x, y, z)),
                simplify(diff(harmonic, x, z, z)),
                simplify(diff(harmonic, y, y, y)),
                simplify(diff(harmonic, y, y, z)),
                simplify(diff(harmonic, y, z, z)),
                simplify(diff(harmonic, z, z, z)),
            )
            print('{}, {}, {}, {}, {}, {}, {}, {}, {}, {},'.format(*res))


if __name__ == "__main__":

    derivatives_1()
    print('-' * 99)
    derivatives_2()
    print('-' * 99)
    derivatives_3()
