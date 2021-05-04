#!/usr/bin/env python3
from sympy import *

r, Lg = symbols('r Lg')

ae_cutoff = (r/Lg)**2 * (6 - 8 * (r/Lg) + 3 * (r/Lg)**2)

print(simplify(diff(ae_cutoff, r)))

print(simplify(diff(r**2 * diff(ae_cutoff, r), r)/r**2))
