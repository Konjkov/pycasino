#!/usr/bin/env python3
from sympy import *

r, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4 = symbols('r alpha_0 alpha_1 alpha_2 alpha_3 alpha_4')

cusp = exp(alpha_0 + alpha_1 * r + alpha_2 * r**2 + alpha_3 * r**3 + alpha_4 * r**4)

print(simplify(diff(cusp, r)))

print(simplify(diff(r**2 * diff(cusp, r), r)/r**2))
