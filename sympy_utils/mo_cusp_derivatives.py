#!/usr/bin/env python3
from sympy import *

r, alpha, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, c = symbols('r alpha alpha_0 alpha_1 alpha_2 alpha_3 alpha_4 c')

cusp = exp(alpha_0 + alpha_1 * r + alpha_2 * r**2 + alpha_3 * r**3 + alpha_4 * r**4)

substitutions = {-2*alpha: c, 4*alpha**2: c**2}

print(simplify(diff(cusp, r)/cusp))

print(simplify(diff(r**2 * diff(cusp, r), r)/r**2/cusp))

print(simplify(diff(diff(cusp, r), r)/cusp))

print(simplify(diff(diff(diff(cusp, r), r), r)/cusp))

gauss = exp(-alpha * r**2)

print(simplify(diff(gauss, r).subs(substitutions)))

print(simplify(diff(diff(gauss, r), r).subs(substitutions)))

print(simplify(diff(diff(diff(gauss, r), r), r).subs(substitutions)))
