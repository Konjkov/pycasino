"""Candidate analytic forms for the radial Jastrow and backflow terms.

Every form is f(r) * w(r/L) with the smooth window
    w(x) = (1 - x)^C (1 + C x),  x < 1;  w = 0 otherwise.
w(0) = 1 and w'(0) = 0, so the value and slope at r = 0 are those of f alone and the
cusp (or no-cusp) condition is imposed on f without reference to the cutoff length L.
At x = 1 the window and its first C - 1 derivatives vanish, as for the CASINO (r - L)^C factor.

A form is a dict:
    params  - names of the shape parameters (L is added by the fitter)
    guess   - starting points for the shape parameters
    f       - f(r, p, channel); channel carries the cusp value (u), or whether the
              channel has a free slope at r = 0 (eta antiparallel, mu at an AE nucleus)
    meaning - physical interpretation of the parameters
    interpretable - False for purely empirical forms
"""

import numpy as np

TRUNC = 3


def window(r, L, C=TRUNC):
    x = np.minimum(r / L, 1.0)
    return (1 - x) ** C * (1 + C * x)


def value(form, r, p, channel, L):
    return form['f'](r, p, channel) * window(r, L)


def ring(r, r0, s):
    """Even Gaussian ring: a shell of radius r0 and width s with zero slope at r = 0."""
    return np.exp(-(((r - r0) / s) ** 2)) + np.exp(-(((r + r0) / s) ** 2))


# ---------- u(r_ij), channel = cusp value gamma: f'(0) = gamma


def u_exp(r, p, gamma):
    b = p[0]
    return -gamma * b * np.exp(-r / b)


def u_pade(r, p, gamma):
    b = p[0]
    return -gamma * b / (1 + r / b)


def u_rpa(r, p, gamma):
    F = p[0]
    x = np.maximum(r / F, 1e-12)
    return -2 * gamma * F * (-np.expm1(-x)) / x


def u_exp_cusp(r, p, gamma):
    A, b = p
    return (-A + (gamma - A / b) * r) * np.exp(-r / b)


def u_exp2(r, p, gamma):
    b1, A2, b2 = p
    return -gamma * b1 * np.exp(-r / b1) - A2 * (1 + r / b2) * np.exp(-r / b2)


U_FORMS = {
    'exp': {
        'params': ['b'],
        'guess': [[0.5], [1.0], [2.0]],
        'f': u_exp,
        'meaning': 'b: correlation-hole radius, u(0) = -gamma*b; the slope is the Kato cusp',
    },
    'pade': {
        'params': ['b'],
        'guess': [[0.5], [1.0], [2.0]],
        'f': u_pade,
        'meaning': 'b: hole radius; shifted Pade gamma*r/(1+r/b), tail -gamma*b^2/r',
    },
    'rpa': {
        'params': ['F'],
        'guess': [[0.5], [1.0], [2.0]],
        'f': u_rpa,
        'meaning': 'F: RPA screening length, u = -(A/r)(1-exp(-r/F)) with A = 2*gamma*F^2 from the cusp',
    },
    'exp_cusp': {
        'params': ['A', 'b'],
        'guess': [[0.3, 1.0], [0.6, 1.5], [1.0, 2.0]],
        'f': u_exp_cusp,
        'meaning': 'A: hole depth u(0) = -A, b: hole radius; the cusp is restored by the r*exp term',
    },
    'exp2': {
        'params': ['b1', 'A2', 'b2'],
        'guess': [[0.5, 0.2, 2.0], [1.0, 0.5, 3.0], [0.3, 0.1, 1.5]],
        'f': u_exp2,
        'meaning': 'b1: short-range cusp hole; A2, b2: depth and range of a cusp-less long-range tail',
    },
}

# ---------- chi(r_iI): f'(0) = 0 (the Kato e-n cusp is carried by the orbitals)


def chi_window(r, p, channel):
    return p[0] * np.ones_like(r)


def chi_gauss(r, p, channel):
    A, a = p
    return A * np.exp(-((r / a) ** 2))


def chi_yukawa(r, p, channel):
    A, a = p
    return A * (1 + r / a) * np.exp(-r / a)


def chi_lorentz(r, p, channel):
    A, a = p
    return A / (1 + (r / a) ** 2)


def chi_sech(r, p, channel):
    A, a = p
    return A / np.cosh(r / a)


def chi_yukawa2(r, p, channel):
    A1, a1, A2, a2 = p
    return A1 * (1 + r / a1) * np.exp(-r / a1) + A2 * (1 + r / a2) * np.exp(-r / a2)


CHI_FORMS = {
    'window': {
        'params': ['A'],
        'guess': [[2.0], [4.0], [1.0]],
        'f': chi_window,
        'meaning': 'A: one-body boost at the nucleus, L: radius of the compensation region (the window is the whole shape)',
    },
    'gauss': {
        'params': ['A', 'a'],
        'guess': [[2.0, 1.0], [4.0, 2.0], [1.0, 3.0]],
        'f': chi_gauss,
        'meaning': 'A: one-body boost at the nucleus, a: radius of the region where the e-e hole is compensated',
    },
    'yukawa': {
        'params': ['A', 'a'],
        'guess': [[2.0, 0.5], [4.0, 1.0], [1.0, 2.0]],
        'f': chi_yukawa,
        'meaning': 'A: boost at the nucleus, a: decay length of an exponential shell',
    },
    'lorentz': {
        'params': ['A', 'a'],
        'guess': [[2.0, 1.0], [4.0, 2.0], [1.0, 3.0]],
        'f': chi_lorentz,
        'meaning': 'A: boost at the nucleus, a: half-width, algebraic decay',
    },
    'sech': {
        'params': ['A', 'a'],
        'guess': [[2.0, 1.0], [4.0, 2.0], [1.0, 3.0]],
        'f': chi_sech,
        'meaning': 'A: boost at the nucleus, a: decay length',
    },
    'yukawa2': {
        'params': ['A1', 'a1', 'A2', 'a2'],
        'guess': [[2.0, 0.3, 1.0, 1.5], [4.0, 0.5, 2.0, 2.0], [1.0, 0.2, 1.0, 1.0]],
        'f': chi_yukawa2,
        'meaning': 'two shells: inner (A1, a1) and outer (A2, a2)',
    },
}

# ---------- eta(r_ij), channel = True for antiparallel (free slope), False for parallel (zero slope)


def eta_exp(r, p, antiparallel):
    A, b = p
    if antiparallel:
        return A * np.exp(-r / b)
    return A * (1 + r / b) * np.exp(-r / b)


def eta_gauss(r, p, antiparallel):
    A, b = p
    return A * np.exp(-((r / b) ** 2))


def eta_exp_osc(r, p, antiparallel):
    A, b, k = p
    return eta_exp(r, [A, b], antiparallel) * np.cos(k * r)


ETA_FORMS = {
    'exp': {
        'params': ['A', 'b'],
        'guess': [[0.05, 0.5], [0.1, 1.0], [-0.02, 1.0]],
        'f': eta_exp,
        'meaning': 'A: backflow strength at coalescence, b: range of the e-e backflow',
    },
    'gauss': {
        'params': ['A', 'b'],
        'guess': [[0.05, 0.5], [0.1, 1.0], [-0.02, 1.0]],
        'f': eta_gauss,
        'meaning': 'A: backflow strength at coalescence, b: range',
    },
    'exp_osc': {
        'params': ['A', 'b', 'k'],
        'guess': [[0.05, 1.0, 1.0], [0.1, 1.0, 2.0], [-0.02, 2.0, 1.0]],
        'f': eta_exp_osc,
        'meaning': 'damped oscillation: A amplitude, b damping length, k wave number',
    },
}

# ---------- mu(r_iI), channel = True at an AE nucleus (mu ~ r^2), False at a PP nucleus (zero slope)


def mu_shell(r, p, ae):
    A, a = p
    if ae:
        return A * (r / a) ** 2 * np.exp(-r / a)
    return A * (1 + r / a) * np.exp(-r / a)


def mu_ring(r, p, ae):
    A, r0, s = p
    res = A * ring(r, r0, s)
    if ae:
        res = res * (-np.expm1(-((r / s) ** 2)))
    return res


MU_FORMS = {
    'shell': {
        'params': ['A', 'a'],
        'guess': [[-0.1, 0.5], [-0.3, 1.0], [0.05, 1.0]],
        'f': mu_shell,
        'meaning': 'AE: A amplitude, extremum at r = 2a; PP: A value at the core, a decay length',
    },
    'ring': {
        'params': ['A', 'r0', 's'],
        'guess': [[-0.1, 1.0, 0.5], [-0.3, 2.0, 0.7], [0.05, 1.0, 1.0]],
        'f': mu_ring,
        'meaning': 'A: depth of the well, r0: radius of the displaced shell, s: its width',
    },
}

for forms in (U_FORMS, CHI_FORMS, ETA_FORMS, MU_FORMS):
    for form in forms.values():
        form.setdefault('interpretable', True)
