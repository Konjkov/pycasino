"""Radial fitting targets and fitters shared by the analysis scripts."""

import json
import os

import numpy as np
from scipy.optimize import least_squares, minimize_scalar

from forms import TRUNC, value
from terms import CHI_SPIN_LABELS, HERE, U_CUSP, U_SPIN_LABELS, chi_profile, eta_profile, mu_profile, species_label, species_z, u_profile

LENGTH_PARAMS = {'a', 'a1', 'a2', 'b', 'b1', 'b2', 'F', 's', 'r0', 'k'}


def load_densities():
    with open(os.path.join(HERE, 'data', 'densities.json')) as f:
        return json.load(f)


def pair_weight(density):
    w = np.array(density['pair'])
    return w / w.max()


def en_weight(density, labels):
    r = np.array(density['grid'])
    rho = np.mean([density['rho'][i] for i in labels], axis=0)
    w = 4 * np.pi * r**2 * rho
    return w / w.max()


def channel_used(entry, label):
    neu, ned = entry['neu'], entry['ned']
    if label == 'uu':
        return neu > 1 or ned > 1
    if label == 'dd':
        return ned > 1
    if label == 'ud':
        return neu > 0 and ned > 0
    if label == 'u':
        return neu > 0
    if label == 'd':
        return ned > 0
    return True


def targets(entries, densities, term):
    """Radial profiles to fit: one target per (system, term, nuclear set) with its spin channels."""
    res = []
    for target in all_targets(entries, densities, term):
        if any(np.any(channel['y'] != 0) for channel in target['channels']):
            res.append(target)
    return res


def all_targets(entries, densities, term):
    res = []
    for entry in entries:
        key = f'{entry["basis"]}:{entry["system"]}'
        if key not in densities:
            continue
        density = densities[key]
        r = np.array(density['grid'])
        base = {'key': key, 'basis': entry['basis'], 'system': entry['system'], 'kind': entry['kind'], 'r': r}
        if term == 'u' and 'jastrow' in entry and 'u' in entry['jastrow']:
            jastrow = entry['jastrow']
            u = jastrow['u']
            channels = []
            for spin, label in enumerate(U_SPIN_LABELS[len(u['parameters']) - 1]):
                if channel_used(entry, label):
                    channels.append({'label': label, 'channel': U_CUSP[label], 'constraint': 'cusp', 'y': u_profile(u, jastrow['trunc'], spin, r)})
            if channels:
                res.append(
                    dict(
                        base,
                        species='e-e',
                        Z=0,
                        L=u['cutoff'],
                        n_casino=len(channels) * (len(u['parameters'][0]) - 1) + 1,
                        channels=channels,
                        w=pair_weight(density),
                    )
                )
        if term == 'chi' and 'jastrow' in entry:
            jastrow = entry['jastrow']
            for chi in jastrow['chi']:
                channels = []
                for spin, label in enumerate(CHI_SPIN_LABELS[len(chi['parameters']) - 1]):
                    if channel_used(entry, label):
                        channels.append(
                            {'label': label, 'channel': None, 'constraint': 'zero_slope', 'y': chi_profile(chi, jastrow['trunc'], spin, r)}
                        )
                pp = entry['is_pseudoatom'][chi['labels'][0]]
                res.append(
                    dict(
                        base,
                        species=species_label(entry, chi['labels']),
                        Z=species_z(entry, chi['labels']),
                        pp=pp,
                        labels=chi['labels'],
                        L=chi['cutoff'],
                        n_casino=len(channels) * (len(chi['parameters'][0]) - 1) + 1,
                        channels=channels,
                        w=en_weight(density, chi['labels']),
                    )
                )
        if term == 'eta' and 'backflow' in entry and 'eta' in entry['backflow']:
            backflow = entry['backflow']
            eta = backflow['eta']
            channels = []
            for spin, label in enumerate(['uu', 'ud']):
                if channel_used(entry, label):
                    antiparallel = label == 'ud'
                    constraint = 'free'
                    if not antiparallel:
                        constraint = 'zero_slope'
                    channels.append(
                        {'label': label, 'channel': antiparallel, 'constraint': constraint, 'y': eta_profile(eta, backflow['trunc'], spin, r)}
                    )
            if channels:
                n_coef = len(eta['parameters'][0])
                n_casino = sum(n_coef - int(c['constraint'] == 'zero_slope') for c in channels) + len(eta['cutoff'])
                res.append(dict(base, species='e-e', Z=0, L=max(eta['cutoff']), n_casino=n_casino, channels=channels, w=pair_weight(density)))
        if term == 'mu' and 'backflow' in entry:
            backflow = entry['backflow']
            for mu in backflow['mu']:
                ae = bool(mu['cusp'])
                constraint = 'zero_slope'
                if ae:
                    constraint = 'r2'
                channels = [{'label': 'ud', 'channel': ae, 'constraint': constraint, 'y': mu_profile(mu, backflow['trunc'], 0, r)}]
                n_coef = len(mu['parameters'][0])
                res.append(
                    dict(
                        base,
                        species=species_label(entry, mu['labels']),
                        Z=species_z(entry, mu['labels']),
                        pp=not ae,
                        labels=mu['labels'],
                        L=mu['cutoff'],
                        n_casino=n_coef - 1 - int(ae) + 1,
                        channels=channels,
                        w=en_weight(density, mu['labels']),
                    )
                )
    return res


def weighted_error(target, predictions):
    num = den = 0.0
    for channel, y_fit in zip(target['channels'], predictions):
        num += np.sum(target['w'] * (y_fit - channel['y']) ** 2)
        den += np.sum(target['w'] * channel['y'] ** 2)
    return np.sqrt(num / den)


def derivative_error(target, predictions):
    """Same metric for dy/dr: insensitive to the constant that u can trade with chi."""
    r = target['r']
    num = den = 0.0
    for channel, y_fit in zip(target['channels'], predictions):
        dy = np.gradient(channel['y'], r)
        num += np.sum(target['w'] * (np.gradient(y_fit, r) - dy) ** 2)
        den += np.sum(target['w'] * dy**2)
    return np.sqrt(num / den)


def fit_form(target, form):
    """Joint fit of all spin channels with a shared cutoff; returns (error, parameters, L, predictions)."""
    r = target['r']
    n_shape = len(form['params'])
    n_channels = len(target['channels'])
    sqrt_w = np.sqrt(target['w'])
    lower = []
    for name in form['params']:
        if name in LENGTH_PARAMS:
            lower.append(0.02)
        else:
            lower.append(-np.inf)
    lower = lower * n_channels + [0.5]
    upper = [np.inf] * (n_shape * n_channels) + [15.0]

    def predictions(x):
        L = x[-1]
        res = []
        for i, channel in enumerate(target['channels']):
            res.append(value(form, r, x[i * n_shape : (i + 1) * n_shape], channel['channel'], L))
        return res

    def residuals(x):
        res = []
        for channel, y_fit in zip(target['channels'], predictions(x)):
            res.append(sqrt_w * (y_fit - channel['y']))
        return np.concatenate(res)

    best = None
    for guess in form['guess']:
        for L0 in (target['L'], 1.3 * target['L']):
            x0 = np.array(list(guess) * n_channels + [min(L0, 14.9)])
            x0 = np.clip(x0, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
            with np.errstate(all='ignore'):
                sol = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=4000)
            if best is None or sol.cost < best.cost:
                best = sol
    parameters = [best.x[i * n_shape : (i + 1) * n_shape].tolist() for i in range(n_channels)]
    y_fit = predictions(best.x)
    return weighted_error(target, y_fit), parameters, float(best.x[-1]), y_fit


def polynomial_design(r, L, order, constraint, cusp):
    """Columns of the CASINO polynomial (1 - r/L)^C sum c_k r^k with the r = 0 constraint eliminated."""
    C = TRUNC
    cut = np.where(r < L, (1 - r / L) ** C, 0.0)
    phi = [r**k * cut for k in range(order + 1)]
    offset = np.zeros_like(r)
    if constraint == 'free':
        columns = phi
    elif constraint == 'r2':
        columns = phi[2:]
    else:
        # c1 = C c0 / L (+ cusp): zero or cusp slope at r = 0
        columns = [phi[0] + C / L * phi[1]] + phi[2:]
        if constraint == 'cusp':
            offset = cusp * phi[1]
    return np.array(columns).T, offset


def fit_polynomial(target, order):
    """CASINO-type polynomial of lower order refitted to the profile; the cutoff is optimized by a 1D scan."""
    r = target['r']
    sqrt_w = np.sqrt(target['w'])

    def solve(L):
        y_fit = []
        for channel in target['channels']:
            a, offset = polynomial_design(r, L, order, channel['constraint'], channel['channel'])
            coef, *_ = np.linalg.lstsq(sqrt_w[:, None] * a, sqrt_w * (channel['y'] - offset), rcond=None)
            y_fit.append(a @ coef + offset)
        return y_fit

    sol = minimize_scalar(lambda L: weighted_error(target, solve(L)), bounds=(0.5, 15.0), method='bounded')
    y_fit = solve(sol.x)
    n_params = 1
    for channel in target['channels']:
        a, _ = polynomial_design(r, sol.x, order, channel['constraint'], channel['channel'])
        n_params += a.shape[1]
    return weighted_error(target, y_fit), n_params, float(sol.x), y_fit


def n_form_params(form, target):
    return len(form['params']) * len(target['channels']) + 1


def match(targets_a, target):
    """Target of the same system, term and nuclear set in another list (e.g. varmin for an emin target)."""
    for other in targets_a:
        if other['key'] == target['key'] and other['species'] == target['species'] and other.get('labels') == target.get('labels'):
            return other
    return None


def noise_error(target, other):
    """Weighted difference between two optimizations of the same term, in the metric of weighted_error."""
    if other is None:
        return np.nan, np.nan
    predictions = []
    for channel in target['channels']:
        found = [c['y'] for c in other['channels'] if c['label'] == channel['label']]
        if not found:
            return np.nan, np.nan
        predictions.append(found[0])
    return weighted_error(target, predictions), derivative_error(target, predictions)
