#!/usr/bin/env python3
"""Number of optimizable parameters: CASINO polynomial (counted in the correlation.out files) vs proposed forms.

Proposed, per term ('minimal' / 'accurate' = within the CASINO/PyCasino reproducibility for most systems):
    u    exp hole (1) per used spin channel, exp2 (3) for a parallel channel with u(0) > 0 / exp2 (3); + 1 cutoff
    chi  window A w(r/L) (1) per used spin channel / yukawa2 (4); + 1 cutoff per set
    f    rank-1 product g(r1) g(r2) h(r12) (10) / symmetric cubic in (r1+r2, (r1-r2)^2, r12) (20) per spin channel; + 1 cutoff per set
    eta  exp (2) / exp_osc (3) per used spin channel; + 1 cutoff
    mu   shell (2) / ring (3); + 1 cutoff per set
    Phi  no reduced form found: CASINO count kept
"""

import os
import re

import numpy as np

from radial import channel_used
from terms import HERE, ROOT_DIR, U_SPIN_LABELS, load, u_profile

PATTERNS = {
    'u': r'! alpha_',
    'chi': r'! beta_',
    'f': r'! gamma_',
    'eta': r'! c_',
    'mu': r'! mu_',
    'phi': r'! (phi|theta)_',
}


def casino_count(entry, term):
    with open(os.path.join(ROOT_DIR, entry['path'], entry['file'])) as f:
        text = f.read()
    count = len(re.findall(PATTERNS[term], text))
    if term in ('u',):
        return count + 1
    if term in ('chi', 'f'):
        return count + len(entry['jastrow'][term])
    if term == 'eta':
        return count + len(entry['backflow']['eta']['cutoff'])
    return count + len(entry['backflow'][term])


def proposed_count(entry, term, accurate):
    if term == 'u':
        u = entry['jastrow']['u']
        n = 1
        for spin, label in enumerate(U_SPIN_LABELS[len(u['parameters']) - 1]):
            if not channel_used(entry, label):
                continue
            if accurate or u_profile(u, entry['jastrow']['trunc'], spin, np.zeros(1))[0] > 0:
                n += 3
            else:
                n += 1
        return n
    if term == 'chi':
        n = 0
        for chi in entry['jastrow']['chi']:
            per_channel = 1
            if accurate:
                per_channel = 4
            n += 1 + per_channel * len(chi['parameters'])
        return n
    if term == 'f':
        per_channel = 10
        if accurate:
            per_channel = 20
        return sum(1 + per_channel * len(f_set['parameters']) for f_set in entry['jastrow']['f'])
    if term == 'eta':
        used = [label for label in ('uu', 'ud') if channel_used(entry, label)]
        return 1 + (2 + int(accurate)) * len(used)
    if term == 'mu':
        return (3 + int(accurate)) * len(entry['backflow']['mu'])
    return casino_count(entry, term)


def main():
    lines = ['| system | term | CASINO | minimal | accurate |', '|---|---|---|---|---|']
    totals = {}
    for kind, terms in (('Jastrow_emin', ('u', 'chi', 'f')), ('Backflow_emin', ('eta', 'mu', 'phi'))):
        for entry in load([kind]):
            if entry['basis'] not in ('gwfn', 'stowfn', 'pp'):
                continue
            for term in terms:
                if term == 'u' and 'u' not in entry['jastrow']:
                    continue
                casino = casino_count(entry, term)
                minimal, accurate = proposed_count(entry, term, False), proposed_count(entry, term, True)
                lines.append(f'| {entry["basis"]}:{entry["system"]} | {term} | {casino} | {minimal} | {accurate} |')
                total = totals.setdefault(term, [0, 0, 0])
                total[0] += casino
                total[1] += minimal
                total[2] += accurate
    summary = ['| term | CASINO (sum over all systems) | minimal | accurate |', '|---|---|---|---|']
    for term, (casino, minimal, accurate) in totals.items():
        summary.append(f'| {term} | {casino} | {minimal} ({minimal / casino:.2f}) | {accurate} ({accurate / casino:.2f}) |')
    with open(os.path.join(HERE, 'results', 'param_count.md'), 'w') as f:
        f.write('\n'.join(summary) + '\n\n' + '\n'.join(lines) + '\n')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
