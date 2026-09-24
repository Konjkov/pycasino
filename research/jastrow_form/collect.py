#!/usr/bin/env python3
"""Collect optimized Jastrow and backflow parameters from examples/**/correlation.out.*

For every directory the last optimization stage (highest N in correlation.out.N) is read
with the PyCasino readers and the fixed (cusp/constraint-completed) parameter arrays are
dumped to data/parameters.json together with the geometry of the system.
"""

import glob
import json
import os
import re
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from casino.readers import CasinoConfig  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

SOURCES = [
    ('gwfn', 'examples/gwfn/{system}/HF/cc-pVQZ/CBCS/{kind}', ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'B2H6', 'O3']),
    ('gwfn-casscf', 'examples/gwfn/{system}/MP2-CASSCF(2.4)/cc-pVQZ/CBCS/{kind}', ['Be']),
    ('stowfn', 'examples/stowfn/{system}/HF/QZ4P/CBCS/{kind}', ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3']),
    ('pp', 'examples/ppotential_HF/{system}/HF/aug-cc-pVQZ-CDF/CBCS/{kind}', ['H', 'B', 'C', 'N', 'O', 'F', 'Ne', 'B2H6']),
]
KINDS = ['Jastrow_emin', 'Jastrow_varmin', 'Backflow_emin', 'Backflow_varmin', 'Jastrow_emin/linear', 'Backflow_emin/linear']


def last_stage(path):
    stages = []
    for file_name in glob.glob(os.path.join(path, 'correlation.out.*')):
        match = re.search(r'correlation\.out\.(\d+)$', file_name)
        if match:
            stages.append(int(match.group(1)))
    if not stages:
        return None
    return max(stages)


def read_config(path, file_name):
    """Read the directory with file_name in place of correlation.data.
    PyCasino runs in a subdirectory (linear/) take input and orbitals from the parent directory.
    """
    files_path = path
    if not os.path.isfile(os.path.join(path, 'input')):
        files_path = os.path.dirname(path)
    with tempfile.TemporaryDirectory() as tmp:
        for name in os.listdir(files_path):
            src = os.path.join(files_path, name)
            if os.path.isfile(src) and not name.startswith('correlation'):
                os.symlink(src, os.path.join(tmp, name))
        os.symlink(os.path.join(path, file_name), os.path.join(tmp, 'correlation.data'))
        config = CasinoConfig(tmp)
        config.read()
    return config


def jastrow_dict(jastrow):
    res = {'trunc': int(jastrow.trunc)}
    if jastrow.u_parameters.size:
        res['u'] = {
            'cutoff': float(jastrow.u_cutoff[0]['value']),
            'parameters': jastrow.u_parameters.tolist(),
        }
    res['chi'] = []
    for labels, parameters, cutoff, cusp in zip(jastrow.chi_labels, jastrow.chi_parameters, jastrow.chi_cutoff, jastrow.chi_cusp):
        res['chi'].append({'labels': labels.tolist(), 'cutoff': float(cutoff['value']), 'cusp': bool(cusp), 'parameters': parameters.tolist()})
    res['f'] = []
    for labels, parameters, cutoff, no_dup_u, no_dup_chi in zip(
        jastrow.f_labels, jastrow.f_parameters, jastrow.f_cutoff, jastrow.no_dup_u_term, jastrow.no_dup_chi_term
    ):
        res['f'].append(
            {
                'labels': labels.tolist(),
                'cutoff': float(cutoff['value']),
                'no_dup_u': bool(no_dup_u),
                'no_dup_chi': bool(no_dup_chi),
                'parameters': parameters.tolist(),
            }
        )
    return res


def backflow_dict(backflow):
    res = {'trunc': int(backflow.trunc), 'ae_cutoff': backflow.ae_cutoff.tolist()}
    if backflow.eta_parameters.size:
        res['eta'] = {'cutoff': backflow.eta_cutoff['value'].tolist(), 'parameters': backflow.eta_parameters.tolist()}
    res['mu'] = []
    for labels, parameters, cutoff, cusp in zip(backflow.mu_labels, backflow.mu_parameters, backflow.mu_cutoff, backflow.mu_cusp):
        res['mu'].append({'labels': labels.tolist(), 'cutoff': float(cutoff['value']), 'cusp': bool(cusp), 'parameters': parameters.tolist()})
    res['phi'] = []
    for labels, phi, theta, cutoff, cusp, irrot in zip(
        backflow.phi_labels, backflow.phi_parameters, backflow.theta_parameters, backflow.phi_cutoff, backflow.phi_cusp, backflow.phi_irrotational
    ):
        res['phi'].append(
            {
                'labels': labels.tolist(),
                'cutoff': float(cutoff['value']),
                'cusp': bool(cusp),
                'irrotational': bool(irrot),
                'phi': phi.tolist(),
                'theta': theta.tolist(),
            }
        )
    return res


def collect_one(basis, system, kind, path):
    stage = last_stage(path)
    if stage is None:
        return None
    config = read_config(path, f'correlation.out.{stage}')
    initial_path = path
    if not os.path.isfile(os.path.join(path, 'correlation.data')):
        initial_path = os.path.dirname(path)
    initial = read_config(initial_path, 'correlation.data')
    wfn = config.wfn
    entry = {
        'basis': basis,
        'system': system,
        'kind': kind,
        'path': os.path.relpath(path, ROOT),
        'file': f'correlation.out.{stage}',
        'neu': int(config.input.neu),
        'ned': int(config.input.ned),
        'atom_numbers': wfn.atom_numbers.tolist(),
        'atom_charges': wfn.atom_charges.tolist(),
        'atom_positions': wfn.atom_positions.tolist(),
        'is_pseudoatom': wfn.is_pseudoatom.tolist(),
        'initial_cutoffs': {},
    }
    if config.jastrow is not None:
        entry['jastrow'] = jastrow_dict(config.jastrow)
        entry['initial_cutoffs']['u'] = float(initial.jastrow.u_cutoff[0]['value'])
        entry['initial_cutoffs']['chi'] = initial.jastrow.chi_cutoff['value'].tolist()
        entry['initial_cutoffs']['f'] = initial.jastrow.f_cutoff['value'].tolist()
    if config.backflow is not None:
        entry['backflow'] = backflow_dict(config.backflow)
        entry['initial_cutoffs']['eta'] = initial.backflow.eta_cutoff['value'].tolist()
        entry['initial_cutoffs']['mu'] = initial.backflow.mu_cutoff['value'].tolist()
        entry['initial_cutoffs']['phi'] = initial.backflow.phi_cutoff['value'].tolist()
    return entry


def main():
    entries = []
    for basis, template, systems in SOURCES:
        for system in systems:
            for kind in KINDS:
                path = os.path.join(ROOT, template.format(system=system, kind=kind))
                if not os.path.isdir(path):
                    continue
                try:
                    entry = collect_one(basis, system, kind, path)
                except ValueError as error:
                    print(f'skipped {os.path.relpath(path, ROOT)}: {error}')
                    continue
                if entry is not None:
                    print(f'{entry["path"]}/{entry["file"]}')
                    entries.append(entry)
    # Be atom (STO) expansion-order scans
    for path in sorted(glob.glob(os.path.join(ROOT, 'examples/jastrow/3_*/[0-9][0-9]'))):
        entry = collect_one('stowfn-scan', 'Be', 'Jastrow_scan_' + '/'.join(path.split('/')[-2:]), path)
        if entry is not None:
            print(f'{entry["path"]}/{entry["file"]}')
            entries.append(entry)
    for path in sorted(glob.glob(os.path.join(ROOT, 'examples/backflow/[0-9]_[0-9]_[0-9]/[0-9][0-9]'))):
        entry = collect_one('stowfn-scan', 'Be', 'Backflow_scan_' + '/'.join(path.split('/')[-2:]), path)
        if entry is not None:
            print(f'{entry["path"]}/{entry["file"]}')
            entries.append(entry)
    os.makedirs(os.path.join(HERE, 'data'), exist_ok=True)
    with open(os.path.join(HERE, 'data', 'parameters.json'), 'w') as f:
        json.dump(entries, f, indent=1)
    print(len(entries), 'entries')


if __name__ == '__main__':
    main()
