#!/usr/bin/env python3
"""Molecules B2H6 and O3: atom <-> molecule comparison, anisotropy, cutoffs vs bond lengths.

1. chi and mu of a species in the molecule vs the isolated atom (profiles and the weighted
   difference on the molecular weight, full range and valence region r > 1 bohr).
2. Anisotropy: along the bond lines the three-body terms are compared with the two-body ones
   at a typical pair separation b:
       f:   rms over directions n of  sum_I f_I(|P - R_I|, |P + b n - R_I|, b)   vs  |u(b)|
       Phi: rms over n of |sum_I Phi_I r_12 + Theta_I r_1I|                     vs  pair-weighted rms |eta(r) r|
   A ratio << 1 means the e-n part can be kept radial with two-body e-e terms.
3. Cutoff lengths vs nearest internuclear distances.
"""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from density import directions  # noqa: E402
from radial import load_densities, targets  # noqa: E402
from terms import HERE, PLOTS, ELEMENTS, chi_profile, eta_profile, f_value, load, mu_profile, phi_value, species_label, u_profile  # noqa: E402

MOLECULES = ('B2H6', 'O3')
ATOM_OF = {'O': [('pp', 'O'), ('gwfn', 'N'), ('gwfn', 'Ne')], 'B': [('pp', 'B')], 'H': [('pp', 'H')]}
BOND_SEPARATION = 1.0
DIRECTIONS = directions(50)


def find(entries, basis, system):
    for entry in entries:
        if entry['basis'] == basis and entry['system'] == system:
            return entry
    return None


def compare_profiles():
    densities = load_densities()
    lines = ['| term | molecule | species (atoms) | atom | eps (all r) | eps (r > 1 bohr) |', '|---|---|---|---|---|---|']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row_index, (term, kind, profile) in enumerate((('chi', 'Jastrow_emin', chi_profile), ('mu', 'Backflow_emin', mu_profile))):
        entries = [e for e in load([kind]) if e['basis'] in ('gwfn', 'stowfn', 'pp')]
        mol_targets = [t for t in targets(entries, densities, term) if t['system'] in MOLECULES]
        atom_targets = [t for t in targets(entries, densities, term) if t['system'] not in MOLECULES]
        for target in mol_targets:
            element = target['species']
            if element not in ATOM_OF:
                continue
            ax = axes[row_index, int(element == 'B')]
            r = target['r']
            y = target['channels'][0]['y']
            ax.plot(r, y, lw=2, label=f'{target["key"]} {element}{target["labels"]}')
            for basis, system in ATOM_OF[element]:
                atoms = [t for t in atom_targets if t['basis'] == basis and t['system'] == system]
                if not atoms:
                    continue
                y_atom = atoms[0]['channels'][0]['y']
                w = target['w']
                valence = w * (r > 1.0)
                eps = np.sqrt(np.sum(w * (y_atom - y) ** 2) / np.sum(w * y**2))
                eps_valence = np.sqrt(np.sum(valence * (y_atom - y) ** 2) / np.sum(valence * y**2))
                lines.append(f'| {term} | {target["key"]} | {element} {target["labels"]} | {basis}:{system} | {eps:.3f} | {eps_valence:.3f} |')
            ax.set_title(f'{term}: nuclei in molecules vs atoms')
        for element, ax in (('O', axes[row_index, 0]), ('B', axes[row_index, 1])):
            for basis, system in ATOM_OF[element]:
                atoms = [t for t in atom_targets if t['basis'] == basis and t['system'] == system]
                if atoms:
                    ax.plot(atoms[0]['r'], atoms[0]['channels'][0]['y'], '--', label=f'atom {basis}:{system}')
            ax.set_xlim(0, 7)
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('r (bohr)')
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'molecule_vs_atom.png'), dpi=80)
    plt.close(fig)
    return lines


def set_of(sets, atom):
    for term_set in sets:
        if atom in term_set['labels']:
            return term_set
    return None


def typical_eta(backflow_entry, density):
    """Pair-weighted rms of the antiparallel eta displacement |eta(r) r|."""
    r = np.array(density['grid'])
    w = np.array(density['pair'])
    eta = backflow_entry['backflow']['eta']
    displacement = eta_profile(eta, backflow_entry['backflow']['trunc'], 1, r) * r
    return np.sqrt(np.sum(w * displacement**2) / np.sum(w))


def anisotropy(entry, backflow_entry, a, b, density):
    """Three-body / two-body ratios along the line from nucleus a to nucleus b."""
    positions = np.array(entry['atom_positions'])
    jastrow = entry['jastrow']
    t = np.linspace(0, 1, 41)
    u = jastrow['u']
    u_b = abs(u_profile(u, jastrow['trunc'], 1, np.array(BOND_SEPARATION)))
    f_ratio, phi_ratio = [], []
    for s in t:
        p1 = positions[a] + s * (positions[b] - positions[a])
        p2 = p1 + BOND_SEPARATION * DIRECTIONS
        f_sum = np.zeros(len(DIRECTIONS))
        phi_sum = np.zeros((len(DIRECTIONS), 3))
        for atom in range(len(positions)):
            r1 = np.linalg.norm(p1 - positions[atom])
            r2 = np.linalg.norm(p2 - positions[atom], axis=1)
            f_set = set_of(jastrow['f'], atom)
            if f_set is not None:
                f_sum += f_value(f_set, jastrow['trunc'], 1, r1, r2, BOND_SEPARATION)
            if backflow_entry is not None:
                phi_set = set_of(backflow_entry['backflow']['phi'], atom)
                if phi_set is not None:
                    trunc = backflow_entry['backflow']['trunc']
                    phi = phi_value(phi_set, trunc, 1, r1, r2, BOND_SEPARATION, 'phi')
                    theta = phi_value(phi_set, trunc, 1, r1, r2, BOND_SEPARATION, 'theta')
                    phi_sum += phi[:, None] * (p1 - p2) + theta[:, None] * (p1 - positions[atom])[None, :]
        f_ratio.append(np.sqrt(np.mean(f_sum**2)) / u_b)
        if backflow_entry is not None:
            phi_ratio.append(np.sqrt(np.mean(np.sum(phi_sum**2, axis=1))) / typical_eta(backflow_entry, density))
    return t, np.array(f_ratio), np.array(phi_ratio)


def anisotropy_all():
    densities = load_densities()
    jastrow_entries = load(['Jastrow_emin'])
    backflow_entries = load(['Backflow_emin'])
    lines = ['| molecule | bond | max f/u (nucleus) | f/u (midpoint) | max Phi/eta (nucleus) | Phi/eta (midpoint) |', '|---|---|---|---|---|---|']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for basis in ('gwfn', 'stowfn', 'pp'):
        for molecule, bonds in (('O3', [(0, 1)]), ('B2H6', [(0, 1), (0, 2), (0, 4)])):
            entry = find(jastrow_entries, basis, molecule)
            if entry is None:
                continue
            backflow_entry = find(backflow_entries, basis, molecule)
            for a, b in bonds:
                t, f_ratio, phi_ratio = anisotropy(entry, backflow_entry, a, b, densities[f'{basis}:{molecule}'])
                name = f'{ELEMENTS[entry["atom_numbers"][a]]}{a}-{ELEMENTS[entry["atom_numbers"][b]]}{b}'
                axes[0].plot(t, f_ratio, label=f'{basis}:{molecule} {name}')
                axes[1].plot(t, phi_ratio, label=f'{basis}:{molecule} {name}')
                ends = max(f_ratio[0], f_ratio[-1])
                phi_ends = max(phi_ratio[0], phi_ratio[-1])
                lines.append(
                    f'| {basis}:{molecule} | {name} | {ends:.3f} | {f_ratio[len(t) // 2]:.3f} | {phi_ends:.3f} | {phi_ratio[len(t) // 2]:.3f} |'
                )
    axes[0].set_title(f'f / |u| at r12 = {BOND_SEPARATION} bohr along bonds')
    axes[1].set_title(f'|Phi, Theta| at r12 = {BOND_SEPARATION} bohr / typical |eta r|')
    for ax in axes:
        ax.set_xlabel('fraction of the bond')
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'molecule_anisotropy.png'), dpi=80)
    plt.close(fig)
    return lines


def cutoffs():
    lines = [
        '| molecule | kind | species | L_chi | L_f | L_mu | L_phi | nearest distances (bohr) | nuclei within L_chi |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    for entry in load(['Jastrow_emin', 'Backflow_emin']):
        if entry['system'] not in MOLECULES or entry['basis'] not in ('gwfn', 'stowfn', 'pp'):
            continue
        positions = np.array(entry['atom_positions'])
        distance = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
        jastrow = entry['jastrow']
        for index, chi in enumerate(jastrow['chi']):
            atom = chi['labels'][0]
            nearest = np.sort(distance[atom])[1:4]
            within = int(np.sum(distance[atom] < chi['cutoff']) - 1)
            L_mu = L_phi = '-'
            if 'backflow' in entry:
                L_mu = f'{entry["backflow"]["mu"][index]["cutoff"]:.2f}'
                L_phi = f'{entry["backflow"]["phi"][index]["cutoff"]:.2f}'
            lines.append(
                f'| {entry["basis"]}:{entry["system"]} | {entry["kind"]} | {species_label(entry, chi["labels"])} | {chi["cutoff"]:.2f} | '
                f'{jastrow["f"][index]["cutoff"]:.2f} | {L_mu} | {L_phi} | {", ".join(f"{d:.2f}" for d in nearest)} | {within} |'
            )
    return lines


def main():
    lines = ['# Molecules (generated by molecules.py)', '', '## Molecule vs atom profiles', '']
    lines += compare_profiles()
    lines += ['', '## Anisotropy: three-body vs two-body along bonds', '']
    lines += anisotropy_all()
    lines += ['', '## Cutoffs vs internuclear distances', '']
    lines += cutoffs()
    with open(os.path.join(HERE, 'results', 'molecules.md'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
