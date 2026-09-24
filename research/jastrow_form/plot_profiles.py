#!/usr/bin/env python3
"""Radial profiles u(r_ij), chi(r_iI), eta(r_ij), mu(r_iI) for every system (last emin stage)."""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from terms import CHI_SPIN_LABELS, PLOTS, U_SPIN_LABELS, chi_profile, eta_profile, load, mu_profile, short_name, species_label, u_profile  # noqa: E402

BASES = ['gwfn', 'stowfn', 'pp']


def plot_u(entries, ax_uu, ax_ud):
    for entry in entries:
        jastrow = entry['jastrow']
        term = jastrow['u']
        r = np.linspace(0, term['cutoff'], 300)
        labels = U_SPIN_LABELS[len(term['parameters']) - 1]
        for spin, label in enumerate(labels):
            ax = ax_uu
            if label == 'ud':
                ax = ax_ud
            if label == 'dd':
                continue
            ax.plot(r, u_profile(term, jastrow['trunc'], spin, r), label=short_name(entry))
    ax_uu.set_title('u parallel')
    ax_ud.set_title('u antiparallel')


def plot_chi(entries, ax):
    for entry in entries:
        jastrow = entry['jastrow']
        for term in jastrow['chi']:
            r = np.linspace(0, term['cutoff'], 300)
            spin_labels = CHI_SPIN_LABELS[len(term['parameters']) - 1]
            for spin, label in enumerate(spin_labels):
                name = f'{short_name(entry)}:{species_label(entry, term["labels"])}:{label}'
                ax.plot(r, chi_profile(term, jastrow['trunc'], spin, r), label=name)
    ax.set_title('chi')


def plot_eta(entries, ax_uu, ax_ud):
    for entry in entries:
        backflow = entry['backflow']
        term = backflow['eta']
        L = max(term['cutoff'])
        r = np.linspace(0, L, 300)
        ax_uu.plot(r, eta_profile(term, backflow['trunc'], 0, r), label=short_name(entry))
        ax_ud.plot(r, eta_profile(term, backflow['trunc'], 1, r), label=short_name(entry))
    ax_uu.set_title('eta parallel')
    ax_ud.set_title('eta antiparallel')


def plot_mu(entries, ax):
    for entry in entries:
        backflow = entry['backflow']
        for term in backflow['mu']:
            r = np.linspace(0, term['cutoff'], 300)
            name = f'{short_name(entry)}:{species_label(entry, term["labels"])}'
            ax.plot(r, mu_profile(term, backflow['trunc'], 0, r), label=name)
    ax.set_title('mu')


def main():
    os.makedirs(PLOTS, exist_ok=True)
    for basis in BASES:
        jastrow_entries = [e for e in load(['Jastrow_emin']) if e['basis'] == basis]
        backflow_entries = [e for e in load(['Backflow_emin']) if e['basis'] == basis]
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plot_u(jastrow_entries, axes[0, 0], axes[0, 1])
        plot_chi(jastrow_entries, axes[0, 2])
        plot_eta(backflow_entries, axes[1, 0], axes[1, 1])
        plot_mu(backflow_entries, axes[1, 2])
        for ax in axes.flat:
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel('r (bohr)')
            ax.legend(fontsize=6)
        fig.suptitle(f'{basis}: u, chi from Jastrow_emin; eta, mu from Backflow_emin (last stage)')
        fig.tight_layout()
        fig.savefig(os.path.join(PLOTS, f'profiles_{basis}.png'), dpi=90)
        plt.close(fig)


if __name__ == '__main__':
    main()
