#!/usr/bin/env python3
"""Pooled profiles on the scaled axes used by symreg.py, with the closed forms found by the regression."""

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from symreg import pool  # noqa: E402
from terms import PLOTS  # noqa: E402

PANELS = [
    ('u', 'ud', 'u / |u(0)|  vs  r / b0,  b0 = -u(0)/gamma', 'exp(-x)', lambda x: -np.exp(-x)),
    ('u', 'uu', 'u / |u(0)|  vs  r / b0 (holes only)', 'exp(-x)', lambda x: -np.exp(-x)),
    ('chi', 'ud', 'chi / chi(0)  vs  r / r_half', 'exp(-ln2 x^2)', lambda x: np.exp(-np.log(2) * x**2)),
    ('eta', 'ud', 'eta / |eta(0)|  vs  r / r_half', 'exp(-ln2 x)', lambda x: np.exp(-np.log(2) * x)),
]


def main():
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (term, channel, title, name, closed) in zip(axes, PANELS):
        x, y, w, names = pool(term, channel)
        order = np.argsort(x)
        ax.scatter(x, y, s=3 + 12 * w, alpha=0.3, label=f'{len(names)} profiles (size ~ weight)')
        ax.plot(x[order], closed(x[order]), 'k', lw=2, label=name)
        ax.set_xlim(0, 4)
        ax.set_title(f'{term} {channel}: {title}', fontsize=9)
        ax.axhline(0, color='k', lw=0.5)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'universal_profiles.png'), dpi=80)
    plt.close(fig)


if __name__ == '__main__':
    main()
