#!/usr/bin/env python3
"""Write examples/gwfn/<system>/HF/cc-pVQZ/CBCS/Jastrow_emin_uncut: u and chi without cutoff and an empty f-term.

    u, chi  the optimized u and chi (Functional form 2) of the last stage of
            examples/stowfn/<system>/HF/QZ4P/CBCS/Jastrow_emin_uncut (correlation.out.4)
    f       the empty polynomial F TERM of examples/gwfn/<system>/HF/cc-pVQZ/CBCS/Jastrow_emin/correlation.data

The e-n cusp comes from the cusp correction of the Gaussian orbitals, chi has zero slope at the nucleus.
The input is that of the gwfn Jastrow_emin example without vm_filter, gwfn.data is linked as there.
"""

import os

from collect import ROOT

SYSTEMS = ['He', 'Be', 'N', 'Ne', 'Ar', 'Kr', 'O3']
STOWFN = 'examples/stowfn/{}/HF/QZ4P/CBCS/Jastrow_emin_uncut'
GWFN = 'examples/gwfn/{}/HF/cc-pVQZ/CBCS'


def block(text, start, end):
    return text[text.index(start) : text.index(end) + len(end) + 1]


def build(system):
    with open(os.path.join(ROOT, STOWFN.format(system), 'correlation.out.4')) as f:
        uncut = f.read()
    with open(os.path.join(ROOT, GWFN.format(system), 'Jastrow_emin', 'correlation.data')) as f:
        polynomial = f.read()
    jastrow = uncut[uncut.index(' START JASTROW') : uncut.index(' START F TERM')]
    title = f'  {system}, exponential u-term and Gaussian chi-term without cutoff, empty f-term'
    jastrow = jastrow.replace('  no title given', title, 1)
    header = polynomial[: polynomial.index(' START JASTROW')]
    return header + jastrow + block(polynomial, ' START F TERM', ' END F TERM') + ' END JASTROW\n'


def main():
    for system in SYSTEMS:
        source = os.path.join(ROOT, GWFN.format(system), 'Jastrow_emin')
        target = os.path.join(ROOT, GWFN.format(system), 'Jastrow_emin_uncut')
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, 'correlation.data'), 'w') as f:
            f.write(build(system))
        with open(os.path.join(source, 'input')) as f:
            lines = [line for line in f if not line.startswith('vm_filter')]
        with open(os.path.join(target, 'input'), 'w') as f:
            f.writelines(lines)
        link = os.path.join(target, 'gwfn.data')
        if not os.path.lexists(link):
            os.symlink(os.readlink(os.path.join(source, 'gwfn.data')), link)
        print(os.path.relpath(target, ROOT))


if __name__ == '__main__':
    main()
