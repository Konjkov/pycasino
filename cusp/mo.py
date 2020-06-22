#!/usr/bin/env python3

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from readers.gwfn import Gwfn
from readers.input import Input


def wfn_s(r, atom, mo, shells, atoms):
    """wfn of single electron of s-orbitals an each atom"""
    orbital = np.zeros(mo.shape)
    orbital_derivative = np.zeros(mo.shape)
    orbital_second_derivative = np.zeros(mo.shape)
    for i in range(mo.shape[0]):
        ao = 0
        for nshell in range(shells.shape[0]):
            # angular momentum
            l = shells[nshell].moment
            s_part = 0.0
            s_derivative_part = 0.0
            s_second_derivative_part = 0.0
            if np.allclose(shells[nshell].position, atoms[atom].position) and shells[nshell].momoent == 0:
                for primitive in range(shells[nshell].primitives):
                    alpha = shells[nshell].exponents[primitive]
                    exponent = shells[nshell].coefficients[primitive] * np.exp(-alpha * r * r)
                    s_part += exponent
                    s_derivative_part -= 2 * alpha * r * exponent
                    s_second_derivative_part += 2 * alpha * (2 * alpha * r * r - 3) * exponent
            orbital[i, ao] = s_part
            orbital_derivative[i, ao] = s_derivative_part
            orbital_second_derivative[i, ao] = s_second_derivative_part
            ao += 2 * l + 1
    return np.dot(mo, orbital.T)[:, 0], np.dot(mo, orbital_derivative.T)[:, 0], np.dot(mo, orbital_second_derivative.T)[:, 0]


def initial_phi_data(mo, shells, atoms):
    """Calculate initial phi coefficients."""
    for atom in range(atoms.shape[0]):
        rc = 1/atom[atom].charge
        wfn_0, wfn_derivative_0, _ = wfn_s(0.0, atom, mo, shells, atoms)
        wfn_rc, wfn_derivative_rc, wfn_second_derivative_rc = wfn_s(rc, atom, mo, shells, atoms)
        for i in range(mo.shape[0]):
            C = 0 if np.sign(wfn_0[i]) == np.sign(wfn_rc[i]) else 1.1 * wfn_rc[i]
            print(f"atom {atom}, s-orbital at r=0 {wfn_0[i]}, at r=rc {wfn_rc[i]}, C={C}, psi-sign {np.sign(wfn_0[i])}")
            X1 = np.log(np.abs(wfn_rc[i] - C))
            X2 = wfn_derivative_rc[i] / (wfn_rc[i] - C)
            X3 = wfn_second_derivative_rc[i] / (wfn_rc[i] - C)
            X4 = wfn_derivative_0[i] / (wfn_0[i] - C)
            X5 = np.log(np.abs(wfn_0[i] - C))
            print(f"X1={X1} X2={X2} X3={X3} X4={X4} X5={X5}")
            alpha0 = X5
            alpha1 = X4
            alpha2 = 6*X1/rc**2 - 3*X2/rc + X3/2 - 3*X4/rc - 6*X5/rc**2 - X2**2/2
            alpha3 = -8*X1/rc**3 + 5*X2/rc**2 - X3/rc + 3*X4/rc**2 + 8*X5/rc**3 + X2**2/rc
            alpha4 = 3*X1/rc**4 - 2*X2/rc**3 + X3/2/rc**2 - X4/rc**3 - 3*X5/rc**4 - X2**2/2/rc**2


#@nb.jit(nopython=True, cache=True)
def cusp_graph(atom, mo, shells, atoms):
    """In nuclear position dln(phi)/dr|r=r_nucl = -Z_nucl
    """
    x = np.linspace(0, 1/atom[atom].charge**2, 1000)
    args = (atom, mo, shells, atoms)
    wfn = [wfn_s(r, *args)[1]/wfn_s(r, *args)[0] for r in x]
    plt.plot(x, wfn, x, -atoms[atom].charge*np.ones(1000))
    plt.show()


def ma_cusp():
    """
    Scheme for adding electron–nucleus cusps to Gaussian orbitals
    A. Ma, M. D. Towler, N. D. Drummond, and R. J. Needs

    An orbital, psi, expanded in a Gaussian basis set can be written as:
    psi = phi + eta
    where phi is the part of the orbital arising from the s-type Gaussian functions
    centered on the nucleus in question (which, for convenience is at r = 0)

    In our scheme we seek a corrected orbital, tilde_psi, which differs from psi
    only in the part arising from the s-type Gaussian functions centered on the nucleus,
    i.e., so that tilde_psi obeys the cusp condition at r=0

    tilde_psi = tilde_phi + eta

    We apply a cusp correction to each orbital at each nucleus at which it is nonzero.
    Inside some cusp correction radius rc we replace phi, the part of the orbital arising
    from s-type Gaussian functions centered on the nucleus in question, by:

    tilde_phi = C + sign[tilde_phi(0)] * exp(p(r))

    In this expression sign[tilde_phi(0)], reflecting the sign of tilde_phĩ at the nucleus,
    and C is a shift chosen so that tilde_phi − C is of one sign within rc.
    """


if __name__ == '__main__':
    """
    """

    # gwfn = Gwfn('test/h/HF/cc-pVQZ/gwfn.data')
    # inp = Input('test/h/HF/cc-pVQZ/input')
    gwfn = Gwfn('test/be/HF/cc-pVQZ/gwfn.data')
    inp = Input('test/be/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/be2/HF/cc-pVQZ/gwfn.data')
    # inp = Input('test/be2/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/acetic/HF/cc-pVQZ/gwfn.data')
    # inp = Input('test/acetic/HF/cc-pVQZ/input')
    # gwfn = Gwfn('test/acetaldehyde/HF/cc-pVQZ/gwfn.data')
    # inp = Input('test/acetaldehyde/HF/cc-pVQZ/input')

    mo = gwfn.mo
    neu = inp.neu
    ned = inp.ned

    mo_u = mo[0][:neu]
    mo_d = mo[0][:ned]
    # since neu => neb, only up-orbitals are needed to calculate wfn.
    cusp_graph(0, mo_u, gwfn.shells, gwfn.atoms)
    #initial_phi_data(mo_u, gwfn.nshell, gwfn.shell_types, gwfn.shell_positions, gwfn.primitives, gwfn.contraction_coefficients, gwfn.exponents, gwfn.atomic_positions, gwfn.atom_charges)

