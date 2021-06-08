#!/usr/bin/env python3

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from readers.casino import Casino


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
            if np.allclose(shells[nshell].position, atoms[atom].position) and shells[nshell].moment == 0:
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
    return (mo @ orbital.T)[:, 0], (mo @ orbital_derivative.T)[:, 0], (mo @ orbital_second_derivative.T)[:, 0]


def initial_phi_data(mo, shells, atoms):
    """Calculate initial phi coefficients.
    shift variable chosen so that (phi−C) is of one sign within rc.
    eta = gauss0_full - gauss0_s contribution from Gaussians on other nuclei
    if abs(gauss0_s_n(orbital, ion_s, spin_type_full)) < 10**-7:
        print('Orbital s component effectively zero at this nucleus.')
    """
    alpha = np.zeros((atoms.shape[0], 5))
    for atom in range(atoms.shape[0]):
        rc = 1/atom[atom].charge
        phi_0, phi_1, _ = wfn_s(0.0, atom, mo, shells, atoms)
        gauss0, gauss1, gauss2 = wfn_s(rc, atom, mo, shells, atoms)
        eta = 0  # contribution from Gaussians on other nuclei
        for i in range(mo.shape[0]):
            shift = 0 if np.sign(phi_0[i]) == np.sign(gauss0[i]) else 1.1 * gauss0[i]
            zeff = atom[atom].charge * (1 + eta/phi_0[i])
            print(f"atom {atom}, s-orbital at r=0 {phi_0[i]}, at r=rc {gauss0[i]}, C={shift}, psi-sign {np.sign(phi_0[i])}")
            X1 = np.log(np.abs(gauss0[i] - shift))                     # (9)
            X2 = gauss1[i] / (gauss0[i] - shift)                       # (10)
            X3 = gauss2[i] / (gauss0[i] - shift)                       # (11)
            X4 = -zeff * phi_0[i] / (phi_0[i] - shift)                 # (12)
            X5 = np.log(np.abs(phi_0[i] - shift))                      # (13)
            print(f"X1={X1} X2={X2} X3={X3} X4={X4} X5={X5}")
            # (14)
            alpha[atom, 0] = X5
            alpha[atom, 1] = X4
            alpha[atom, 2] = 6*X1/rc**2 - 3*X2/rc + X3/2 - 3*X4/rc - 6*X5/rc**2 - X2**2/2
            alpha[atom, 3] = -8*X1/rc**3 + 5*X2/rc**2 - X3/rc + 3*X4/rc**2 + 8*X5/rc**3 + X2**2/rc
            alpha[atom, 4] = 3*X1/rc**4 - 2*X2/rc**3 + X3/2/rc**2 - X4/rc**3 - 3*X5/rc**4 - X2**2/2/rc**2

    return alpha


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

    To ensure orbitals expanded in Gaussians have correct cusp at the nucleus,
    we replace the s-part of the orbital within r=rcusp by disign*exp[p(r)],
    where disign is the sign of the orbital at r=0 and p(r) is the polynomial
    p = acusp(1) + acusp(2)*r + acusp(3)*r^2 + acusp(4)*r^3 + acusp(5)*r^4
    such that p(rcusp), p'(rcusp), and p''(rcusp) are continuous, and
    p'(0) = -zeff.  Local energy at r=0 is set by a smoothness criterion.

    This setup routine calculates appropriate values of rcusp and acusp for
    each orbital, ion and spin.

    Work out appropriate cusp radius rcusp by looking at fluctuations in the local
    energy within rcmax. Set rcusp to the largest radius at which the deviation
    from the "ideal" curve has a magnitude greater than zatom^2/cusp_control.
    """


if __name__ == '__main__':
    """
    """

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Slater/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Slater/'

    casino = Casino(path)

    neu, ned = casino.input.neu, casino.input.ned
    mo_up, mo_down = casino.mdet.mo_up[0], casino.mdet.mo_down[0]

    # since neu => neb, only up-orbitals are needed to calculate wfn.
    cusp_graph(0, mo_up, gwfn.shells, gwfn.atoms)
    initial_phi_data(mo_up, gwfn.nshell, gwfn.atomic_positions)
