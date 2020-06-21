#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from readers.gwfn import Gwfn
from readers.input import Input


def wfn_s(r, atom, shells, contraction_coefficients, exponents, atomic_positions):
    """wfn of single electron of any s-orbital an each atom"""
    orbital = []
    p = 0
    for nshell in range((shells.shape[0])):
        s_part = 0.0
        if np.allclose(shells[nshell][1], atomic_positions[atom]) and shells[nshell][0] == 0:
            for primitive in range(p, p + shells[nshell][2]):
                s_part += contraction_coefficients[primitive] * np.exp(-exponents[primitive] * r * r)
            orbital.append(s_part)
        p += shells[nshell][2]
    return orbital


def fit(fit_function, xdata, ydata, p0):
    """Fit gaussian basis by slater"""

    plot = True

    try:
        popt, pcov = curve_fit(fit_function, xdata, ydata, p0, maxfev=1000000)
        perr = np.sqrt(np.diag(pcov))
        if plot:
            plt.plot(xdata, ydata, 'b-', label='data')
            plt.plot(xdata, fit_function(xdata, *popt), 'r-', label='fit')
    except Exception as e:
        print(e)
        popt, perr = [], []
        if plot:
            plt.plot(xdata, ydata, 'b-', label='data')
    if plot:
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()
    return popt, perr


def multiple_fits(shells, contraction_coefficients, exponents, atomic_positions):
    """Fit all orbitals in GTO basis
    use Laguerre polynomials
    """

    def slater_1(r, a1, zeta1):
        """dln(phi)/dr|r=r_nucl = -zeta1"""
        return a1*np.exp(-zeta1*r)

    def slater_2(r, a1, zeta1, a2, zeta2):
        """dln(phi)/dr|r=r_nucl = -(a1 * zeta1 + a2 * zeta2)/(a1 + a2)"""
        return a1*np.exp(-zeta1*r) + a2*np.exp(-zeta2*r)

    def slater_3(r, a1, zeta1, a2, zeta2, a3, zeta3):
        """Best fit for n=0 orbital from ano-pVDZ set for He-Ne"""
        return a1*np.exp(-zeta1*r) + a2*np.exp(-zeta2*r) + a3*np.exp(-zeta3*r)

    def slater_4(r, a1, zeta1, a2, zeta2, a3, zeta3, a4, zeta4):
        """Best fit for n=0 orbital from ano-pVDZ set for He-Ne"""
        return a1*np.exp(-zeta1*r) + a2*np.exp(-zeta2*r) + a3*np.exp(-zeta3*r) + a4*np.exp(-zeta4*r)

    fit_function = slater_2
    p0 = (1, 1, 1, 1)

    for atom in range(atomic_positions.shape[0]):
        xdata = np.linspace(0, 3.0, 50)
        ydatas = wfn_s(xdata, atom, shells, contraction_coefficients, exponents, atomic_positions)
        for ydata in ydatas:
            popt, perr = fit(fit_function, xdata, ydata, p0)
            print(popt, perr)


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

    multiple_fits(gwfn.shells, gwfn.contraction_coefficients, gwfn.exponents, gwfn.atomic_positions)
