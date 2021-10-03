#!/usr/bin/env python3

import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def wfn_s(r, coefficients, exponents):
    """wfn of single electron of any s-orbital"""
    s_part = 0.0
    for primitive in range(coefficients.shape[0]):
        s_part += coefficients[primitive] * np.exp(-exponents[primitive] * r * r)
    return s_part


def fit(fit_function, xdata, ydata, p0, plot):
    """Fit gaussian basis by slater"""

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


def multiple_fits(coefficients, exponents, Z):
    """Fit all orbitals in GTO basis with slater orbitals
    """

    def slater_2(r, a1, zeta1, zeta2):
        """dln(phi)/dr|r=r_nucl = -Z"""
        a2 = -a1
        return (
            a1 * np.exp(-zeta1*r)/(zeta1-Z) +
            a2 * np.exp(-zeta2*r)/(zeta2-Z)
        )

    def slater_3(r, a1, a2, zeta1, zeta2, zeta3):
        """dln(phi)/dr|r=r_nucl = -Z"""
        a3 = - (a1 + a2)
        return (
            a1 * np.exp(-zeta1*r)/(zeta1-Z) +
            a2 * np.exp(-zeta2*r)/(zeta2-Z) +
            a3 * np.exp(-zeta3*r)/(zeta3-Z)
        )

    def slater_4(r, a1, a2, a3, zeta1, zeta2, zeta3, zeta4):
        """dln(phi)/dr|r=r_nucl = -Z"""
        a4 = - (a1 + a2 + a3)
        return (
            a1 * np.exp(-zeta1*r)/(zeta1-Z) +
            a2 * np.exp(-zeta2*r)/(zeta2-Z) +
            a3 * np.exp(-zeta3*r)/(zeta3-Z) +
            a4 * np.exp(-zeta4*r)/(zeta4-Z)
        )

    def slater_5(r, *args):
        """dln(phi)/dr|r=r_nucl = -Z"""
        res = 0
        for coefficient, zeta in zip(coefficients, args):
            res += coefficient * np.exp(-zeta*r)/(zeta-Z)
        return Z * res

    def slater_6(r, *args):
        """dln(phi)/dr|r=r_nucl = -Z"""
        res = 0
        for coefficient, zeta in zip(coefficients, args):
            res += coefficient * np.exp(-zeta*r)/(zeta-Z)
        return Z * res

    fit_function = slater_2
    initial_guess = (1, Z-1, Z+1)

    # fit_function = slater_3
    # initial_guess = (1, 1, Z-1, Z+0.1, Z+1)

    # fit_function = slater_4
    # initial_guess = (1, 1, -1, Z-1, Z-1, Z+1, Z+1)

    # fit_function = slater_5
    # initial_guess = [1] * len(coefficients)

    xdata = np.linspace(0.01, 3.0, 50)
    ydata = wfn_s(xdata, coefficients, exponents)
    popt, perr = fit(fit_function, xdata, ydata, initial_guess, True)
    print(popt, perr)
    new_primitives = len(popt) // 2 + 1
    new_exponents = popt[new_primitives-1:]
    new_coefficients = np.append(popt[:new_primitives-1], -np.sum(popt[:new_primitives-1])) / (new_exponents - Z)
    print('dln(phi)/dr|r=r_nucl =', np.sum(new_coefficients * new_exponents/np.sum(new_coefficients)))
    return new_primitives, new_coefficients, new_exponents
