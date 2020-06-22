#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


def wfn_s(r, shell):
    """wfn of single electron of any s-orbital"""
    s_part = 0.0
    for primitive in range(shell['primitives']):
        s_part += shell['coefficients'][primitive] * np.exp(-shell['exponents'][primitive] * r * r)
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


def multiple_fits(shell):
    """Fit all orbitals in GTO basis with slater orbitals
    """

    def slater(r, a1, zeta1):
        """dln(phi)/dr|r=r_nucl = -zeta1"""
        return a1*np.exp(-zeta1*r)

    fit_function = slater
    initial_guess = (1, 1)

    xdata = np.linspace(0.1, 3.0, 50)
    ydata = wfn_s(xdata, shell)
    return fit(fit_function, xdata, ydata, initial_guess, False)
