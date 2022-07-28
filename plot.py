#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import polyval, polyval3d
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from readers.backflow import Backflow
from readers.jastrow import Jastrow


class JastrowPlot(Jastrow):

    def u_term(self, r_ee, spin_dep):
        """Jastrow u-term
        """
        C, L = self.trunc, self.u_cutoff
        parameters = np.copy(self.u_parameters)
        if r_ee < self.u_cutoff:
            u_set = spin_dep % parameters.shape[1]
            poly = 0.0
            for k in range(parameters.shape[0]):
                if k == 1:
                    p = self.u_cusp_const[spin_dep]
                else:
                    p = parameters[k, u_set]
                poly += p * r_ee ** k
            return poly * (r_ee - L) ** C
        return 0.0

    def chi_term(self, r_eI, spin_dep, atom_set):
        """Jastrow chi-term
        """
        C, L, parameters, chi_labels = self.trunc, self.chi_cutoff[atom_set], self.chi_parameters[atom_set], self.chi_labels[atom_set]
        if r_eI < L:
            chi_set = spin_dep % parameters.shape[1]
            return polyval(r_eI, parameters[:, chi_set]) * (r_eI - L) ** C
        return 0.0

    def f_term(self, r_e1I, r_e2I, r_ee, spin_dep, atom_set):
        """Jastrow f-term
        :return:
        """
        C, L, parameters,  f_labels = self.trunc, self.f_cutoff[atom_set], self.f_parameters[atom_set], self.f_labels[atom_set]
        if r_e1I < L and r_e2I < L:
            f_set = spin_dep % parameters.shape[3]
            return polyval3d(r_e1I, r_e2I, r_ee, parameters[:, :, :, f_set]) * (r_e1I - L) ** C * (r_e2I - L) ** C
            # poly = 0.0
            # for l in range(parameters.shape[0]):
            #     for m in range(parameters.shape[1]):
            #         for n in range(parameters.shape[2]):
            #             poly += parameters[l, m, n, f_set] * r_e1I ** l * r_e2I ** m * r_ee ** n
            # return poly * (r_e1I - L) ** C * (r_e2I - L) ** C
        return 0.0

    def plot(self, term):
        """Plot terms"""
        steps = 100

        if term == 'u':
            x_min, x_max = 0, self.u_cutoff
            x_grid = np.linspace(x_min, x_max, steps)
            for spin_dep in range(self.u_parameters.shape[1]):
                y_grid = np.zeros((steps,))
                for i in range(steps):
                    y_grid[i] = self.u_term(x_grid[i], spin_dep)
                    if spin_dep == 1:
                        y_grid[i] /= 2.0
                plt.plot(x_grid, y_grid, label=['uu', 'ud/2', 'dd'][spin_dep])
            plt.xlabel('r_ee (au)')
            plt.ylabel('polynomial part')
            plt.title('JASTROW u-term')
        elif term == 'chi':
            for atom_set in range(self.chi_cutoff.shape[0]):
                x_min, x_max = 0, self.chi_cutoff[atom_set]
                x_grid = np.linspace(x_min, x_max, steps)
                for spin_dep in range(self.chi_parameters[atom_set].shape[1]):
                    y_grid = np.zeros((steps,))
                    for i in range(100):
                        y_grid[i] = self.chi_term(x_grid[i], spin_dep, atom_set)
                    plt.plot(x_grid, y_grid, label=f"atom_set {atom_set} {['u', 'd'][spin_dep]}")
            plt.xlabel('r_eN (au)')
            plt.ylabel('polynomial part')
            plt.title('JASTROW chi-term')
        elif term == 'f':
            figure = plt.figure()
            axis = figure.add_subplot(111, projection='3d')
            for atom_set in range(self.f_cutoff.shape[0]):
                x_min, x_max = 0.0, self.f_cutoff[atom_set]
                y_min, y_max = 0.0, self.f_cutoff[atom_set]
                x = np.linspace(x_min, x_max, steps)
                y = np.linspace(y_min, y_max, steps)
                x_grid, y_grid = np.meshgrid(x, y)
                for spin_dep in range(self.f_parameters[atom_set].shape[3]):
                    z_grid = np.zeros((steps, steps))
                    c = np.cos(175 * np.pi / 180)
                    for i in range(100):
                        for j in range(100):
                            c = np.cos(y_grid[i, j] * np.pi / y_max)
                            # r_e1I = x_grid[i, j]
                            # r_e2I = y_grid[i, j]
                            # r_ee = np.sqrt(r_e1I**2 + r_e2I**2 - 2 * r_e1I * r_e2I * c)
                            r_e1I = x_grid[i, j]
                            r_e2I = x_grid[i, j]
                            r_ee = np.sqrt(r_e1I**2 + r_e2I**2 - 2 * r_e1I * r_e2I * c)
                            z_grid[i, j] = self.f_term(r_e1I, r_e2I, r_ee, spin_dep, atom_set)
                    axis.plot_wireframe(x_grid, y_grid, z_grid, label=f"atom_set {atom_set} {['uu', 'ud', 'dd'][spin_dep]}", color=['green', 'cyan', 'pink'][spin_dep])
            axis.set_xlabel('r_e1N (au)')
            axis.set_ylabel('r_e2N (au)')
            axis.set_zlabel('polynomial part')
            plt.title('JASTROW f-term')
        elif term == 'all':
            figure = plt.figure()
            axis = figure.add_subplot(111, projection='3d')
            for atom_set in range(self.f_cutoff.shape[0]):
                x_min, x_max = 0.0, self.f_cutoff[atom_set]
                y_min, y_max = 0.0, self.f_cutoff[atom_set]
                x = np.linspace(x_min, x_max, steps)
                y = np.linspace(y_min, y_max, steps)
                x_grid, y_grid = np.meshgrid(x, y)
                for spin_dep in range(self.f_parameters[atom_set].shape[3]):
                    z_grid = np.zeros((steps, steps))
                    for i in range(100):
                        for j in range(100):
                            c = np.cos(y_grid[i, j] * np.pi / y_max)
                            # r_e1I = x_grid[i, j]
                            # r_e2I = y_grid[i, j]
                            # r_ee = np.sqrt(r_e1I**2 + r_e2I**2 - 2 * r_e1I * r_e2I * c)
                            r_e1I = x_grid[i, j]
                            r_e2I = x_grid[i, j]
                            r_ee = np.sqrt(r_e1I**2 + r_e2I**2 - 2 * r_e1I * r_e2I * c)
                            z_grid[i, j] = self.u_term(r_ee, spin_dep) + self.chi_term(r_e1I, spin_dep, atom_set) + self.f_term(r_e1I, r_e2I, r_ee, spin_dep, atom_set)
                    axis.plot_wireframe(x_grid, y_grid, z_grid, label=f"atom_set {atom_set} {['uu', 'ud', 'dd'][spin_dep]}", color=['green', 'cyan', 'pink'][spin_dep])
            axis.set_xlabel('r_e1N (au)')
            axis.set_ylabel('r_e2N (au)')
            axis.set_zlabel('polynomial part')
            plt.title('JASTROW f-term')

        plt.grid(True)
        plt.legend()
        plt.show()


class BackflowPlot(Backflow):

    def eta_term(self, r_ee, spin_dep):
        """Backflow eta-term
        :return:
        """
        parameters = self.eta_parameters
        eta_set = spin_dep % parameters.shape[1]
        C, L = self.trunc, self.eta_cutoff[eta_set % self.eta_cutoff.shape[0]]
        if r_ee < L:
            return polyval(r_ee, parameters[:, eta_set]) * (1 - r_ee / L) ** C * r_ee
        return 0.0

    def mu_term(self, r_eI, spin_dep, atom_set):
        """Backflow eta-term
        :return:
        """
        C, L, parameters, mu_labels = self.trunc, self.mu_cutoff[atom_set], self.mu_parameters[atom_set], self.mu_labels[atom_set]
        if r_eI < L:
            mu_set = spin_dep % parameters.shape[1]
            return polyval(r_eI, parameters[:, mu_set]) * (1 - r_eI / L) ** C * r_eI
        return 0.0

    def phi_term(self, r_e1I, r_e2I, r_ee, spin_dep, atom_set):
        """Backflow phi-term
        :return:
        """
        C, L, phi_parameters, phi_labels = self.trunc, self.phi_cutoff[atom_set], self.phi_parameters[atom_set],  self.phi_labels[atom_set]
        if r_e1I < L and r_e2I < L:
            phi_set = spin_dep % phi_parameters.shape[3]
            phi_poly = 0.0
            for k in range(phi_parameters.shape[0]):
                for l in range(phi_parameters.shape[1]):
                    for m in range(phi_parameters.shape[2]):
                        phi_poly += phi_parameters[k, l, m, phi_set] * r_e1I ** k * r_e2I ** l * r_ee ** m
            return (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C * phi_poly * r_ee
        return 0.0

    def theta_term(self, r_e1I, r_e2I, r_ee, spin_dep, atom_set):
        """Backflow theta-term
        :return:
        """
        C, L, theta_parameters, phi_labels = self.trunc, self.phi_cutoff[atom_set], self.theta_parameters[atom_set], self.phi_labels[atom_set]
        if r_e1I < L and r_e2I < L:
            phi_set = spin_dep % theta_parameters.shape[3]
            theta_poly = 0.0
            for k in range(theta_parameters.shape[0]):
                for l in range(theta_parameters.shape[1]):
                    for m in range(theta_parameters.shape[2]):
                        theta_poly += theta_parameters[k, l, m, phi_set] * r_e1I ** k * r_e2I ** l * r_ee ** m
            return (1 - r_e1I / L) ** C * (1 - r_e2I / L) ** C * theta_poly * r_e1I
        return 0.0

    def plot(self, term):
        """Plot terms"""
        steps = 100

        if term == 'eta':
            x_min, x_max = 0, np.max(self.eta_cutoff)
            x_grid = np.linspace(x_min, x_max, steps)
            for spin_dep in range(self.eta_parameters.shape[1]):
                y_grid = np.zeros((steps, ))
                for i in range(100):
                    y_grid[i] = self.eta_term(x_grid[i], spin_dep)
                plt.plot(x_grid, y_grid, label=['uu', 'ud', 'dd'][spin_dep])
            plt.xlabel('r_ee (au)')
            plt.ylabel('polynomial part')
            plt.title('Backflow eta-term')
        elif term == 'mu':
            for atom_set in range(self.mu_cutoff.shape[0]):
                x_min, x_max = 0, self.mu_cutoff[atom_set]
                x_grid = np.linspace(x_min, x_max, steps)
                for spin_dep in range(self.mu_parameters[atom_set].shape[1]):
                    y_grid = np.zeros((steps, ))
                    for i in range(100):
                        y_grid[i] = self.mu_term(x_grid[i], spin_dep, atom_set)
                    plt.plot(x_grid, y_grid, label=f"atom_set {atom_set} {['u', 'd'][spin_dep]}")
            plt.xlabel('r_eN (au)')
            plt.ylabel('polynomial part')
            plt.title('Backflow mu-term')
        elif term == 'phi':
            for atom_set in range(self.phi_cutoff.shape[0]):
                pass
        elif term == 'theta':
            for atom_set in range(self.phi_cutoff.shape[0]):
                pass

        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    """Plot Jastrow terms
    """
    path = ''

    # path = 'test/stowfn/He/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Be/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/N/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Ne/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Ar/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/Kr/HF/QZ4P/CBCS/Jastrow/'
    # path = 'test/stowfn/O3/HF/QZ4P/CBCS/Jastrow/'

    term = 'u'
    jastrow_plot = JastrowPlot()
    jastrow_plot.read(os.path.join(path, 'correlation.data'))
    jastrow_plot.plot(term)
    term = 'chi'
    jastrow_plot.read(os.path.join(path, 'correlation.data'))
    jastrow_plot.plot(term)
    # term = 'f'
    # JastrowPlot(os.path.join(path, 'correlation.data')).plot(term)
    # term = 'all'
    # JastrowPlot(os.path.join(path, 'correlation.data')).plot(term)

    # path = 'test/stowfn/He/HF/QZ4P/Backflow/'
    # path = 'test/stowfn/Be/HF/QZ4P/Backflow/'
    # path = 'test/stowfn/Ne/HF/QZ4P/Backflow/'
    # path = 'test/stowfn/Ar/HF/QZ4P/Backflow/'
    # path = 'test/stowfn/Kr/HF/QZ4P/Backflow/'
    # path = 'test/stowfn/O3/HF/QZ4P/Backflow/'

    term = 'eta'
    backflow_plot = BackflowPlot(np.zeros(shape=(1,)))
    backflow_plot.read(os.path.join(path, 'correlation.data'))
    backflow_plot.plot(term)
    term = 'mu'
    backflow_plot.read(os.path.join(path, 'correlation.data'))
    backflow_plot.plot(term)
