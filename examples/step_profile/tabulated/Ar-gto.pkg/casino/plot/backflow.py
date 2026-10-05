#!/usr/bin/env python3

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D

from casino.readers.main import CasinoConfig
from casino.backflow import Backflow


if __name__ == '__main__':
    """Plot Backflow terms
    """

    term = 'all'
    config_path = '../../tests/stowfn/He/HF/QZ4P/CBCS/Backflow/'

    config = CasinoConfig(config_path)
    config.read()
    backflow = Backflow(
        config.input.neu, config.input.ned,
        config.backflow.trunc, config.backflow.eta_parameters, config.backflow.eta_parameters_optimizable,
        config.backflow.eta_cutoff,
        config.backflow.mu_parameters, config.backflow.mu_parameters_optimizable, config.backflow.mu_cutoff,
        config.backflow.mu_cusp, config.backflow.mu_labels,
        config.backflow.phi_parameters, config.backflow.phi_parameters_optimizable,
        config.backflow.theta_parameters, config.backflow.theta_parameters_optimizable,
        config.backflow.phi_cutoff, config.backflow.phi_cusp, config.backflow.phi_labels, config.backflow.phi_irrotational,
        config.backflow.ae_cutoff, config.backflow.ae_cutoff_optimizable
    )

    if term == 'all':
        plot_type = 2
        xy_nucl = (0.0, 0.0)
        xy_elec = (1.0, 0.0)
        for atom in range(config.wfn.atom_positions.shape[0]):
            max_l = max((backflow.eta_cutoff, backflow.mu_cutoff[atom], backflow.phi_cutoff[atom]))
            x_max = y_max = max_l
            x_min = y_min = -max_l
            x_steps = 25
            y_steps = 25
            x_grid = np.linspace(x_min, x_max, x_steps)
            y_grid = np.linspace(y_min, y_max, y_steps)
            ij_certesian = np.meshgrid(x_grid, y_grid, indexing='ij')
            xy_certesian = np.meshgrid(x_grid, y_grid, indexing='xy')
            phi_ij_certesian = np.zeros((2, x_steps, y_steps))
            phi_xy_certesian = np.zeros((2, x_steps, y_steps))
            for i in range(x_steps):
                for j in range(y_steps):
                    r_e = np.array([[x_grid[i], y_grid[j], 0.0], [1.0, 0.0, 0.0]]) + config.wfn.atom_positions[atom]
                    sl = slice(atom, atom + 1)
                    # FIXME: e_vectors, n_vectors = self.wfn._relative_coordinates(pos)
                    e_vectors = np.expand_dims(r_e, 1) - np.expand_dims(r_e, 0)
                    e_powers = backflow.ee_powers(e_vectors)
                    # n_vectors = -subtract_outer(config.wfn.atom_positions[sl], r_e)
                    n_vectors = np.expand_dims(r_e, 0) - np.expand_dims(config.wfn.atom_positions[sl], 1)
                    n_powers = backflow.en_powers(n_vectors)
                    phi_ij_certesian[:, i, j] = backflow.value(e_vectors, n_vectors)[0, 0:2]
                    phi_xy_certesian[:, j, i] = backflow.value(e_vectors, n_vectors)[0, 0:2]
            fig_2D, axs = plt.subplots(1, 2)
            for spin_dep in range(2):
                axs[spin_dep].clear()
                backflow.neu = 2 - spin_dep
                backflow.ned = spin_dep
                axs[spin_dep].set_title('{} backflow {} term'.format('all', ['u-u', 'u-d'][spin_dep]))
                axs[spin_dep].set_aspect('equal', adjustable='box')
                axs[spin_dep].plot(*xy_nucl, 'ro', label='nucleus')
                axs[spin_dep].plot(*xy_elec, 'mo', label='electron')
                axs[spin_dep].set_xlabel('X axis')
                axs[spin_dep].set_ylabel('Y axis')
                if plot_type == 0:
                    axs[spin_dep].quiver(
                        *ij_certesian,
                        *phi_ij_certesian,
                        angles='xy', scale_units='xy',
                        scale=1, color=['blue', 'green'][spin_dep]
                    )
                elif plot_type == 1:
                    axs[spin_dep].plot(
                        *(ij_certesian + phi_ij_certesian),
                        color=['blue', 'green'][spin_dep]
                    )
                    axs[spin_dep].plot(
                        *(xy_certesian + phi_xy_certesian),
                        color=['blue', 'green'][spin_dep]
                    )
                elif plot_type == 2:
                    x_steps = 10
                    y_steps = 25
                    r = np.linspace(0, x_max, x_steps)[:, np.newaxis]
                    theta = np.linspace(0, 2 * np.pi, y_steps)
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    ij_radial = np.array([x, y])
                    phi_ij_radial = np.zeros((2, x_steps, y_steps))
                    axs[spin_dep].plot(
                        *(ij_radial + phi_ij_radial),
                        color=['blue', 'green'][spin_dep]
                    )

                    x_steps = 25
                    y_steps = 10
                    theta = np.linspace(0, 2 * np.pi, x_steps)[:, np.newaxis]
                    r = np.linspace(0, x_max, y_steps)
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    xy_radial = np.array([x, y])
                    phi_xy_radial = np.zeros((2, x_steps, y_steps))
                    axs[spin_dep].plot(
                        *(xy_radial + phi_xy_radial),
                        color=['blue', 'green'][spin_dep]
                    )
                elif plot_type == 3:
                    pass
                    # contours = axs[spin_dep].contour(
                    #     grid_3D('ij')[0][:, :, 1],
                    #     grid_3D('ij')[1][:, :, 1],
                    #     jacobian_det('ij', ri_spin, rj_spin, self.set)[:, :, 1],
                    #     10,
                    #     colors='black'
                    # )
                    # plt.clabel(contours, inline=True, fontsize=8)

                if backflow.eta_cutoff is not None:
                    axs[spin_dep].add_patch(Circle(xy_elec, backflow.eta_cutoff, fill=False, linestyle=':', label='ETA e-e cutoff'))
                if backflow.mu_cutoff[atom] is not None:
                    axs[spin_dep].add_patch(Circle(xy_nucl, backflow.mu_cutoff[atom], fill=False, color='c', label='MU e-n cutoff'))
                if backflow.phi_cutoff[atom] is not None:
                    axs[spin_dep].add_patch(Circle(xy_nucl, backflow.phi_cutoff[atom], fill=False, color='y', label='PHI e-n cutoff'))
                if backflow.ae_cutoff[atom] is not None:
                    axs[spin_dep].add_patch(Circle(xy_nucl, backflow.ae_cutoff[atom], fill=False, label='AE cutoff'))
                axs[spin_dep].legend()
    plt.grid(True)
    plt.legend()
    plt.show()
