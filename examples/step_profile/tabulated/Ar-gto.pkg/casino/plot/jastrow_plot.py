#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyval, polyval2d, polyval3d
import sys
from itertools import combinations
from math import exp
from yaml import safe_load
from mpl_toolkits.mplot3d import Axes3D

from jastrow import Jastrow


class Plot(Jastrow):
    """Plot along the line."""

    def __init__(self, term_num, file):
        super().__init__(term_num, file)

    @property
    def grid(self):
        x = np.linspace(self.x_min, self.x_max, self.x_steps)
        y = np.linspace(self.y_min, self.y_max, self.y_steps)
        return np.meshgrid(x, y)

    def plot1D_a(self, replot=False):
        self.x_min = 0.0
        self.x_max = 5.0
        self.y_min = 0.0
        self.y_max = 0.0
        self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.x_steps = 200
        self.y_steps = 1
        for channel in self.channels:
            plt.plot(self.grid[0][0], self.jastrow(self.term, channel, self.grid, self.xy_elec)[0], label=channel)
        plt.xlabel('r (au)')
        plt.ylabel('value')
        plt.title('JASTROW term')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot1D_b(self, replot=False):
        self.x_min = 0.0
        self.x_max = 5.0
        self.y_min = 0.0
        self.y_max = 0.0
        self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.x_steps = 200
        self.y_steps = 1
        for channel in self.channels:
            plt.plot(self.grid[0][0], self.jastrow(self.term, channel, self.grid, self.xy_nucl)[0], label=channel)
        plt.xlabel('r (au)')
        plt.ylabel('value')
        plt.title('JASTROW term')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot2D(self, replot=False):
        self.x_min = -5.0
        self.x_max = 5.0
        self.y_min = -5.0
        self.y_max = 5.0
        self.xy_nucl_1 = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
        self.xy_nucl_2 = np.array([2.0, 0.0])[:, np.newaxis, np.newaxis]
        self.x_steps = self.y_steps = 40
        fig = plt.figure()
        self.ax = fig.add_subplot(111, projection='3d')
        for channel in self.channels:
            self.ax.plot_wireframe(*self.grid, self.jastrow(self.term, channel, self.grid, self.xy_nucl_1, self.xy_nucl_2))
        fig.canvas.mpl_connect('key_press_event', self.onpress)
        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        plt.show()

    def plot3D(self, replot=False):

        if not replot:
            self.x_min = -3.0
            self.x_max = 3.0
            self.y_min = -3.0
            self.y_max = 3.0
            self.xy_elec = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
            self.xy_nucl = np.array([0.0, 0.0])[:, np.newaxis, np.newaxis]
            self.x_steps = self.y_steps = 30
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')
            # self.ax = self.fig.add_subplot(111)
        self.ax.clear()
        for i, channel in enumerate(self.channels):
            self.ax.plot_wireframe(*self.grid, self.jastrow(self.term, channel, self.grid, self.xy_elec, self.xy_nucl), color=['blue', 'green', 'red'][i])
            #
            # contours = self.ax.contour(*self.xy_grid(), self.jastrow3D(np.array(self.xy_grid()), self.xy_elec, self.xy_nucl, channel), 10, colors='black')
            # plt.clabel(contours, inline=True, fontsize=8)
            # indexing = 'ij' for origin = 'lower' or indexing = 'xy' for origin = 'upper'
            # img = self.ax.imshow(self.jastrow3D(np.array(self.xy_grid()), self.xy_elec, self.xy_nucl, channel), extent=[self.x_min, self.x_max, self.y_min, self.y_max], origin='lower', cmap='summer')
            # plt.colorbar(img)

        self.ax.set_xlabel('X axis')
        self.ax.set_ylabel('Y axis')
        self.ax.set_zlabel('Z axis')
        if replot:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.fig.canvas.mpl_connect('key_press_event', self.onpress)
            plt.show()

    def plot(self, replot=False):
        """Plot single JASTROW term"""
        if self.rank(self.term) == [2, 0]:
            self.plot1D_a(replot)
        elif self.rank(self.term) == [1, 1]:
            self.plot1D_b(replot)
        elif self.rank(self.term) == [1, 2]:
            self.plot2D(replot)
        elif self.rank(self.term) == [2, 1]:
            self.plot3D(replot)
        else:
            print('term with {} rank is not supported'.format(self.rank(self.term)))

    def onclick(self, event):
        """On click"""
        if not (event.xdata and event.ydata):
            return
        if not (self.x_min < event.xdata < self.x_max or self.y_min < event.xdata < self.y_max):
            return
        self.xy_elec = np.array([event.xdata, event.ydata])[:, np.newaxis, np.newaxis]
        self.plot(replot=True)

    def onpress(self, event):
        """On key pressed"""
        delta = 0.2
        if event.key == 'right':
            self.xy_elec[0] += delta
        elif event.key == 'left':
            self.xy_elec[0] -= delta
        elif event.key == 'up':
            self.xy_elec[1] += delta
        elif event.key == 'down':
            self.xy_elec[1] -= delta
        self.plot(replot=True)


def main():
    parser = argparse.ArgumentParser(
        description="This script plot JASTOW terms.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'term_num',
        type=int,
        help="JASTROW term number"
    )
    parser.add_argument(
        'casl_file',
        type=str,
        default='parameters.casl',
        nargs='?',
        help="name of *.casl file"
    )

    args = parser.parse_args()

    Plot(args.term_num, args.casl_file).plot()


if __name__ == "__main__":
    main()
