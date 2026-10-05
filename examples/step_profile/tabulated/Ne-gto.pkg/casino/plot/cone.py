#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


class ConicalIntersection:
    # Range for the grid
    R = 5
    # Point density
    N = 20

    def __init__(self, a, b, c, d, k):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.k = k
        self.fig = plt.figure()
        self.ax = plt.axes(projection='3d')

    def cone(self):
        """Parametrization of a cone in:

        cylindrical coordinates r = k|z|,
        spherical coordinates tan(θ) = k,
        or Cartesian coordinates x² + y² = k²z²
        """
        u = np.linspace(0, 2 * np.pi, 2 * self.N)
        z = np.linspace(-self.R, self.R, 2 *self.N - 1)
        uu, zz = np.meshgrid(u, z)
        xx = zz * np.cos(uu)
        yy = zz * np.sin(uu)
        self.ax.plot_surface(xx, yy, zz / self.k, color='blue', alpha=0.3)

    def plane(self):
        """Parametrization of a plane: ax + by + cz + d = 0"""
        t = np.linspace(-self.R, self.R, self.N)
        if self.a:
            # Create a grid in y and z, compute x
            yy, zz = np.meshgrid(t, t)
            xx = -(self.b * yy + self.c * zz + self.d) / self.a
        elif self.b:
            # Create a grid in x and z, compute y
            xx, zz = np.meshgrid(t, t)
            yy = -(self.a * xx + self.c * zz + self.d) / self.b
        elif self.c:
            # Create a grid in x and y, compute z
            xx, yy = np.meshgrid(t, t)
            zz = -(self.a * xx + self.b * yy + self.d) / self.c
        else:
            raise ValueError("All coefficients a, b, c are zero — this is not a plane.")
        self.ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')

    def find_phi(self):
        """Returns a list of intervals [phi1, phi2] in [0, 2π) where the inequality holds.
        
        -R < -d * k / (a * cos(phi) + b * sin(phi) + c * k) < R
        """
        res = []
        for R in (-self.R, self.R):
            numerator = self.k * (np.abs(self.d) - self.c * R)
            denominator = self.R * np.hypot(self.a, self.b)
            if np.abs(numerator/denominator) <= 1:
                alpha = -np.arctan2(self.a, self.b)
                beta = np.arcsin(numerator/denominator)
                if R > 0:
                    res.append([alpha + beta, alpha - beta + np.pi])
                else:
                    res.append([alpha + beta - np.pi, alpha - beta])
        return res or [[0, 2 * np.pi]]

    def intersection(self):
        """Conic section in spherical coordinates."""
        if self.d:
            for res in self.find_phi():
                phi = np.linspace(*res, 2 * self.N - 1)
                z = -self.d * self.k / (self.a * np.cos(phi) + self.b * np.sin(phi) + self.c * self.k)
                x = z * self.k * np.cos(phi)
                y = z * self.k * np.sin(phi)
                self.ax.plot(x, y, z, 'r-', linewidth=2)
        elif np.abs(self.c) <= self.k * np.hypot(self.a, self.b):
            phi = np.arctan2(self.b, self.a)
            cos_theta = -self.c / (self.k * np.hypot(self.a, self.b))
            for sgn in (1, -1):
                theta = phi + sgn * np.arccos(cos_theta)
                z = np.linspace(-self.R, self.R, self.N)
                x = self.k * np.cos(theta) * z
                y = self.k * np.sin(theta) * z
                self.ax.plot(x, y, z, 'r-', linewidth=2)
        else:
            self.ax.scatter(0, 0, 0, color='red', s=2)

    def plot(self):
        self.cone()
        self.plane()
        self.intersection()
        # Can set your view from different angles.
        self.ax.view_init(azim=15, elev=15)
        self.ax.set_xlim(-self.R, self.R)
        self.ax.set_ylim(-self.R, self.R)
        self.ax.set_zlim(-self.R, self.R)
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')
        # maximize window in Qt5
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()


if __name__ == '__main__':
    for i in range(10):
        intersection = ConicalIntersection(a=i/3, b=0, c=1, d=-2, k=1)
        intersection.plot()

    ellipse = ConicalIntersection(a=0, b=0, c=1, d=2, k=1)
    ellipse.plot()
    parabola = ConicalIntersection(a=1, b=0, c=1, d=2, k=1)
    parabola.plot()
    hyperbola = ConicalIntersection(a=1, b=0, c=2, d=0, k=1)
    hyperbola.plot()
