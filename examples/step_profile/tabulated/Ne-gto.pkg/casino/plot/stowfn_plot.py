#!/usr/bin/env python3

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d
except:
    print('sudo apt-get install python3-matplotlib')
from numpy.linalg import norm, det
from plot_implicit import plot_implicit
import numpy as np


# http://www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital.html
def slater_1s(r):
    Z = 3.7
    return np.sqrt(Z**3/np.pi) * np.exp(-Z*norm(r, axis=0))


def slater_2s(r):
    Z = 3.00
    return np.sqrt(Z**5/(3*np.pi)) * norm(r, axis=0) * np.exp(-Z*norm(r, axis=0))


def slater_2px(r):
    Z = 1.85
    return np.sqrt(Z**5/np.pi) * r[0] * np.exp(-Z*norm(r, axis=0))


def slater_2py(r):
    Z = 1.85
    return np.sqrt(Z**5/np.pi) * r[1] * np.exp(-Z*norm(r, axis=0))


def slater_2pz(r):
    Z = 1.85
    return np.sqrt(Z**5/np.pi) * r[2] * np.exp(-Z*norm(r, axis=0))


def wfn_stack(r, wfns):
    return np.stack([wfn(r) for wfn in wfns], axis=2)


def Be_1s2s(r1, r2):
    return slater_1s(r1) * slater_2s(r2) - slater_1s(r2) * slater_2s(r1)


def Be_1s2px(r1, r2):
    return slater_1s(r1) * slater_2px(r2) - slater_1s(r2) * slater_2px(r1)


def Be_1s2py(r1, r2):
    return slater_1s(r1) * slater_2py(r2) - slater_1s(r2) * slater_2py(r1)


def Be_1s2pz(r1, r2):
    return slater_1s(r1) * slater_2pz(r2) - slater_1s(r2) * slater_2pz(r1)


def Be(r12_minus, r34_minus, r12_plus):
    u"""HF = |1s(r1)2s(r2)| * |1s(r3)2s(r4)|"""
    # r12_plus = r12_plus + 3.0
    r34_plus = np.full((100, 100), 6.0)
    vec_1 = vec_3 = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
    phi = np.pi/4.0
    theta = np.pi/2.0
    vec_2 = vec_4 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])[:, np.newaxis, np.newaxis]
    r1n = (r12_plus + r12_minus)/2.0
    r2n = (r12_plus - r12_minus)/2.0
    r1 = vec_1 * r1n[np.newaxis]
    r2 = vec_2 * r2n[np.newaxis]
    r3n = (r34_plus + r34_minus)/2.0
    r4n = (r34_plus - r34_minus)/2.0
    r3 = vec_3 * r3n[np.newaxis]
    r4 = vec_4 * r4n[np.newaxis]
    return Be_1s2s(r1, r2) * Be_1s2s(r3, r4)


def Be_4det(r12_minus, r34_minus, r12_plus):
    u"""CI = phi(1s2, 2s2) + C * phi(1s2, 2p2)"""
    # r12_plus = r12_plus + 3.0
    r34_plus = np.full((100, 100), 6.0)
    vec_1 = vec_3 = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
    phi = np.pi/4.0
    theta = np.pi/2.0
    vec_2 = vec_4 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])[:, np.newaxis, np.newaxis]
    C = -0.15
    r1n = (r12_plus + r12_minus)/2.0
    r2n = (r12_plus - r12_minus)/2.0
    r1 = vec_1 * r1n[np.newaxis]
    r2 = vec_2 * r2n[np.newaxis]
    r3n = (r34_plus + r34_minus)/2.0
    r4n = (r34_plus - r34_minus)/2.0
    r3 = vec_3 * r3n[np.newaxis]
    r4 = vec_4 * r4n[np.newaxis]
    return Be_1s2s(r1, r2) * Be_1s2s(r3, r4) + \
           C * Be_1s2pz(r1, r2) * Be_1s2pz(r3, r4) + \
           C * Be_1s2py(r1, r2) * Be_1s2py(r3, r4) + \
           C * Be_1s2px(r1, r2) * Be_1s2px(r3, r4)


def Be_4det_theta(r12_minus, r34_minus, theta):
    u"""CI = phi(1s2, 2s2) + C * phi(1s2, 2p2)"""
    r12_plus = np.full((100, 100), 6.0)
    r34_plus = np.full((100, 100), 6.0)
    vec_1 = vec_3 = np.array([1, 0, 0])[:, np.newaxis, np.newaxis]
    phi = np.pi/4.0
    vec_2 = vec_4 = np.array([np.cos(phi)*np.sin(theta), np.sin(phi)*np.sin(theta), np.cos(theta)])
    if not theta.shape:
        vec_2 = vec_4 = vec_2[:, np.newaxis, np.newaxis]
    C = -0.15
    r1n = (r12_plus + r12_minus)/2.0
    r2n = (r12_plus - r12_minus)/2.0
    r1 = vec_1 * r1n[np.newaxis]
    r2 = vec_2 * r2n[np.newaxis]
    r3n = (r34_plus + r34_minus)/2.0
    r4n = (r34_plus - r34_minus)/2.0
    r3 = vec_3 * r3n[np.newaxis]
    r4 = vec_4 * r4n[np.newaxis]
    return Be_1s2s(r1, r2) * Be_1s2s(r3, r4) + \
           C * Be_1s2pz(r1, r2) * Be_1s2pz(r3, r4) + \
           C * Be_1s2py(r1, r2) * Be_1s2py(r3, r4) + \
           C * Be_1s2px(r1, r2) * Be_1s2px(r3, r4)


if __name__ == '__main__':
    plot_implicit(Be)
    #plot_implicit(Be_4det)
    #plot_implicit(Be_4det_theta)
