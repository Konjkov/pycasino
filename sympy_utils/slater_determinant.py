#!/usr/bin/env python3
from sympy import *
from sympy.utilities.codegen import codegen


def slater(dim):
    """Slater determinant"""
    return Matrix([[phi[i](r[j]) for j in range(dim)] for i in range(dim)])


def slater_derivatives_1(dim):
    """Slater determinant of first derivatives"""
    return Matrix([[diff(phi[i](r[j]), r[j]) for j in range(dim)] for i in range(dim)])


def slater_derivatives_2(dim):
    """Slater determinant of second derivatives"""
    return Matrix([[diff(phi[i](r[j]), r[j], r[j]) for j in range(dim)] for i in range(dim)])


def gradient(dim):
    """Gradient of slater determinant"""
    return sum(diff(slater(dim).det(), r[i]) for i in range(dim))


def laplacian(dim):
    """Laplacian of slater determinant"""
    return sum(diff(diff(slater(dim).det(), r[i]), r[i]) for i in range(dim))


def derivatives_1(dim):
    res = 0
    d1 = slater_derivatives_1(dim)
    for i in range(dim):
        tmp = slater(dim)
        tmp.col_del(i)
        tmp = tmp.col_insert(i, d1.col(i))
        res += tmp.det()
    return res


def derivatives_2(dim):
    res = 0
    d2 = slater_derivatives_2(dim)
    for i in range(dim):
        tmp = slater(dim)
        tmp.col_del(i)
        tmp = tmp.col_insert(i, d2.col(i))
        res += tmp.det()
    return res


if __name__ == "__main__":

    N = 4
    r = [Symbol(f'r{i}') for i in range(N)]
    phi = [Function(f'phi{i}') for i in range(N)]

    for dim in range(1, N):
        print(f'gradient  {dim} {gradient(dim) - derivatives_1(dim)}')
        print(f'laplacian {dim} {laplacian(dim) - derivatives_2(dim)}')
