#!/usr/bin/env python3
from sympy import *


def slater(dim):
    """Slater determinant"""
    return Matrix(
        [[phi[i](r[j]) for j in range(dim)] for i in range(dim)]
    )


def gradient(dim):
    """Gradient of slater determinant"""
    return sum(diff(slater(dim).det(), r[i]) for i in range(dim))


def laplacian(dim):
    """Laplacian of slater determinant"""
    return sum(diff(diff(slater(dim).det(), r[i]), r[i]) for i in range(dim))


def derivatives_1(dim):
    return sum(det(Matrix(
        [[diff(phi[i](r[j]), r[j]) if j == k else phi[i](r[j]) for j in range(dim)] for i in range(dim)]
    )) for k in range(dim))


def derivatives_2(dim):
    return sum(det(Matrix(
        [[diff(phi[i](r[j]), r[j], r[j]) if j == k else phi[i](r[j]) for j in range(dim)] for i in range(dim)]
    )) for k in range(dim))


if __name__ == "__main__":

    N = 4
    r = [Symbol(f'r{i}') for i in range(N)]
    phi = [Function(f'phi{i}') for i in range(N)]

    for dim in range(1, N):
        print(f'gradient  {dim} {gradient(dim) - derivatives_1(dim)}')
        print(f'laplacian {dim} {laplacian(dim) - derivatives_2(dim)}')
