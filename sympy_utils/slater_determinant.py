#!/usr/bin/env python3
from sympy import *


def slater(dim):
    """Slater determinant"""
    return Matrix(
        [[phi[i](r[j]) for j in range(dim)] for i in range(dim)]
    )


def gradient(dim):
    """Gradient of slater determinant"""
    return [diff(slater(dim).det(), r[i]) for i in range(dim)]


def hessian(dim):
    """Hessian of slater determinant"""
    return [[diff(diff(slater(dim).det(), r[i]), r[j]) for i in range(dim)] for j in range(dim)]


def laplacian(dim):
    """Laplacian of slater determinant"""
    return sum(diff(diff(slater(dim).det(), r[i]), r[i]) for i in range(dim))


def gradient_simplified(dim):
    return [det(Matrix(
        [[diff(phi[i](r[j]), r[j]) if j == k else phi[i](r[j]) for j in range(dim)] for i in range(dim)]
    )) for k in range(dim)]


def hessian_simplified(dim):

    def element(i, j, k, l):
        if j == k == l:
            return diff(phi[i](r[j]), r[j], r[j])
        elif j == k or j == l:
            return diff(phi[i](r[j]), r[j])
        else:
            return phi[i](r[j])

    return [[det(Matrix(
        [[element(i, j, k, l) for j in range(dim)] for i in range(dim)]
    )) for k in range(dim)] for l in range(dim)]


def laplacian_simplified(dim):
    return sum(det(Matrix(
        [[diff(phi[i](r[j]), r[j], r[j]) if j == k else phi[i](r[j]) for j in range(dim)] for i in range(dim)]
    )) for k in range(dim))


if __name__ == "__main__":

    N = 4
    r = [Symbol(f'r{i}') for i in range(N)]
    phi = [Function(f'phi{i}') for i in range(N)]

    for dim in range(1, N):
        print(f'laplacian {dim} {laplacian(dim) - laplacian_simplified(dim)}')
        for i in range(dim):
            print(f'gradient  {dim} {gradient(dim)[i] - gradient_simplified(dim)[i]}')
        for i in range(dim):
            for j in range(dim):
                print(f'hessian {dim} {hessian(dim)[i][j] - hessian_simplified(dim)[i][j]}')

