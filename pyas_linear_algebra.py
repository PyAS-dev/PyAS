#Python Algebra System [LINEAR ALGEBRA]

import sympy as sp
from sympy import sqrt, Eq, solve

# ======== Functions ========

def apply_matrix(M, P):
    # Case 1: 2D point, 2×2 matrix
    if P.rows == 2 and M.shape == (2, 2):
        return M * P

    # Case 2: 2D point, 3×3 matrix (homogeneous)
    if P.rows == 2 and M.shape == (3, 3):
        homog = sp.Matrix([P[0], P[1], 1])
        x, y, z = M * homog
        return sp.Matrix([x/z, y/z])

    # Case 3: 3D point, 3×3 matrix
    if P.rows == 3 and M.shape == (3, 3):
        return M * P

    # Case 4: 3D point, 2×2 matrix → extend to 3×3
    if P.rows == 3 and M.shape == (2, 2):
        a, c = M[0, 0], M[0, 1]
        b, d = M[1, 0], M[1, 1]
        N = sp.Matrix([
            [a, c, 0],
            [b, d, 0],
            [0, 0, 1]
        ])
        return N * P
    else:
        return "Unsupported matrix/point size combination"

def det(M):
    return M.det()

def dimension(M):
    return M.shape

def identity_matrix(n):
    return sp.eye(n)

def invert(M):
    return M.inv()

def rank(M):
    return M.rank()

def perpendicular_vector(v):
    perp = sp.Matrix([-v[1], v[0]])
    return perp

def rref(M):
    return M.rref()

def transpose(M):
    return M.T

def unit_perpendicular_vector(a, b):
    v = sp.Matrix([a, b])
    unit_perp = sp.Matrix([-v[1], v[0]]) / sqrt(v[0]**2 + v[1]**2)
    return unit_perp

def unit_vector(x, y):
    length = sqrt(x**2 + y**2)
    if length == 0:
        return "Zero vector has no direction."
    else:
        return sp.Matrix([x/length, y/length])

def vector(x, y):
    return sp.Matrix([x, y])

FUNCTIONS = {apply_matrix: "apply_matrix", det: "det", dimension: "dimension",
             identity_matrix: "identity_matrix", invert: "invert", rank: "rank",
             perpendicular_vector: "perpendicular_vector", rref: "rref", transpose: "transpose",
             unit_perpendicular_vector: "unit_perpendicular_vector", unit_vector: "unit_vector",
             vector: "vector"
             }