import sympy as sp
import functions
import algebra

x, y = sp.symbols('x, y')

# ======== Functions ========

def ApplyMatrix(M, P):
    P = sp.Matrix(P)
    
    if P.rows == 2 and M.shape == (2, 2):
        return M * P

    if P.rows == 2 and M.shape == (3, 3):
        homogenus = sp.Matrix([P[0], P[1], 1])
        result = M * homogenus

        x, y, z = result
        if z == 0:
            raise ValueError("Point mapped to infinity (z = 0)")

        return sp.Matrix([x / z, y / z])

    if P.rows == 3 and M.shape == (3, 3):
        return M * P

    if P.rows == 3 and M.shape == (2, 2):
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]

        N = sp.Matrix([
            [a, b, 0],
            [c, d, 0],
            [0, 0, 1]
        ])

        return N * P

    raise ValueError("Unsupported matrix/point size combination")

def det(M):
    return M.det()

def dim(M):
    return M.shape

def IdentityMatrix(n):
    return sp.eye(n)

def inv(M):
    return M.inv()

def InverseFunction(f):
    inverse = algebra.solve_equation(y, f, x)
    return inverse

def rank(M):
    return M.rank()

def PerpendicularVector(v):
    perpendicular = sp.Matrix([-v[1], v[0]])
    return perpendicular

def rref(M):
    return M.rref()

def T(M):
    return M.T

def UnitPerpendicularVector(a, b):
    v = sp.Matrix([a, b])
    unit_perpendicular_vector = sp.Matrix([-v[1], v[0]]) / functions.sqrt(v[0]**2 + v[1]**2)
    return unit_perpendicular_vector

def UnitVector(x, y):
    length = pyas_math.sqrt(x**2 + y**2)
    if length == 0:
        return "Zero vector has no direction."
    else:
        return sp.Matrix([x/length, y/length])

def vector(x, y):
    return sp.Matrix([x, y])
