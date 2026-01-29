# Python Algebra System [ALGEBRA]

import sympy as sp
from sympy.polys import factor
from sympy import QQ

x, y = sp.symbols('x y')

# -----------------------------
# Algebra Operations
# -----------------------------

def common_denominator(P1, Q1, P2, Q2):
    """Return the common denominator of P1/Q1 and P2/Q2"""
    return sp.simplify(Q1 * Q2)

def complete_square(b, c):
    """Return completed square of x^2 + bx + c"""
    return sp.simplify((x + b/2)**2 + (c - b**2/4))

def vector_cross_product(u, v):
    """Return cross product of 3D vectors u and v"""
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    return u.cross(v)

def imaginary_solutions(l, r):
    """Solve a single complex equation l = r"""
    eq = sp.Eq(l, r)
    return sp.solve(eq, x)

def imaginary_solutions_system(l1, r1, l2, r2):
    """Solve a system of complex equations"""
    eq1 = sp.Eq(l1, r1)
    eq2 = sp.Eq(l2, r2)
    return sp.solve((eq1, eq2), (x, y))

def division_integer(a, b):
    """Integer division"""
    return a // b

def division_polynomial(P, Q):
    """Polynomial division"""
    quotient, remainder = sp.div(P, Q)
    return quotient, remainder

def divisors_count(n):
    return sp.divisor_count(n)

def divisors_list(n):
    return sp.divisors(n)

def sigma(n):
    return sp.divisor_sigma(n)

def vector_dot_product(u, v):
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    return u.dot(v)

def expand(expr):
    return sp.expand(expr)

def factorise(expr):
    return sp.factor(expr)

def from_base(n_str, base):
    return int(n_str, base)

def gcd(*nums):
    return sp.gcd(list(nums))

def irrational_factor(expr):
    return factor(expr, extension=QQ.algebraic_field())

def is_factored(expr):
    return expr == sp.factor(expr)

def is_prime(n):
    return sp.isprime(n)

def lcm(*nums):
    return sp.lcm(list(nums))

def maximum(*nums):
    return max(nums)

def minimum(*nums):
    return min(nums)

def mod_number(a, n):
    return a % n

def mod_polynomial(P, Q):
    return sp.rem(P, Q)

def next_prime(n):
    return sp.nextprime(n)

def n_solutions(eq_l, eq_r, start=None):
    eq = sp.Eq(eq_l, eq_r)
    if start is not None:
        return sp.nsolve(eq, x, start)
    return sp.nsolve(eq, x)

def previous_prime(n):
    return sp.prevprime(n)

def prime_factors(n):
    return sp.factorint(n)

def simplify(expr):
    return sp.simplify(expr)

def solve_equation(l, r):
    eq = sp.Eq(l, r)
    return sp.solve(eq)

def solve_inequality(expr):
    return sp.solve_univariate_inequality(expr, x)

def to_base(n, base):
    digits = []
    num = n
    while num > 0:
        digits.append(num % base)
        num //= base
    return digits[::-1] if digits else [0]

# === MENU SETUP ===

FUNCTIONS = {
    "common_denominator": common_denominator, "complete_square": complete_square,
    "vector_cross_product": vector_cross_product, "imaginary_solutions": imaginary_solutions,
    "imaginary_solutions_system": imaginary_solutions_system, "division_integer": division_integer,
    "division_polynomial": division_polynomial, "divisors_count": divisors_count,
    "divisors_list": divisors_list, "sigma": sigma, "vector_dot_product": vector_dot_product,
    "expand": expand, "factorise": factorise, "from_base": from_base, "gcd": gcd,
    "irrational_factor": irrational_factor, "is_factored": is_factored, "is_prime": is_prime, "lcm": lcm,
    "maximum": maximum, "minimum": minimum, "mod_number": mod_number, "mod_polynomial": mod_polynomial,
    "next_prime": next_prime, "n_solutions": n_solutions, "previous_prime": previous_prime,
    "prime_factors": prime_factors, "simplify": simplify, "solve_equation": solve_equation,
    "solve_inequality": solve_inequality, "to_base": to_base
}
