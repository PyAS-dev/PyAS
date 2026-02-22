# Python Algebra System [ALGEBRA]

import sympy as sp
from sympy.polys import factor
from sympy import QQ
import pyas_math

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

def complex_solutions(l, r):
    """Solve a single complex equation l = r"""
    equation = sp.Eq(l, r)
    return sp.solve(equation, x)

def complex_solutions_system(left1, right1, left2, right2):
    """Solve a system of complex equations"""
    equation1 = sp.Eq(left1, right1)
    equation2 = sp.Eq(left2, right2)
    return sp.solve((equation1, equation2), (x, y))

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

def from_base(n_string, base):
    return int(n_string, base)

def gcd(*nums):
    return sp.gcd(list(nums))

def irrational_factor(expression):
    return "Did you really think factoring is simple? \n"
    "https://en.wikipedia.org/wiki/Field_extension"

def is_factored(expression):
    if expression == sp.factor(expression):
        return True
    else:
        return False

def is_prime(n):
    return sp.isprime(n)

def lcm(*nums):
    return pyas_math.absolute_value(*nums)/gcd(*nums)

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

def n_solutions(left, right, start=None):
    equation = sp.Eq(left, right)
    if start is not None:
        return sp.nsolve(equation, x, start)
    return sp.nsolve(equation, x)

def previous_prime(n):
    return sp.prevprime(n)

def prime_factors(n):
    return sp.factorint(n)

def simplify(expr):
    return sp.simplify(expr)

def solve_equation(left, right):
    equation = sp.Eq(left, right)
    return sp.solve(equation)

def solve_inequality(expression):
    return sp.solve_univariate_inequality(expression, x)

def to_base(n, base):
    digits = []
    number = n
    while number > 0:
        digits.append(number % base)
        number //= base
    return digits[::-1] if digits else [0]

# === MENU SETUP ===

FUNCTIONS = {
    "common_denominator": common_denominator, "complete_square": complete_square,
    "vector_cross_product": vector_cross_product, "complex_solutions": complex_solutions,
    "complex_solutions_system": complex_solutions_system, "division_integer": division_integer,
    "division_polynomial": division_polynomial, "divisors_count": divisors_count,
    "divisors_list": divisors_list, "sigma": sigma, "vector_dot_product": vector_dot_product,
    "expand": expand, "factorise": factorise, "from_base": from_base, "gcd": gcd,
    "irrational_factor": irrational_factor, "is_factored": is_factored, "is_prime": is_prime, "lcm": lcm,
    "maximum": maximum, "minimum": minimum, "mod_number": mod_number, "mod_polynomial": mod_polynomial,
    "next_prime": next_prime, "n_solutions": n_solutions, "previous_prime": previous_prime,
    "prime_factors": prime_factors, "simplify": simplify, "solve_equation": solve_equation,
    "solve_inequality": solve_inequality, "to_base": to_base
}
