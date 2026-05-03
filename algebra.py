import sympy as sp
import math

x, y = sp.symbols('x y')

# -----------------------------
# Algebra Operations
# -----------------------------

def common_denominator(f, g):
    f = sp.fraction(f)
    g = sp.fraction(g)
    return sp.lcm(f[1], g[1])

def complete_square(P):
    P = list(sp.Poly(P, x).all_coeffs())
    a = P[0]
    b = P[1]
    c = P[2]
    return sp.simplify(a*(x + b/(2*a))**2 + (c - b**2/(4*a)))

def vector_cross_product(u, v):
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    return u.cross(v)

def complex_solutions(left, right, variable):
    equation = sp.Eq(left, right)
    return sp.solve(equation, variable)

def complex_solutions_system(*equations, variables):
    variables_ = list(variables)
    sides = equations
    if len(sides) % 2 != 0:
        raise ValueError("You must pass left/right pairs")
    equations = [sp.Eq(sides[i], sides[i + 1]) for i in range(0, len(sides), 2)]
    return sp.solve(equations, variables)

def division_integer(a, b):
    return a // b

def division_polynomial(P, Q):
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

def expand(expression):
    return sp.expand(expression)

def factorise(expression):
    return sp.factor(expression)

def from_base(n_string, base):
    return int(n_string, base)

def gcd(*expressions):
    return sp.gcd(list(expressions))

def irrational_factor(expression):
    return "Did you really think factoring is simple? https://en.wikipedia.org/wiki/Field_extension"

def is_factored(expression):
    if expression == factorise(expression):
        return True
    else:
        return False

def is_prime(n):
    return sp.isprime(n)

def lcm(*expressions):
    return sp.lcm(list(expressions))

def maximum(*numbers):
    return sp.Max(numbers)

def maximum_function(f, variable):
    return sp.maximum(f, variable)

def minimum(*numbers):
    return sp.Min(numbers)

def minimum_function(f, variable):
    return sp.minimum(f, variable)

def mod_number(a, n):
    return a % n

def mod_polynomial(P, Q):
    return sp.rem(P, Q)

def next_prime(n):
    return sp.nextprime(n)

def n_solve_equation(left, right, start, variable):
    equation = sp.Eq(left, right)
    if start is not None:
        return sp.nsolve(equation, variable, start)
    
def n_solve_equation_system(*equations, variables):
    variables_ = list(variables)
    sides = equations
    if len(sides) % 2 != 0:
        raise ValueError("You must pass left/right pairs")
    equations = [sp.Eq(sides[i], sides[i + 1]) for i in range(0, len(sides), 2)]
    return sp.N(sp.solve(equations, variables))

def previous_prime(n):
    return sp.prevprime(n)

def prime_factors(n):
    return sp.factorint(n)

def simplify(expression):
    return sp.simplify(expression)

def solve_equation(left, right, variable):
    equation = sp.Eq(left, right)
    return sp.solve(equation, variable)

def solve_equation_system(*equations, variables):
    variables_ = list(variables)
    sides = equations
    if len(sides) % 2 != 0:
        raise ValueError("You must pass left/right pairs")
    equations = [sp.Eq(sides[i], sides[i + 1]) for i in range(0, len(sides), 2)]
    return sp.solve(equations, variables)

def solve_inequality(expression):
    return sp.solve_univariate_inequality(expression, x)

def to_base(n, base):
    digits = []
    number = n
    while number > 0:
        digits.append(number % base)
        number //= base
    return digits[::-1] if digits else [0]
