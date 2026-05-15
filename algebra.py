import sympy as sp

x, y = sp.symbols('x y')

# -----------------------------
# Algebra Operations
# -----------------------------

def CommonDenominator(f, g):
    f = sp.fraction(f)
    g = sp.fraction(g)
    return sp.lcm(f[1], g[1])

def CompleteSquare(P):
    P = list(sp.Poly(P, x).all_coeffs())
    a = P[0]
    b = P[1]
    c = P[2]
    return sp.simplify(a*(x + b/(2*a))**2 + (c - b**2/(4*a)))

def CrossProduct(u, v):
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    return u.cross(v)

def DivisionInteger(a, b):
    return a // b

def DivisionPolynomial(P, Q):
    quotient, remainder = sp.div(P, Q)
    return quotient, remainder

def DivisorsCount(n):
    return sp.divisor_count(n)

def DivisorsList(n):
    return sp.divisors(n)

def sigma(n):
    return sp.divisor_sigma(n)

def DotProduct(u, v):
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    return u.dot(v)

def expand(expression):
    return sp.expand(expression)

def factor(expression):
    return sp.factor(expression)

def FromBase(n_string, base):
    return int(n_string, base)

def gcd(*expressions):
    return sp.gcd(list(expressions))

def IrrationalFactor(expression):
    return "Did you really think factoring is simple? https://en.wikipedia.org/wiki/Field_extension"

def IsFactored(expression):
    if expression == factor(expression):
        return True
    else:
        return False

def IsPrime(n):
    return sp.isprime(n)

def lcm(*expressions):
    return sp.lcm(list(expressions))

def maximum(*numbers):
    return sp.Max(numbers)

def MaximumFunction(f, variable):
    return sp.maximum(f, variable)

def minimum(*numbers):
    return sp.Min(numbers)

def MinimumFunction(f, variable):
    return sp.minimum(f, variable)

def ModNumber(a, n):
    return a % n

def ModPolynomial(P, Q):
    return sp.rem(P, Q)

def NextPrime(n):
    return sp.nextprime(n)

def NSolveEquation(left, right, start, variable):
    equation = sp.Eq(left, right)
    if start is not None:
        return sp.nsolve(equation, variable, start)
    
def NSolveEquationSystem(*equations, variables):
    variables_ = list(variables)
    sides = equations
    if len(sides) % 2 != 0:
        raise ValueError("You must pass left/right pairs")
    equations = [sp.Eq(sides[i], sides[i + 1]) for i in range(0, len(sides), 2)]
    return sp.N(sp.solve(equations, variables))

def PreviousPrime(n):
    return sp.prevprime(n)

def PrimeFactors(n):
    return sp.factorint(n)

def simplify(expression):
    return sp.simplify(expression)

def SolveEquation(left, right, variable):
    equation = sp.Eq(left, right)
    return sp.solve(equation, variable)

def SolveEquationSystem(equations, variables):
    if len(equations) % 2 != 0:
        raise ValueError("Equations must be in left/right pairs (even number of items).")
    
    eqs = [
        sp.Eq(equations[i], equations[i + 1])
        for i in range(0, len(equations), 2)
    ]
    
    return sp.solve(eqs, variables)

def SolveInequality(expression):
    return sp.solve_univariate_inequality(expression, x)

def ToBase(n, base):
    digits = []
    number = n
    while number > 0:
        digits.append(number % base)
        number //= base
    return digits[::-1] if digits else [0]
