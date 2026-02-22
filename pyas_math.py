# Python Algebra System [MATHEMATICAL FUNCTIONS]

import sympy as sp
from random import uniform
import pyas_calculus

# === BASIC FUNCTIONS ===

def random(x):
    return uniform(0, 1)

def sqrt(x):
    return x**sp.Rational(1, 2)

def cbrt(x):
    return x**sp.Rational(1, 3)

def nroot(x, n):
    return x**sp.Rational(1, n)

def absolute_value(x):
    if x >= 0:
        return x
    else:
        return -x

def sign(x):
    if x > 0:
        return 1
    elif x == 0:
        return 0
    else:
        return -1

def arg(z):
    return sp.arg(z)

def conjugate(z):
    return sp.conjugate(z)

def real(z):
    return sp.re(z)

def imaginary(z):
    return sp.im(z)

def floor(x):
    return sp.floor(x)

def ceil(x):
    return sp.ceiling(x)

def nearest_integer(x):
    return floor(x + sp.Rational(1, 2))

def fractional_part(x):
    return x - floor(x)

# === LOGARITHMIC AND EXPONENTIAL ===

def log(x, b):
    return sp.log(x, b)

def exp(x):
    return sp.exp(x)

def ln(x):
    return sp.log(x)

def log10(x):
    return ln(x)/ln(10)

def log2(x):
    return ln(x)/ln(2)

# === TRIGONOMETRIC FUNCTIONS ===

def sin(x):
    return sp.sin(x)

def cos(x):
    return sp.cos(x)

def tan(x):
    return sin(x)/cos(x)

def sec(x):
    return 1/cos(x)

def csc(x):
    return 1/sin(x)

def cot(x):
    return cos(x)/sin(x)

def arcsin(x):
    return sp.asin(x)

def arccos(x):
    return sp.acos(x)

def arctan(x):
    return sp.atan(x)

def arctan2(y, x):
    if x > 0:
        return arctan(y/x)
    elif x < 0 and y >= 0:
        return arctan(y/x) + sp.pi
    elif x < 0 and y < 0:
        return arctan(y/x) - sp.pi
    elif x == 0 and y > 0:
        return sp.pi/2
    elif x == 0 and y < 0:
        return -sp.pi/2
    else:
        return "undefined"

# === HYPERBOLIC FUNCTIONS ===

def sinh(x):
    return (exp(x) - exp(-x))/2

def cosh(x):
    return (exp(x) + exp(-x))/2
    
def tanh(x):
    return sinh(x)/cosh(x)

def sech(x):
    return 1/cosh(x)

def csch(x):
    return 1/sinh(x)

def coth(x):
    return cosh(x)/sinh(x)

def arcsinh(x):
    return ln(x + sqrt(x**2 + 1))

def arccosh(x):
    return ln(x + sqrt(x**2 - 1))

def arctanh(x):
    return 1/2 * ln((1 + x)/(1 - x))

# === SPECIAL FUNCTIONS ===

def gamma(x):
    t = sp.symbols('t')
    return pyas_calculus.definite_integral(t**(x - 1) * exp(-t), 0, sp.oo, t)

def lower_incomplete_gamma(a, x):
    t = sp.symbols('t')
    return pyas_calculus.definite_integral(t**(a - 1) * exp(-t), 0, x, t)

def gamma_regularized(a, x):
    t = sp.symbols('t')
    a = sp.sympify(a)
    x = sp.sympify(x)
    return lower_incomplete_gamma(a, x)/gamma(a)

def psi(x):
    return pyas_calculus.derivative_single_variable(gamma(x), 1)/gamma(x)

def beta(a, b):
    return (gamma(a)*gamma(b))/gamma(a + b)

def incomplete_beta(a, b, x):
    t = sp.symbols('t')
    a = sp.sympify(a)
    b = sp.sympify(b)
    return pyas_calculus.definite_integral(t**(a - 1) * (1 - t)**(b - 1), 0, x, t)

def beta_regularized(a, b, x):
    t = sp.symbols('t')
    return incomplete_beta(a, b, x)/beta(a, b)

def erf(x):
    t = sp.symbols('t')
    return 2/sqrt(sp.pi) * pyas_calculus.definite_integral(exp(-t**2), 0, x, t)

def nPr(n, r):
    return sp.factorial(n)/sp.factorial(n - r)

def nCr(n, r):
    return sp.factorial(n)/(sp.factorial(r)*sp.factorial(n - r))

def sin_integral(x):
    t = sp.symbols('t')
    return pyas_calculus.definite_integral(sin(t)/t, 0, x, t)

def cos_integral(x):
    t = sp.symbols('t')
    return -pyas_calculus.definite_integral(cos(t)/t, x, sp.oo, t)

def exp_integral(x):
    t = sp.symbols('t')
    return pyas_calculus.definite_integral(exp(t)/t, -sp.oo, x, t)

def zeta(s):
    s = sp.sympify(s)
    n = sp.symbols('n', integer=True, positive=True)
    return sp.summation(1/n**s, (n, 1, sp.oo))

def dirac(x):
    if x != 0:
        return 0
    else:
        return sp.oo

def heaviside(x):
    if x < 0:
        return 0
    else:
        return 1

# === MENU SETUP ===

FUNCTIONS = {
    'random': random, 'sqrt': sqrt, 'cbrt': cbrt, 'nroot': nroot, 'absolute_value': absolute_value, 'sign': sign, 'arg': arg,
    'conjugate': conjugate, 'real': real, 'imaginary': imaginary, 'floor': floor, 'ceil': ceil, 'nearest_integer': nearest_integer,
    'fractional_part':fractional_part, 'log': log, 'exp': exp, 'ln': ln, 'log10': log10, 'log2': log2, 'sin': sin, 'cos': cos,
    'tan': tan, 'sec': sec, 'csc': csc, 'cot': cot, 'arcsin': arcsin, 'arccos': arccos, 'arctan':arctan, 'arctan2': arctan2, 'sinh': sinh,
    'cosh': cosh, 'tanh': tanh, 'sech': sech, 'csch': csch, 'coth': coth, 'arcsinh': arcsinh, 'arccosh': arccosh, 'arctanh': arctanh,
    'gamma': gamma, 'lower_incomplete_gamma': lower_incomplete_gamma, 'gamma_regularized': gamma_regularized, 'psi': psi,
    'beta': beta, 'incomplete_beta': incomplete_beta, 'beta_regularized': beta_regularized, 'erf': erf, 'nPr': nPr, 'nCr': nCr,
    'sin_integral': sin_integral, 'cos_integral': cos_integral, 'exp_integral': exp_integral, 'zeta': zeta, 'dirac': dirac,
    'heaviside': heaviside
}
