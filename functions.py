import sympy as sp
from random import uniform
import calculus

# === BASIC FUNCTIONS ===

def random(x):
    return uniform(0, 1)

def sqrt(x):
    return x**sp.Rational(1, 2)

def cbrt(x):
    return x**sp.Rational(1, 3)

def nroot(x, n):
    return x**sp.Rational(1, n)

def AbsoluteValue(x):
    return sp.Piecewise(
        (x, x >= 0),
        (-x, True)
        )

def sign(x):
    return sp.Piecewise(
        (1, x > 0),
        (0, x == 0),
        (-1, True)
        )

def arg(z):
    return sp.arg(z)

def conjugate(z):
    return sp.conjugate(z)

def Re(z):
    return sp.re(z)

def Im(z):
    return sp.im(z)

def floor(x):
    return sp.floor(x)

def ceil(x):
    return sp.ceiling(x)

def NearestInteger(x):
    return floor(x + sp.Rational(1, 2))

def FractionalPart(x):
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
    return sp.Piecewise(
        (arctan(y/x), x > 0),
        (arctan(y/x) + sp.pi, sp.And(x < 0, y >= 0)),
        (arctan(y/x) - sp.pi, sp.And(x < 0, y < 0)),
        (sp.pi/2, sp.And(x == 0, y > 0)),
        (-sp.pi/2, sp.And(x == 0, y < 0)),
        (sp.nan, sp.And(x == 0, y == 0))
    )

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
    return calculus.DefiniteIntegral(t**(x - 1) * exp(-t), 0, sp.oo, t)

def LowerIncompleteGamma(a, x):
    t = sp.symbols('t')
    return calculus.DefiniteIntegral(t**(a - 1) * exp(-t), 0, x, t)

def GammaRegularized(a, x):
    t = sp.symbols('t')
    a = sp.sympify(a)
    x = sp.sympify(x)
    return LowerIncompleteGamma(a, x)/gamma(a)

def psi(x):
    return calculus.DerivativeSingleVariable(gamma(x), 1)/gamma(x)

def beta(a, b):
    return (gamma(a)*gamma(b))/gamma(a + b)

def IncompleteBeta(a, b, x):
    t = sp.symbols('t')
    a = sp.sympify(a)
    b = sp.sympify(b)
    return calculus.DefiniteIntegral(t**(a - 1) * (1 - t)**(b - 1), 0, x, t)

def BetaRegularized(a, b, x):
    t = sp.symbols('t')
    return IncompleteBeta(a, b, x)/beta(a, b)

def erf(x):
    t = sp.symbols('t')
    return 2/sqrt(sp.pi) * calculus.DefiniteIntegral(exp(-t**2), 0, x, t)

def nPr(n, r):
    return sp.factorial(n)/sp.factorial(n - r)

def nCr(n, r):
    return sp.factorial(n)/(sp.factorial(r)*sp.factorial(n - r))

def SinIntegral(x):
    t = sp.symbols('t')
    return calculus.DefiniteIntegral(sin(t)/t, 0, x, t)

def CosIntegral(x):
    t = sp.symbols('t')
    return -calculus.DefiniteIntegral(cos(t)/t, x, sp.oo, t)

def ExpIntegral(x):
    t = sp.symbols('t')
    return calculus.DefiniteIntegral(exp(t)/t, -sp.oo, x, t)

def zeta(s):
    s = sp.sympify(s)
    n = sp.symbols('n', integer=True, positive=True)
    return sp.summation(1/n**s, (n, 1, sp.oo))

def DiracDelta(x):
    return sp.Piecewise(
        (0, x != 0), 
        (sp.oo, True)
        )

def heaviside(x):
    return sp.Piecewise(
        (0, x < 0), 
        (1, True)
        )
