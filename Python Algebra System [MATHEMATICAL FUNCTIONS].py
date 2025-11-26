# Python Algebra System [MATHEMATICAL FUNCTIONS]

import sympy as sp
from random import uniform

# === INPUT AND OUTPUT HELPERS ===

def get_input(prompt):
    """Safely parse numeric or symbolic user input into a SymPy object"""
    while True:
        try:
            expr_str = input(prompt).strip()
            if expr_str == "":
                print("Empty input not allowed.")
                continue
            expr = sp.sympify(expr_str)
            return expr
        except (sp.SympifyError, SyntaxError, TypeError):
            print("Invalid input. Please try again.")

def show(expr):
    """Display symbolic result and optionally numeric evaluation"""
    if expr is None:
        return
    expr_simplified = sp.simplify(expr)
    print(f"Symbolic: {expr_simplified}")
    try:
        val = expr_simplified.evalf()
        if val != expr_simplified:
            print(f"Numeric:  {val}")
    except Exception:
        pass

# === BASIC FUNCTIONS ===

def random_func():
    print(f"Random: {uniform(0,1)}")

def sqrt():
    x = get_input("x = ")
    if x is not None:
        show(sp.sqrt(x))

def cbrt():
    x = get_input("x = ")
    if x is not None:
        show(sp.real_root(x, 3))

def nroot():
    x = get_input("x = ")
    n = get_input("n = ")
    if None not in (x, n):
        show(x ** sp.Rational(1, n))

def absolute_value():
    x = get_input("x = ")
    if x is not None:
        show(sp.Abs(x))

def sgn():
    x = get_input("x = ")
    if x is not None:
        show(sp.sign(x))

def arg():
    z = get_input("z = ")
    if z is not None:
        show(sp.arg(z))

def conjugate():
    z = get_input("z = ")
    if z is not None:
        show(sp.conjugate(z))

def real():
    z = get_input("z = ")
    if z is not None:
        show(sp.re(z))

def imaginary():
    z = get_input("z = ")
    if z is not None:
        show(sp.im(z))

def floor():
    x = get_input("x = ")
    if x is not None:
        show(sp.floor(x))

def ceil():
    x = get_input("x = ")
    if x is not None:
        show(sp.ceiling(x))

def nearest_integer():
    x = get_input("x = ")
    if x is not None:
        show(sp.round(x))

def fractional_part():
    x = get_input("x = ")
    if x is not None:
        show(x - sp.floor(x))

# === LOGARITHMIC AND EXPONENTIAL ===

def log_func():
    x = get_input("x = ")
    b = get_input("b = ")
    if None not in (x, b):
        show(sp.log(x, b))

def exp_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.exp(x))

def ln():
    x = get_input("x = ")
    if x is not None:
        show(sp.log(x))

def log10():
    x = get_input("x = ")
    if x is not None:
        show(sp.log(x, 10))

def log2():
    x = get_input("x = ")
    if x is not None:
        show(sp.log(x, 2))

# === TRIGONOMETRIC FUNCTIONS ===

def sin_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.sin(x))

def cos_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.cos(x))

def tan_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.tan(x))

def sec_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.sec(x))

def csc_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.csc(x))

def cot_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.cot(x))

def asin_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.asin(x))

def acos_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.acos(x))

def atan_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.atan(x))

def atan2_func():
    y = get_input("y = ")
    x = get_input("x = ")
    if None not in (x, y):
        show(sp.atan2(y, x))

# === HYPERBOLIC FUNCTIONS ===

def sinh_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.sinh(x))

def cosh_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.cosh(x))

def tanh_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.tanh(x))

def sech_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.sech(x))

def csch_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.csch(x))

def coth_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.coth(x))

def asinh_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.asinh(x))

def acosh_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.acosh(x))

def atanh_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.atanh(x))

# === SPECIAL FUNCTIONS ===

def gamma_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.gamma(x))

def lower_incomplete_gamma():
    x = get_input("x = ")
    a = get_input("a = ")
    if None not in (x, a):
        t = sp.symbols('t', real=True, positive=True)
        expr = sp.integrate(t**(a-1)*sp.exp(-t), (t, 0, x))
        show(expr)

def gamma_regularized():
    x = get_input("x = ")
    a = get_input("a = ")
    if None not in (x, a):
        t = sp.symbols('t', real=True, positive=True)
        expr = sp.integrate(t**(a-1)*sp.exp(-t), (t, 0, x)) / sp.gamma(a)
        show(expr)

def psi_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.digamma(x))

def beta_func():
    a = get_input("a = ")
    b = get_input("b = ")
    if None not in (a, b):
        show(sp.beta(a, b))

def incomplete_beta_func():
    a = get_input("a = ")
    b = get_input("b = ")
    x = get_input("x = ")
    if None not in (a, b, x):
        show(sp.betainc(a, b, 0, x) * sp.beta(a, b))

def beta_regularized_func():
    a = get_input("a = ")
    b = get_input("b = ")
    x = get_input("x = ")
    if None not in (a, b, x):
        show(sp.betainc(a, b, 0, x))

def erf_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.erf(x))

def nPr_func():
    n = get_input("n = ")
    r = get_input("r = ")
    if None not in (n, r):
        if not (n.is_integer and r.is_integer):
            print("n and r must be integers.")
            return
        show(sp.factorial(n) / sp.factorial(n - r))

def nCr_func():
    n = get_input("n = ")
    r = get_input("r = ")
    if None not in (n, r):
        if not (n.is_integer and r.is_integer):
            print("n and r must be integers.")
            return
        show(sp.binomial(n, r))

def sin_integral_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.Si(x))

def cos_integral_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.Ci(x))

def exp_integral_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.Ei(x))

def zeta_func():
    s = get_input("s = ")
    if s is not None:
        show(sp.zeta(s))

def dirac_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.DiracDelta(x))

def heaviside_func():
    x = get_input("x = ")
    if x is not None:
        show(sp.Heaviside(x))

# === MENU SETUP ===

menu = [
    '01 - random', '02 - sqrt', '03 - cbrt', '04 - nroot', '05 - absolute_value', '06 - sgn',
    '07 - arg', '08 - conjugate', '09 - real', '10 - imaginary', '11 - floor', '12 - ceil', '13 - nearest_integer',
    '14 - fractional_part', '15 - log', '16 - exp', '17 - ln', '18 - log10', '19 - log2', '20 - sin',
    '21 - cos', '22 - tan', '23 - sec', '24 - csc', '25 - cot', '26 - asin', '27 - acos', '28 - atan',
    '29 - atan2', '30 - sinh', '31 - cosh', '32 - tanh', '33 - sech', '34 - csch', '35 - coth', '36 - asinh',
    '37 - acosh', '38 - atanh', '39 - gamma', '40 - lower_incomplete_gamma', '41 - gamma_regularized',
    '42 - psi', '43 - beta', '44 - incomplete_beta', '45 - beta_regularized', '46 - erf', '47 - nPr',
    '48 - nCr', '49 - sin_integral', '50 - cos_integral', '51 - exp_integral', '52 - zeta', '53 - dirac',
    '54 - heaviside'
]

funcs = [
    random_func, sqrt, cbrt, nroot, absolute_value, sgn, arg,
    conjugate, real, imaginary, floor, ceil, nearest_integer,
    fractional_part, log_func, exp_func, ln, log10, log2, sin_func, cos_func,
    tan_func, sec_func, csc_func, cot_func, asin_func, acos_func, atan_func, atan2_func, sinh_func,
    cosh_func, tanh_func, sech_func, csch_func, coth_func, asinh_func, acosh_func, atanh_func,
    gamma_func, lower_incomplete_gamma, gamma_regularized, psi_func, beta_func,
    incomplete_beta_func, beta_regularized_func, erf_func, nPr_func, nCr_func, sin_integral_func,
    cos_integral_func, exp_integral_func, zeta_func, dirac_func, heaviside_func
]

# === MAIN LOOP ===

def main():
    while True:
        print("\n=== PyMath Menu ===")
        for i, item in enumerate(menu, start=0):
            print(f"{item:35}", end="")
            if (i + 1) % 3 == 0:
                print()
        print("\n00 - Exit")
        choice = input("Select option: ").strip()
        if choice == "0" or choice.lower() == "00":
            print("Exiting PyAs...")
            break
        elif choice.isdigit() and 1 <= int(choice) <= len(funcs):
            funcs[int(choice) - 1]()
        else:
            print(f"Invalid option. Enter a number between 0 and {len(funcs)}.")

if __name__ == "__main__":
    main()
