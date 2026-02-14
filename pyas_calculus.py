# Python Algebra System [CALCULUS]
import sympy as sp
x, y, z, t = sp.symbols('x, y, z, t')

def asymptote_conic(a, b):
    return [b*x/a, -b*x/a]

def asymptote_function(P, Q):
    f = sp.simplify(P / Q)
    result = {}
    result["vertical"] = sp.solve(Q, x)
    result["horizontal"] = (
        sp.limit(f, x, sp.oo),
        sp.limit(f, x, -sp.oo)
    )
    if sp.degree(P, x) - sp.degree(Q, x) == 1:
        m = sp.limit(f/x, x, sp.oo)
        b = sp.limit(f - m*x, x, sp.oo)
        result["oblique"] = m*x + b
    return result

def bezier_curve(P0, P1, P2, P3):
    return (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3

def coefficients_polynomial(P):
    return sp.Poly(P, x).all_coeffs()

def coefficients_conic(f):
    return sp.Poly(f, x, y).all_coeffs()

def complex_roots(P):
    return sp.solve(P, x)

def curvature_function(f):
    return abs(sp.diff(f, x, 2)) / (1 + sp.diff(f, x)**2)**sp.Rational(3,2)

def curvature_parametric(x_t, y_t):
    num = abs(sp.diff(x_t,t)*sp.diff(y_t,t,2) - sp.diff(y_t,t)*sp.diff(x_t,t,2))
    den = (sp.diff(x_t,t)**2 + sp.diff(y_t,t)**2)**sp.Rational(3,2)
    return num / den

def curvature_vector_function(f):
    f1, f2 = sp.diff(f,x), sp.diff(f,x,2)
    return (f2/(1+f1**2)**sp.Rational(3,2)) * sp.Matrix([-f1,1])

def curvature_vector_parametric(x_t, y_t):
    xp, yp = sp.diff(x_t,t), sp.diff(y_t,t)
    xpp, ypp = sp.diff(x_t,t,2), sp.diff(y_t,t,2)
    num = xp*ypp - yp*xpp
    den = (xp**2 + yp**2)**sp.Rational(3,2)
    return (num/den) * sp.Matrix([-yp, xp])

def curve(x_t, y_t, a, b):
    return (x_t, y_t, (t, a, b))

def degree(P):
    return sp.degree(P, x)

def denominator(f):
    denom = sp.fraction(f)[1]
    return sp.simplify(denom)

def derivative_single_variable(f, n):
    return sp.diff(f, x, n)

def partial_derivative(f, var, n):
    return sp.diff(f, var, n)

def factors_polynomial(P):
    return sp.factor(P)

def factors_integer(n):
    return sp.divisors(n)

def implicit_derivative(f):
    fx = sp.diff(f, x)
    fy = sp.diff(f, y)
    return -fx / fy

def inflection_point(P):
    candidates = sp.solve(sp.diff(P, x,2), x)
    points = [(x0, P.subs(x,x0)) for x0 in candidates]
    return points

def integral(f, var):
    return sp.integrate(f, var)

def definite_integral(f, a, b, var):
    return sp.integrate(f, (var, a, b))

def integral_between(f, g, a, b):
    return sp.integrate(f - g, (x, a, b))

def is_vertex_form(f):
    coeffs = sp.Poly(f, x).all_coeffs()
    if len(coeffs) == 3:
        a,b,c = coeffs
        h = -b/(2*a)
        k = f.subs(x,h)
        vertex_form = a*(x-h)**2 + k
        return vertex_form

def iteration(f, x0, n):
    xi = x0
    for _ in range(n):
        xi = f.subs(x, xi)
    return xi

def iteration_list(f, x0, n):
    seq = [x0]
    xi = x0
    for _ in range(n):
        xi = f.subs(x, xi)
        seq.append(xi)
    return seq

def left_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(f.subs(x, a + i*dx) * dx for i in range(n))

def limit(f, a):
    return sp.limit(f, x, a)

def limit_above(f, a):
    return sp.limit(f, x, a, dir='+')

def limit_below(f, a):
    return sp.limit(f, x, a, dir='-')

def lower_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(min(f.subs(x,a+i*dx), f.subs(x,a+(i+1)*dx)) * dx for i in range(n))

def n_integral(f, a, b):
    return sp.N(sp.integrate(f, (x, a, b)))

def normalize_number(A):
    a, b = min(A), max(A)
    normalized = [(x - a)/(b - a) if b != a else 0 for x in A]
    return normalized

def normalize_vector(v):
    norm = sp.sqrt(sum([coord**2 for coord in v]))
    if norm == 0:
        return "Zero vector, cannot normalize"
    else:
        normalized = [coord/norm for coord in v]
        return normalized

def n_solve_ode(lhs, rhs):
    sol = sp.dsolve(sp.Eq(lhs, rhs))
    return sol

def numerator(f):
    num = sp.fraction(f)[0]
    return num

def osculating_circle(f, P):
    x0, y0 = P
    f1 = sp.diff(f,x)
    f2 = sp.diff(f,x,2)
    k = abs(f2)/(1+f1**2)**sp.Rational(3,2)
    R = 1/k.subs(x,x0)
    N = sp.Matrix([-f1.subs(x,x0),1]) / sp.sqrt(1+f1.subs(x,x0)**2)
    C = sp.Matrix([x0,y0]) + R*N
    return (x-C[0])**2 + (y-C[1])**2 - R**2

def parametric_derivative(x_t, y_t):
    return sp.diff(y_t, t) / sp.diff(x_t, t)

def partial_fractions(f):
    return sp.apart(f)

def path_parameter(x_t, y_t, P, t_start, t_end):
    sol = sp.solve([x_t - P[0], y_t - P[1]], t)
    t0 = next((s for s in sol if t_start <= s <= t_end), None)
    if t0 is None:
        return "Point not on path in interval"
    else:
        s = (t0 - t_start)/(t_end - t_start)
        return s

def polynomial(f):
    return sp.expand(f)

def polynomial_interpolation(f, n, px, py):
    points = []
    for i in range(n):
        points.append((px, py))
        interp = sp.interpolate(points, x)
    return interp

def rectangle_sum(f, a, b, n, d):
    dx = (b - a)/n
    total = 0
    for i in range(n):
        x_i = a + i*dx + d*dx
        total += float(f.subs(x, x_i))*dx
    return total

def removable_discontinuity(P, Q):
    f = sp.simplify(P/Q)
    holes = []
    for r in sp.solve(Q, x):
        if P.subs(x,r) == 0:
            holes.append((r, sp.limit(f,x,r)))
    return f, holes

def right_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(f.subs(x, a + (i+1)*dx) * dx for i in range(n))

def root_polynomial(P):
    return sp.solve(P, x)

def root_initial_value(P, x0):
    return sp.nsolve(P, x, x0)

def root_interval(P, a, b, tol=1e-6, max_iter=100):
    fa, fb = float(P.subs(x, a)), float(P.subs(x, b))
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        c = (a + b) / 2
        fc = float(P.subs(x, c))
        if abs(fc) < tol or abs(b - a) < tol:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise RuntimeError("Bisection did not converge")

def root_list(vals):
    return [(v, 0) for v in vals]

def solve_ode(lhs, rhs):
    return sp.dsolve(sp.Eq(lhs, rhs))

def spline(n, x_val, y_val, order):
    points = []
    for i in range(n):
        points.append((x_val, y_val))
    spline_poly = sp.interpolate(points, x, method='bspline', degree=order)
    return spline_poly

def svd(A):
    U, S, V = sp.Matrix(A).singular_value_decomposition()
    return {"U": U, "S": S, "V": V}

def taylor_polynomial(f, a, n):
    return sp.series(f, x, a, n+1).removeO()

def trapezoid_sum(f, a, b, n):
    dx = (b - a) / n
    return dx * (0.5*f.subs(x,a) + sum(f.subs(x,a+i*dx) for i in range(1,n)) + 0.5*f.subs(x,b))

def trig_expand(expr):
    return sp.expand_trig(expr)

def trig_combine(expr):
    return sp.trigsimp(sp.expand_trig(expr))

def trig_simplify(expr):
    return sp.trigsimp(expr)

def turning_point(f):
    f_prime = sp.diff(f, x)
    critical_points = sp.solve(f_prime, x)
    f_double_prime = sp.diff(f_prime, x)
    points = []
    for cp in critical_points:
        concavity = f_double_prime.subs(x, cp)
        kind = 'Minimum' if concavity>0 else 'Maximum' if concavity<0 else 'Inconclusive'
        points.append((cp, f.subs(x,cp), kind))
    return points

def upper_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(max(f.subs(x,a+i*dx), f.subs(x,a+(i+1)*dx)) * dx for i in range(n))

FUNCTIONS = {asymptote_conic: "asymptote_conic", asymptote_function: "asymptote_function",
             bezier_curve: "bezier_curve", coefficients_polynomial: "coefficients_polynomial",
             coefficients_conic: "coefficients_conic", complex_roots: "complex_roots",
             curvature_function: "curvature_function", curvature_parametric: "curvature_parametric",
             curvature_vector_function: "curvature_vector_function",
             curvature_vector_parametric: "curvature_vector_parametric", curve: "curve", degree: "degree",
             denominator: "denominator", derivative_single_variable: "derivative_single_variable",
             partial_derivative: "partial_derivative", parametric_derivative: "parametric_derivative",
             factors_polynomial: "factors_polynomial", factors_integer: "factors_integer",
             implicit_derivative: "implicit_derivative", inflection_point: "inflection_point",
             integral: "integral", definite_integral: "definite_integral", integral_between: "integral_between",
             is_vertex_form: "is_vertex_form", iteration: "iteration", iteration_list: "iteration_list",
             left_sum: "left_sum", limit: "limit", limit_above: "limit_above", limit_below: "limit_below",
             lower_sum: "lower_sum", n_integral: "n_integral", normalize_number: "normalize_number",
             normalize_vector: "normalize_vector", numerator: "numerator", osculating_circle: "osculating_circle",
             parametric_derivative: "parametric_derivative",partial_fractions: "partial_fractions",
             path_parameter: "path_parameter", polynomial: "polynomial", rectangle_sum: "rectangle_sum",
             removable_discontinuity: "removable_discontinuity", root_polynomial: "root_polynomial",
             root_initial_value: "root_initial_value", root_interval: "root_interval", root_list: "root_list",
             solve_ode: "solve_ode", spline: "spline", svd: "svd", taylor_polynomial: "taylor_polynomial",
             trapezoid_sum: "trapezoid_sum", trig_combine: "trig_combine", trig_expand: "trig_expand",
             trig_simplify: "trig_simplify", turning_point: turning_point
             }
