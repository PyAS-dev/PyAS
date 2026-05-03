import sympy as sp
import functions
import algebra
x, y, z, t = sp.symbols('x, y, z, t')

def asymptote_conic(expression):
    def quadratic_part(expression, vars):
        polynomial = sp.Poly(expression, *vars)
        quadratic = 0

        for (i, j), coefficient in polynomial.terms():
            if i + j == 2:
                quadratic += coefficient * vars[0]**i * vars[1]**j

        return quadratic
    
    dx = sp.diff(expression, x)
    dy = sp.diff(expression, y)
    critical = sp.solve([dx, dy], (x, y), dict=True)

    if not critical:
        raise ValueError("No finite center")

    h, k = critical[0][x], critical[0][y]
    X, Y = sp.symbols('X Y')
    shifted = expression.subs({x: X + h, y: Y + k}).expand()
    quadratic = quadratic_part(shifted, (X, Y))
    factors = sp.factor(quadratic)
    if factors.is_Mul:
        factor_list = factors.args
    else:
        factor_list = [factors]
    asymptotes = [
        sp.Eq(f.subs({X: x - h, Y: y - k}), 0)
        for f in factor_list
    ]
    return asymptotes

def asymptote_function(P, Q):
    f = algebra.simplify(P / Q)
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

def implicit_asymptote(f):
    m, b = sp.symbols('m, b')
    asymptotes = []
    expression = f.subs(y, m*x)
    leading = sp.limit(expression / x**sp.degree(expression, x), x, sp.oo)
    m_solutions = algebra.solve_equation(leading, 0, m)
    for m_value in m_solutions:
        expression_m_b = f.subs(y, m_value*x + b).expand()
        leading_b = sp.limit(
            expression_m_b / x**sp.degree(expression_m_b, x),
            x, sp.oo
        )

        b_solutions = algebra.solve_equation(leading_b, 0, b)
        for b_value in b_solutions:
            asymptotes.append(sp.Eq(y, m_value*x + b_value))
            
    x_candidates = algebra.solve_equation(f, 0, x)

    for x_c in x_candidates:
        try:
            limit1 = sp.limit(f.subs(x, x_c), y, sp.oo)
            limit2 = sp.limit(f.subs(x, x_c), y, -sp.oo)

            if limit1 is sp.oo or limit2 is sp.oo:
                asymptotes.append(sp.Eq(x, x_c))
        except:
            pass

    return asymptotes

def bezier_curve(P0, P1, P2, P3):
    return (1-t)**3*P0 + 3*(1-t)**2*t*P1 + 3*(1-t)*t**2*P2 + t**3*P3

def coefficients_polynomial(P, variable):
    return sp.Poly(P, variable).all_coeffs()

def coefficients_conic(f):
    return sp.Poly(f, x, y).all_coeffs()

def complex_roots(P):
    return algebra.solve_equation(P, 0, x)

def curvature_function(f, a):
    return functions.absolute_value(sp.diff(f, x, 2).subs(x, a)) / ((1 + sp.diff(f, x)**2)**sp.Rational(3,2)).subs(x, a)

def curvature_parametric(x_t, y_t, a):
    numerator = functions.absolute_value(sp.diff(x_t,t)*sp.diff(y_t,t,2).subs(t, a) - sp.diff(y_t,t)*sp.diff(x_t,t,2)).subs(t, a)
    denominator = ((sp.diff(x_t,t)**2).subs(t, a) + (sp.diff(y_t,t)**2)**sp.Rational(3,2)).subs(t, a)
    return numerator / denominator

def curvature_vector_function(f, a):
    f1, f2 = sp.diff(f,x).subs(x, a), sp.diff(f,x,2).subs(x, a)
    return (f2/(1+f1**2)**sp.Rational(3,2)) * sp.Matrix([-f1,1])

def curvature_vector_parametric(x_t, y_t, a):
    x_prime, y_prime = sp.diff(x_t,t).subs(t, a), sp.diff(y_t,t).subs(t, a)
    x_double_prime, y_double_prime = sp.diff(x_t,t,2).subs(t, a), sp.diff(y_t,t,2).subs(t, a)
    numerator = x_prime*y_double_prime - y_prime*x_double_prime
    denominator = (x_prime**2 + y_prime**2)**sp.Rational(3,2)
    return (numerator/denominator) * sp.Matrix([-y_prime, x_prime])

def curve(x_t, y_t, variable, a, b):
    return (x_t, y_t, (variable, a, b))

def degree(P, variable):
    return sp.degree(P, variable)

def denominator(f):
    denominator = sp.fraction(f)[1]
    return algebra.simplify(denominator)

def derivative_single_variable(f, n):
    return sp.diff(f, x, n)

def partial_derivative(f, variable, n):
    return sp.diff(f, variable, n)

def factors_polynomial(P):
    return sp.factor_list(P)

def factors_integer(n):
    return algebra.prime_factors(n)

def implicit_derivative(f):
    f_x = partial_derivative(f, x, 1)
    f_y = partial_derivative(f, y, 1)
    return -f_x / f_y

def inflection_point(P):
    candidates = algebra.solve_equation(derivative_single_variable(P, 2), 0, x)
    points = [(x0, P.subs(x,x0)) for x0 in candidates]
    return points

def integral(f, variable):
    return sp.integrate(f, variable)

def definite_integral(f, a, b, variable):
    return sp.integrate(f, (variable, a, b))

def integral_between(f, g, a, b, variable):
    return definite_integral(f - g, a, b, variable)

def is_vertex_form(f):
    coefficients = sp.Poly(f, x).all_coeffs()
    if len(coefficients) == 3:
        a,b,c = coefficients
        h = -b/(2*a)
        k = f.subs(x,h)
        vertex_form = a*(x-h)**2 + k
        return True
    else:
        return False

def iteration(f, x0, n, variable):
    x_i = x0
    for _ in range(n):
        x_i = f.subs(variable, x_i)
    return x_i

def iteration_list(f, x0, n, variable):
    sequence = [x0]
    x_i = x0
    for _ in range(n):
        x_i = f.subs(x, x_i)
        sequence.append(x_i)
    return sequence

def left_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(f.subs(x, a + i*dx) * dx for i in range(n))

def limit(f, a, variable):
    return sp.limit(f, variable, a)

def limit_above(f, a, variable):
    return sp.limit(f, variable, a, dir='+')

def limit_below(f, a, variable):
    return sp.limit(f, variable, a, dir='-')

def lower_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(sp.Min(f.subs(x,a+i*dx), f.subs(x,a+(i+1)*dx)) * dx for i in range(n))

def n_integral(f, a, b, variable):
    return sp.N(definite_integral(f, a, b, variable))

def normalize_number(A):
    a = min(A)
    b = max(A)
    if b == a:
        return [0 for _ in A]
    return [(x - a)/(b - a) for x in A]

def normalize_vector(v):
    normalized_all = []
    for vector in v:
        norm = functions.sqrt(sum(coordinate**2 for coordinate in vector))

        if norm == 0:
            normalized_all.append("Zero vector, cannot normalize")
        else:
            normalized_all.append([coordinate / norm for coordinate in vector])

    return normalized_all

def n_solve_ode(left, right, function, initial_conditions=None):
    equation = sp.Eq(left, right)
    
    if initial_conditions is not None:
        return sp.N(sp.dsolve(equation, function, ics=initial_conditions))
    else:
        return sp.N(sp.dsolve(equation, function))

def n_solve_ode_system(equations, initial_conditions=None):
    if len(equations) % 2 != 0:
        raise ValueError("You must pass left/right pairs")

    equation_list = [
        sp.Eq(equations[i], equations[i + 1])
        for i in range(0, len(equations), 2)
    ]

    return sp.N(sp.dsolve(equation_list, ics=initial_conditions))

def numerator(f):
    numerator = sp.fraction(f)[0]
    return numerator

def osculating_circle(f, P):
    x0, y0 = P
    f1 = derivative_single_variable(f, 1)
    f2 = derivative_single_variable(f, 2)
    k = functions.absolute_value(f2)/(1+f1**2)**sp.Rational(3,2)
    R = 1/k.subs(x,x0)
    N = sp.Matrix([-f1.subs(x,x0),1]) / functions.sqrt(1+f1.subs(x,x0)**2)
    C = sp.Matrix([x0,y0]) + R*N
    return (x-C[0])**2 + (y-C[1])**2 - R**2

def parametric_derivative(x_t, y_t):
    return derivative_single_variable(y_t, 1) / derivative_single_variable(x_t, 1)

def partial_fractions(f, variable):
    return sp.apart(f, variable)

def path_parameter(x_t, y_t, P, t, t_start, t_end):
    eq1 = sp.Eq(x_t, P[0])
    eq2 = sp.Eq(y_t, P[1])

    solution = sp.solve([eq1, eq2], t, dict=True)

    if not solution:
        return "Point not on path in interval"

    valid_t = None

    for s in solution:
        t0 = s[t]
        if t0.is_real and t0.is_number:
            if t_start <= float(t0) <= t_end:
                valid_t = float(t0)
                break

    if valid_t is None:
        return "Point not on path in interval"

    s_norm = (valid_t - t_start) / (t_end - t_start)
    return s_norm

def polynomial(f, variable):
    return sp.Poly(f, variable)

def polynomial_interpolation(p_x, p_y):
    if len(p_x) != len(p_y):
        return "p_x and p_y must have same length"
    else:
        points = list(zip(p_x, p_y))
    return sp.interpolate(points, x)

def rectangle_sum(f, a, b, n, d):
    dx = (b - a)/n
    total = 0
    for i in range(n):
        x_i = a + i*dx + d*dx
        total += float(f.subs(x, x_i))*dx
    return total

def removable_discontinuity(P, Q):
    f = algebra.simplify(P / Q)
    holes = []
    
    for r in algebra.solve_equation(Q, 0, x):
        if P.subs(x, r) == 0:
            hole_y = limit(f, x, r)
            holes.append((r, hole_y))
    
    return f, holes

def right_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(f.subs(x, a + (i+1)*dx) * dx for i in range(n))

def root_polynomial(P):
    return algebra.solve_equation(P, 0, x)

def root_initial_value(P, x0):
    return algebra.n_solutions(P, 0, x0, x)

def root_interval(P, a, b, tolerance, max_iterations):
    f_a, f_b = float(P.subs(x, a)), float(P.subs(x, b))
    if f_a * f_b > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iterations):
        c = (a + b) / 2
        f_c = float(P.subs(x, c))
        if functions.absolute_value(f_c) < tolerance or functions.absolute_value(b - a) < tolerance:
            return c
        if f_a * f_c < 0:
            b, f_b = c, f_c
        else:
            a, f_a = c, f_c
    raise RuntimeError("Bisection did not converge")

def root_list(values):
    return [(v, 0) for v in values]

def solve_ode(left, right, function, initial_conditions=None):
    equation = sp.Eq(left, right)
    
    if initial_conditions is not None:
        return sp.dsolve(equation, function, ics=initial_conditions)
    else:
        return sp.dsolve(equation, function)

def solve_ode_system(equations, initial_conditions=None):
    if len(equations) % 2 != 0:
        raise ValueError("You must pass left/right pairs")

    equation_list = [
        sp.Eq(equations[i], equations[i + 1])
        for i in range(0, len(equations), 2)
    ]

    return sp.dsolve(equation_list, ics=initial_conditions)

def spline(n, x_values, y_values):
    points = [(x_values[i], y_values[i]) for i in range(n)]
    return sp.interpolate(points, x)

def svd(A):
    U, S, V = sp.Matrix(A).singular_value_decomposition()
    return {"U": U, "S": S, "V": V}

def taylor_polynomial(f, a, n):
    return sp.series(f, x, a, n + 1).removeO()

def partial_taylor_polynomial(f, variable, a, n):
    return sp.series(f, variable, a, n + 1).removeO()

def trapezoid_sum(f, a, b, n):
    dx = (b - a) / n
    return dx * (0.5*f.subs(x,a) + sum(f.subs(x,a+i*dx) for i in range(1,n)) + 0.5*f.subs(x,b))

def trig_expand(expression, target):
    return sp.expand_trig(expression).rewrite(target)

def trig_combine(expression, target):
    expression1 = sp.expand_trig(expression).rewrite(target)
    expression2 = sp.trigsimp(expression1).rewrite(target)
    expression3 = sp.simplify(expression2).rewrite(target)
    return expression3.rewrite(target)

def trig_simplify(expression, target):
    return sp.trigsimp(expression).rewrite(target)

def turning_point(f):
    f_prime = derivative_single_variable(f, 1)
    critical_points = algebra.solve_equation(f_prime, 0, x)
    f_double_prime = derivative_single_variable(f_prime, 1)
    points = []
    for c_p in critical_points:
        concavity = f_double_prime.subs(x, c_p)
        kind = 'Minimum' if concavity>0 else 'Maximum' if concavity<0 else 'Inconclusive'
        points.append((c_p, f.subs(x,c_p), kind))
    return points

def turning_point_interval(f, a, b):
    f_prime = derivative_single_variable(f, 1)
    critical = algebra.solve_equation(f_prime, 0, x)
    critical_points = [p for p in critical if p.is_real and a <= p <= b]
    f2 = derivative_single_variable(f_prime, 1)

    result = []
    for p in critical:
        value = f.subs(x, p)
        second = f2.subs(x, p)
        if second > 0:
            type = "minimum"
        elif second < 0:
            type = "maximum"
        else:
            type = "flat/inflection"
        result.append((p, value, type))
    return result

def upper_sum(f, a, b, n):
    dx = (b - a) / n
    return sum(max(f.subs(x,a+i*dx), f.subs(x,a+(i+1)*dx)) * dx for i in range(n))
