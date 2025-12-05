# Python Algebra System [CALCULUS]
import sympy as sp

class PyAS_Calculus:
    # =================== Utility Input Methods ===================
    def parse_input_expr(self, prompt):
        return sp.sympify(input(prompt))

    def parse_input_list(self, prompt):
        return (float(num) for num in input(prompt).split(','))

    def parse_input_float(self, prompt):
        return float(input(prompt))

    def __init__(self):
        self.x, self.y, self.z, self.t = sp.symbols('x y z t')

    # ========================= Calculus Functions =========================
    def asymptote(self):
        print("Asymptote types: a) Conic, b) Rational function")
        choice = input("Choose a/b: ").strip().lower()
        if choice == 'a':
            a = self.parse_input_float("a = ")
            b = self.parse_input_float("b = ")
            print("Asymptotes: y =", sp.simplify(b*self.x/a), "and y =", sp.simplify(-b*self.x/a))
        elif choice == 'b':
            P = self.parse_input_expr("P(x) = ")
            Q = self.parse_input_expr("Q(x) = ")
            f = sp.simplify(P / Q)
            print("Vertical asymptotes:", sp.solve(Q, self.x))
            print("Horizontal asymptotes:", sp.limit(f, self.x, sp.oo), sp.limit(f, self.x, -sp.oo))
            if sp.degree(P, self.x) - sp.degree(Q, self.x) == 1:
                m = sp.limit(f/self.x, self.x, sp.oo)
                b = sp.limit(f - m*self.x, self.x, sp.oo)
                print("Oblique asymptote: y =", m*self.x + b)

    def bezier_curve(self):
        P0 = sp.Matrix(self.parse_input_list("P0 = "))
        P1 = sp.Matrix(self.parse_input_list("P1 = "))
        P2 = sp.Matrix(self.parse_input_list("P2 = "))
        P3 = sp.Matrix(self.parse_input_list("P3 = "))
        t = self.t
        B = (1 - t)**3 * P0 + 3*(1 - t)**2*t*P1 + 3*(1 - t)*t**2*P2 + t**3*P3
        print("Bezier curve:", sp.simplify(B))

    def coefficients(self):
        choice = input("Polynomial or Conic? p/c: ").strip().lower()
        if choice == 'p':
            P = self.parse_input_expr("P(x) = ")
            print("Coefficients:", sp.Poly(P, self.x).all_coeffs())
        elif choice == 'c':
            f = self.parse_input_expr("f(x,y) = ")
            print("Coefficients:", sp.Poly(f, self.x, self.y).all_coeffs())

    def complex_roots(self):
        P = self.parse_input_expr("P(x) = ")
        print("Roots:", sp.solve(P, self.x))

    def curvature(self):
        choice = input("Function or parametric? f/p: ").strip().lower()
        if choice == 'f':
            f = self.parse_input_expr("f(x) = ")
            kappa = sp.Abs(sp.diff(f, self.x, 2)) / (1 + sp.diff(f, self.x)**2)**sp.Rational(3,2)
            print("Curvature:", sp.simplify(kappa))
        elif choice == 'p':
            x_t = self.parse_input_expr("x(t) = ")
            y_t = self.parse_input_expr("y(t) = ")
            t = self.t
            kappa = sp.Abs(sp.diff(x_t, t)*sp.diff(y_t, t,2) - sp.diff(y_t, t)*sp.diff(x_t, t,2)) / ((sp.diff(x_t,t)**2 + sp.diff(y_t,t)**2)**sp.Rational(3,2))
            print("Curvature:", sp.simplify(kappa))

    def curvature_vector(self):
        choice = input("Function or parametric? f/p: ").strip().lower()
        if choice == 'f':
            f = self.parse_input_expr("f(x) = ")
            f_prime = sp.diff(f, self.x)
            f_double_prime = sp.diff(f, self.x, 2)
            kappa_vec = (f_double_prime / (1 + f_prime**2)**sp.Rational(3,2)) * sp.Matrix([-f_prime,1])
            print("Curvature vector:", sp.simplify(kappa_vec))
        elif choice == 'p':
            x_t = self.parse_input_expr("x(t) = ")
            y_t = self.parse_input_expr("y(t) = ")
            t = self.t
            x_prime = sp.diff(x_t, t)
            y_prime = sp.diff(y_t, t)
            x_double_prime = sp.diff(x_t, t,2)
            y_double_prime = sp.diff(y_t, t,2)
            numerator = x_prime*y_double_prime - y_prime*x_double_prime
            denominator = (x_prime**2 + y_prime**2)**sp.Rational(3,2)
            kappa_vec_param = (numerator/denominator)*sp.Matrix([-y_prime,x_prime])
            print("Curvature vector (parametric):", sp.simplify(kappa_vec_param))

    def curve(self):
        x_t = self.parse_input_expr("x(t) = ")
        y_t = self.parse_input_expr("y(t) = ")
        a = self.parse_input_expr("a = ")
        b = self.parse_input_expr("b = ")
        print(f"Parametric curve from t={a} to t={b}: ({x_t}, {y_t})")

    def degree(self):
        P = self.parse_input_expr("P(x) = ")
        print("Degree:", sp.degree(P, self.x))

    def denominator(self):
        f = self.parse_input_expr("f(x) = ")
        denom = sp.fraction(f)[1]
        print("Denominator:", sp.simplify(denom))

    def derivative(self):
        choice = input("Derivative type: single-variable (s), partial (p), parametric (c)? ").strip().lower()

        if choice == 's':  # Single-variable
            f = self.parse_input_expr("f(x) = ")
            n = int(input("Order n = "))
            deriv = sp.diff(f, self.x, n)
            print(f"{n}-th derivative:", sp.simplify(deriv))

        elif choice == 'p':  # Partial derivative
            f = self.parse_input_expr("f(x,y,...) = ")
            vars_in_f = list(f.free_symbols)
            print("Variables detected:", vars_in_f)
            var_choice = input(f"Differentiate w.r.t which variable? (default {vars_in_f[0]}) ").strip()
            var = sp.symbols(var_choice) if var_choice else vars_in_f[0]
            n = int(input("Order n = "))
            deriv = sp.diff(f, var, n)
            print(f"{n}-th partial derivative w.r.t {var}:", sp.simplify(deriv))

        elif choice == 'c':  # Parametric curve
            x_t = self.parse_input_expr("x(t) = ")
            y_t = self.parse_input_expr("y(t) = ")
            t = self.t
            n = int(input("Order n = "))
        
        # First derivative dy/dx
            dy_dx = sp.diff(y_t, t) / sp.diff(x_t, t)
            deriv = dy_dx
        # Higher-order derivatives using chain rule recursively
            for _ in range(1, n):
                deriv = sp.diff(deriv, t) / sp.diff(x_t, t)
            print(f"{n}-th derivative of parametric curve dy/dx:", sp.simplify(deriv))

        else:
            print("Invalid choice.")

    def factors(self):
        choice = input("Polynomial or Integer? p/i: ").strip().lower()
        if choice == 'p':
            P = self.parse_input_expr("P(x) = ")
            print("Factors:", sp.factor(P))
        elif choice == 'i':
            n = int(input("n = "))
            print("Integer divisors:", sp.divisors(n))

    def implicit_derivative(self):
        f = self.parse_input_expr("f(x,y) = ")
        fx = sp.diff(f, self.x)
        fy = sp.diff(f, self.y)
        print("dy/dx =", sp.simplify(-fx/fy))

    def inflection_point(self):
        P = self.parse_input_expr("P(x) = ")
        candidates = sp.solve(sp.diff(P, self.x,2), self.x)
        points = [(x0, P.subs(self.x,x0)) for x0 in candidates]
        print("Inflection points:", points)

    def integral(self):
        f = self.parse_input_expr("f(x, y, ...) = ")
        vars_in_f = list(f.free_symbols)
        print("Variables detected:", vars_in_f)

        var_choice = input(f"Integrate w.r.t which variable? (default {vars_in_f[0]}): ").strip()
        var = sp.symbols(var_choice) if var_choice else vars_in_f[0]

        definite = input("Definite integral? (y/n): ").strip().lower()
        if definite == 'y':
            a = self.parse_input_expr("Lower limit a = ")
            b = self.parse_input_expr("Upper limit b = ")
            integral_result = sp.integrate(f, (var, a, b))
        else:
            integral_result = sp.integrate(f, var)

        integral_simplified = sp.simplify(integral_result)
        print("Integral result:", integral_simplified)

    # Optional numeric evaluation
        numeric_eval = input("Evaluate numerically? (y/n): ").strip().lower()
        if numeric_eval == 'y':
            print("Numeric value:", sp.N(integral_simplified))

    def integral_between(self):
        f = self.parse_input_expr("f(x) = ")
        g = self.parse_input_expr("g(x) = ")
        a = self.parse_input_expr("a = ")
        b = self.parse_input_expr("b = ")
        print("Integral between f and g:", sp.integrate(f-g, (self.x,a,b)))

    def is_vertex_form(self):
        f = self.parse_input_expr("f(x) = ")
        coeffs = sp.Poly(f, self.x).all_coeffs()
        if len(coeffs) == 3:
            a,b,c = coeffs
            h = -b/(2*a)
            k = f.subs(self.x,h)
            vertex_form = a*(self.x-h)**2 + k
            print("Vertex form:", vertex_form)

    def iteration(self):
        f = self.parse_input_expr("f(x) = ")
        x0 = self.parse_input_float("x0 = ")
        n = int(input("n = "))
        xi = x0
        for _ in range(n):
            xi = f.subs(self.x, xi)
        print(f"f^{n}({x0}) =", xi)

    def iteration_list(self):
        f = self.parse_input_expr("f(x) = ")
        x0 = self.parse_input_float("x0 = ")
        n = int(input("n = "))
        seq = [x0]
        xi = x0
        for _ in range(n):
            xi = f.subs(self.x, xi)
            seq.append(xi)
        print("Iteration sequence:", seq)

    def left_sum(self):
        f = self.parse_input_expr("f(x) = ")
        a = float(input("a = "))
        b = float(input("b = "))
        n = int(input("n = "))
        dx = (b-a)/n
        total = 0
        for i in range(n):
            x_i = a + i*dx
            total += f.subs(self.x, x_i)*dx
        print("Left Riemann sum:", total)

    def limit(self):
        f = self.parse_input_expr("f(x) = ")
        a = self.parse_input_expr("a = ")
        print("Limit:", sp.limit(f,self.x,a))

    def limit_above(self):
        f = self.parse_input_expr("f(x) = ")
        a = self.parse_input_expr("a = ")
        print("Right-hand limit:", sp.limit(f,self.x,a,dir='+'))

    def limit_below(self):
        f = self.parse_input_expr("f(x) = ")
        a = self.parse_input_expr("a = ")
        print("Left-hand limit:", sp.limit(f,self.x,a,dir='-'))

    def lower_sum(self):
        f = self.parse_input_expr("f(x) = ")
        a = float(input("a = "))
        b = float(input("b = "))
        n = int(input("n = "))
        dx = (b-a)/n
        total = 0
        for i in range(n):
            x_i = a + i*dx
            x_next = x_i + dx
            m_i = min(f.subs(self.x,x_i), f.subs(self.x,x_next))
            total += m_i*dx
        print("Lower Riemann sum:", total)

    def n_integral(self):
        f = self.parse_input_expr("f(x) = ")
        a = self.parse_input_expr("a = ")
        b = self.parse_input_expr("b = ")
        print("Numeric integral:", sp.N(sp.integrate(f,(self.x,a,b))))
        
    def normalize(self):
        choice = input("Normalize number or vector? n/v: ").strip().lower()
        if choice == 'n':
            A = self.parse_input_list("Enter numbers (comma-separated): ")
            a, b = min(A), max(A)
            normalized = [(x - a)/(b - a) if b != a else 0 for x in A]
            print("Normalized numbers:", normalized)
        elif choice == 'v':
            v = self.parse_input_list("Enter vector (comma-separated): ")
            norm = sp.sqrt(sum([coord**2 for coord in v]))
            if norm == 0:
                print("Zero vector, cannot normalize")
            else:
                normalized = [coord/norm for coord in v]
                print("Normalized vector:", normalized)

    def n_solve_ode(self):
        lhs = self.parse_input_expr("LHS = ")
        rhs = self.parse_input_expr("RHS = ")
        sol = sp.dsolve(sp.Eq(lhs, rhs))
        print("ODE solution:", sol)

    def numerator(self):
        f = self.parse_input_expr("f(x) = ")
        num = sp.fraction(f)[0]
        print("Numerator:", num)

    def osculating_circle(self):
        f = self.parse_input_expr("f(x) = ")
        point = self.parse_input_list("Enter point P (x,y): ")
        x0, y0 = point
        f_prime = sp.diff(f, self.x)
        f_double_prime = sp.diff(f_prime, self.x)
        kappa = sp.Abs(f_double_prime)/(1 + f_prime**2)**(3/2)
        kappa_val = kappa.subs(self.x, x0)
        R = 1/kappa_val
        N = sp.Matrix([-f_prime.subs(self.x,x0), 1])/sp.sqrt(1 + f_prime.subs(self.x,x0)**2)
        C = sp.Matrix([x0, y0]) + R*N
        xc, yc = C
        circle_eq = sp.simplify((self.x - xc)**2 + (self.y - yc)**2 - R**2)
        print("Osculating circle equation:", circle_eq)

    def parametric_derivative(self):
        x_t = self.parse_input_expr("x(t) = ")
        y_t = self.parse_input_expr("y(t) = ")
        x_prime = sp.diff(x_t, self.t)
        y_prime = sp.diff(y_t, self.t)
        print("Parametric derivative dy/dx =", sp.simplify(y_prime/x_prime))
        
    def partial_fractions(self):
        f = self.part_input_expr("f(x) = ")
        print(sp.apart(f))

    def path_parameter(self):
        x_t = self.parse_input_expr("x(t) = ")
        y_t = self.parse_input_expr("y(t) = ")
        P = self.parse_input_list("Enter point P (x,y): ")
        t_start = self.parse_input_expr("t_start = ")
        t_end = self.parse_input_expr("t_end = ")
        sol = sp.solve([x_t - P[0], y_t - P[1]], self.t)
        t0 = next((s for s in sol if t_start <= s <= t_end), None)
        if t0 is None:
            print("Point not on path in interval")
        else:
            s = (t0 - t_start)/(t_end - t_start)
            print("Path parameter (normalized 0-1):", s)

    def polynomial(self):
        choice = input("Polynomial function or interpolate points? f/p: ").strip().lower()
        if choice == 'f':
            f = self.parse_input_expr("f(x) = ")
            print("Expanded polynomial:", sp.expand(f))
        elif choice == 'p':
            n = int(input("Number of points: "))
            points = []
            for i in range(n):
                px = self.parse_input_float(f"x{i} = ")
                py = self.parse_input_float(f"y{i} = ")
                points.append((px, py))
            interp = sp.interpolate(points, self.x)
            print("Interpolated polynomial:", interp)

    def rectangle_sum(self):
        f = self.parse_input_expr("f(x) = ")
        a = float(input("a = "))
        b = float(input("b = "))
        n = int(input("n = "))
        d = float(input("d = "))
        dx = (b - a)/n
        total = 0
        for i in range(n):
            x_i = a + i*dx + d*dx
            total += float(f.subs(self.x, x_i))*dx
        print("Rectangle sum:", total)

    def removable_discontinuity(self):
        P = self.parse_input_expr("P(x) = ")
        Q = self.parse_input_expr("Q(x) = ")
        f = sp.simplify(P/Q)
        holes = []
        for r in sp.solve(Q, self.x):
            if sp.simplify(P.subs(self.x,r)) == 0:
                y_val = sp.limit(f, self.x, r)
                holes.append((r, y_val))
        print("Simplified function:", f)
        print("Removable discontinuities:", holes)

    def root(self):
        choice = input("Polynomial root, initial value, or interval? p/i/r: ").strip().lower()
        if choice == 'p':
            P = self.parse_input_expr("P(x) = ")
            print("Roots:", sp.solve(P, self.x))
        elif choice == 'i':
            P = self.parse_input_expr("P(x) = ")
            x0 = self.parse_input_float("x0 = ")
            print("Root near x0:", sp.nsolve(P, self.x, x0))
        elif choice == 'r':
            P = self.parse_input_expr("P(x) = ")
            a = self.parse_input_float("a = ")
            b = self.parse_input_float("b = ")
            tol = 1e-6
            fa = float(P.subs(self.x,a))
            fb = float(P.subs(self.x,b))
            if fa*fb>0:
                print("f(a) and f(b) must have opposite signs")
                return
            max_iter = 100
            for _ in range(max_iter):
                c = (a+b)/2
                fc = float(P.subs(self.x,c))
                if abs(fc)<tol or abs(b-a)<tol:
                    print("Root in interval:", c)
                    return
                if fa*fc<0:
                    b, fb = c, fc
                else:
                    a, fa = c, fc
            print("Did not converge")

    def root_list(self):
        vals = self.parse_input_list("Enter roots: ")
        print("Roots list:", [(v,0) for v in vals])

    def solve_ode(self):
        lhs = self.parse_input_expr("LHS = ")
        rhs = self.parse_input_expr("RHS = ")
        ode = sp.Eq(lhs, rhs)
        print("ODE solution:", sp.dsolve(ode))

    def spline(self):
        n = int(input("Number of points: "))
        points = []
        for i in range(n):
            x_val = self.parse_input_float(f"x{i} = ")
            y_val = self.parse_input_float(f"y{i} = ")
            points.append((x_val, y_val))
        order = int(input("Spline order = "))
        spline_poly = sp.interpolate(points, self.x, method='bspline', degree=order)
        print("Spline interpolation:", spline_poly)

    def svd(self):
        A = self.parse_input_expr("Matrix A (list of lists) = ")
        A = sp.Matrix(A)
        U, S, V = A.singular_value_decomposition()
        print("SVD:")
        print("U =", U)
        print("S =", S)
        print("V =", V)

    def taylor_polynomial(self):
        f = self.parse_input_expr("f(x) = ")
        a = self.parse_input_float("a = ")
        n = int(input("Order n = "))
        series = f.series(self.x, a, n+1).removeO()
        print("Taylor polynomial:", sp.expand(series))

    def trapezoid_sum(self):
        f = self.parse_input_expr("f(x) = ")
        a = float(input("a = "))
        b = float(input("b = "))
        n = int(input("n = "))
        dx = (b-a)/n
        total = 0.5*(f.subs(self.x,a)+f.subs(self.x,b))
        for i in range(1,n):
            total += f.subs(self.x,a + i*dx)
        print("Trapezoid sum:", total)

    def trig_combine(self):
        expr = self.parse_input_expr("f = ")
        print("Trig combined:", sp.trigsimp(sp.expand_trig(expr)))

    def trig_expand(self):
        expr = self.parse_input_expr("f = ")
        print("Trig expanded:", sp.trig_expand(expr))

    def trig_simplify(self):
        expr = self.parse_input_expr("f = ")
        print("Trig simplified:", sp.trigsimp(expr))

    def turning_point(self):
        f = self.parse_input_expr("f(x) = ")
        f_prime = sp.diff(f, self.x)
        critical_points = sp.solve(f_prime, self.x)
        f_double_prime = sp.diff(f_prime, self.x)
        points = []
        for cp in critical_points:
            concavity = f_double_prime.subs(self.x, cp)
            kind = 'Minimum' if concavity>0 else 'Maximum' if concavity<0 else 'Inconclusive'
            points.append((cp, f.subs(self.x,cp), kind))
        print("Turning points:", points)

    def upper_sum(self):
        f = self.parse_input_expr("f(x) = ")
        a = float(input("a = "))
        b = float(input("b = "))
        n = int(input("n = "))
        dx = (b-a)/n
        total = 0
        for i in range(n):
            x_left = a + i*dx
            x_right = x_left + dx
            M_i = max(f.subs(self.x,x_left), f.subs(self.x,x_right))
            total += M_i*dx
        print("Upper Riemann sum:", total)

    # =================== Initialization ===================
    def __init__(self):
        self.x, self.y, self.z, self.t = sp.symbols('x y z t')
        self.menu = self.menu = [
    "asymptote",
    "bezier_curve",
    "coefficients",
    "complex_roots",
    "curvature",
    "curvature_vector",
    "parametric_curve",
    "degree",
    "denominator",
    "derivative",
    "factors",
    "implicit_derivative",
    "inflection_point",
    "integral",
    "integral_between",
    "vertex_form",
    "iteration",
    "iteration_list",
    "left_riemann_sum",
    "limit",
    "right_hand_limit",
    "left_hand_limit",
    "lower_riemann_sum",
    "numeric_integral",
    "normalize",
    "solve_ode_numerically",
    "numerator",
    "osculating_circle",
    "parametric_derivative",
    "partial_fractions",
    "path_parameter",
    "polynomial",
    "rectangle_sum",
    "removable_discontinuity",
    "root",
    "root_list",
    "solve_ode",
    "spline",
    "svd",
    "taylor_polynomial",
    "trapezoid_sum",
    "trig_combine",
    "trig_expand",
    "trig_simplify",
    "turning_point",
    "upper_riemann_sum"
]

        # Assign funcs AFTER all methods are defined
        self.funcs = [
            self.asymptote, self.bezier_curve, self.coefficients, self.complex_roots,
            self.curvature, self.curvature_vector, self.curve, self.degree, self.denominator,
            self.derivative, self.factors, self.implicit_derivative, self.inflection_point,
            self.integral, self.integral_between, self.is_vertex_form, self.iteration,
            self.iteration_list, self.left_sum, self.limit, self.limit_above, self.limit_below,
            self.lower_sum, self.n_integral, self.normalize, self.n_solve_ode, self.numerator,
            self.osculating_circle, self.parametric_derivative, self.partial_fractions,
            self.path_parameter, self.polynomial, self.rectangle_sum,
            self.removable_discontinuity, self.root, self.root_list, self.solve_ode,
            self.spline, self.svd, self.taylor_polynomial, self.trapezoid_sum,
            self.trig_combine, self.trig_expand, self.trig_simplify, self.turning_point,
            self.upper_sum
        ]

    # =================== Main Loop ===================
    def run(self):
        while True:
            print("\n=== PyCalculus Menu ===")
            for i, item in enumerate(self.menu, start=1):
                print(f"{i:02d} - {item:30}", end="")
                if i % 3 == 0: print()
            print()
            choice = input("Select option: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(self.funcs):
                self.funcs[int(choice)-1]()
            else:
                print("Invalid option.")

# =================== Run ===================
if __name__ == "__main__":
    calculus = PyAS_Calculus()
    calculus.run()

