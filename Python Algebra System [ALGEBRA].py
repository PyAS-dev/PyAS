# Python Algebra System [ALGEBRA]

import sympy as sp
from sympy.polys import factor
from sympy import QQ

x, y = sp.symbols('x, y')

# 01 - common_denominator
def common_denominator():
    P1 = sp.sympify(input("P1: "))
    Q1 = sp.sympify(input("Q1: "))
    P2 = sp.sympify(input("P2: "))
    Q2 = sp.sympify(input("Q2: "))
    
    f = P1 / Q1
    g = P2 / Q2
    common_den = sp.simplify(Q1 * Q2)
    print("Common denominator:", common_den)

# 02 - complete_square
def complete_square():
    b = sp.sympify(input("b = "))
    c = sp.sympify(input("c = "))
    
    completed_square = sp.simplify((x + b / 2)**2 + (c - b**2 / 4))
    print("Completed square:", completed_square)

# 03 - vector_cross_product
def vector_cross_product():
    u = [sp.sympify(input(f"a{i+1} = ")) for i in range(3)]
    v = [sp.sympify(input(f"b{i+1} = ")) for i in range(3)]
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    print("Cross product:", u.cross(v))

# 04 - imaginary_solutions
def imaginary_solutions():
    print("4A - complex_solutions\n4B - system_of_complex_equations")
    choice = input().lower()
    if choice == 'a':
        l = sp.sympify(input("Left side: "))
        r = sp.sympify(input("Right side: "))
        eq = sp.Eq(l, r)
        print("Solutions:", sp.solve(eq, x))
    elif choice == 'b':
        l1 = sp.sympify(input("Left1: "))
        r1 = sp.sympify(input("Right1: "))
        l2 = sp.sympify(input("Left2: "))
        r2 = sp.sympify(input("Right2: "))
        eq1 = sp.Eq(l1, r1)
        eq2 = sp.Eq(l2, r2)
        print("Solutions:", sp.solve((eq1, eq2), (x, y)))

# 05 - division
def division():
    print("5A - integer_division\n5B - polynomial_division")
    choice = input().lower()
    if choice == 'a':
        a = int(input("a = "))
        b = int(input("b = "))
        print("Integer division result:", a // b)
    elif choice == 'b':
        P = sp.sympify(input("P(x) = "))
        Q = sp.sympify(input("Q(x) = "))
        quotient, remainder = sp.div(P, Q)
        print("Quotient:", quotient)
        print("Remainder:", remainder)

# 06 - divisors
def divisors():
    n = int(input("n = "))
    print("Number of divisors:", sp.divisor_count(n))

# 07 - divisors_list
def divisors_list():
    n = int(input("n = "))
    print("Divisors:", sp.divisors(n))

# 08 - sigma
def sigma():
    n = int(input("n = "))
    print("Sum of divisors:", sp.divisor_sigma(n))

# 09 - vector_dot_product
def vector_dot_product():
    u = [float(num) for num in input("u = ").split(',')]
    v = [float(num) for num in input("v = ").split(',')]
    u = sp.Matrix(u)
    v = sp.Matrix(v)
    print("Dot product:", u.dot(v))

# 10 - expand
def expand():
    expr = sp.sympify(input("Expression: "))
    print("Expanded:", sp.expand(expr))

# 11 - factorise
def factorise():
    expr = sp.sympify(input("Expression: "))
    print("Factorised:", sp.factor(expr))

# 12 - from_base
def from_base():
    n = input("Number in base: ")
    base = int(input("Base: "))
    print("Base 10:", int(n, base))

# 13 - gcd
def gcd():
    nums = [int(num) for num in input("Numbers separated by comma: ").split(',')]
    print("GCD:", sp.gcd(nums))

# 14 - irrational_factor
def irrational_factor():
    expr = sp.sympify(input("Expression: "))
    factored = factor(expr, extension=QQ.algebraic_field())
    print("Factored (irrational):", factored)

# 15 - is_factored
def is_factored():
    expr = sp.sympify(input("Expression: "))
    factored_expr = sp.factor(expr)
    print(expr == factored_expr)

# 16 - is_prime
def is_prime():
    n = int(input("n = "))
    print("Is prime:", sp.isprime(n))

# 17 - lcm
def lcm():
    nums = [int(num) for num in input("Numbers separated by comma: ").split(',')]
    print("LCM:", sp.lcm(nums))

# 18 - left_side
def left_side():
    print("UNDER DEVELOPMENT")

# 19 - maximum
def maximum():
    nums = [float(num) for num in input("Numbers separated by comma: ").split(',')]
    print("Maximum:", max(nums))

# 20 - minimum
def minimum():
    nums = [float(num) for num in input("Numbers separated by comma: ").split(',')]
    print("Minimum:", min(nums))

# 21 - mod
def mod():
    print("21A - mod_number\n21B - mod_polynomial")
    choice = input().lower()
    if choice == 'a':
        a = int(input("a = "))
        n = int(input("n = "))
        print("Modulo:", a % n)
    elif choice == 'b':
        P = sp.sympify(input("P(x) = "))
        Q = sp.sympify(input("Q(x) = "))
        print("Remainder:", sp.rem(P, Q))

# 22 - next_prime
def next_prime():
    n = int(input("n = "))
    print("Next prime:", sp.nextprime(n))

# 23 - nsolutions
def n_solutions():
    eq_l = sp.sympify(input("Left side: "))
    eq_r = sp.sympify(input("Right side: "))
    eq = sp.Eq(eq_l, eq_r)
    start = input("Starting value (optional): ").strip()
    try:
        if start:
            start = sp.sympify(start)
            print("Solution:", sp.nsolve(eq, x, start))
        else:
            print("Solution:", sp.nsolve(eq, x))
    except Exception as e:
        print("Error:", e)

# 24 - previous_prime
def previous_prime():
    n = int(input("n = "))
    print("Previous prime:", sp.prevprime(n))

# 25 - prime_factors
def prime_factors():
    n = int(input("n = "))
    print("Prime factors:", sp.factorint(n))

# 26 - right_side
def right_side():
    print("UNDER DEVELOPMENT")

# 27 - simplify
def simplify():
    expr = sp.sympify(input("Expression: "))
    print("Simplified:", sp.simplify(expr))
    
# 28 - solve
def solve():
    def solve_equation():
        l = input()
        r = input()
        l = sp.sympify(l)
        r = sp.sympify(r)
        print(sp.solve(sp.Eq(l, r)))
    def solve_inequality():
        expression = input()
        expression = sp.sympify(expression)
        print(sp.solve_univariate_inequality(expression, x))
    print("28A - solve_equation\n28B - solve_inequality")
    choice = input().lower()
    if choice == 'a':
        solve_equation()
    elif choice == 'b':
        solve_inequality()

# 29 - to_base
def to_base():
    n = int(input("n = "))
    b = int(input("base = "))
    digits = []
    num = n
    while num > 0:
        digits.append(num % b)
        num //= b
    print("Digits in base", b, ":", digits[::-1] if digits else [0])

# Menu list
menu = ['01 - common_denominator', '02 - complete_square', '03 - vector_cross_product', '04 - imaginary_solutions', '05 - division',
    '06 - divisors', '07 - divisors_list', '08 - sigma', '09 - vector_dot_product', '10 - expand', '11 - factorise', '12 - from_base',
    '13 - gcd', '14 - irrational_factor [UNDER DEVELOPMENT]', '15 - is_factored', '16 - is_prime', '17 - lcm', '18 - left_side [UNDER DEVELOPMENT]',
    '19 - max', '20 - min', '21 - mod', '22 - next_prime', '23 - nsolutions', '24 - previous_prime', '25 - prime_factors',
    '26 - right_side', '27 - simplify', '28 - solve', '29 - to_base'
]

# Main loop
funcs = [
    common_denominator, complete_square, vector_cross_product, imaginary_solutions, division,
    divisors, divisors_list, sigma, vector_dot_product, expand, factorise, from_base,
    gcd, irrational_factor, is_factored, is_prime, lcm, left_side, maximum, minimum,
    mod, next_prime, n_solutions, previous_prime, prime_factors, right_side, simplify, solve, to_base
]

while True:
    print("=== PyAlgebra Menu ===")
    # print items in landscape format
    for i, item in enumerate(menu, start=1):
        print(f"{item:35}", end="")  # width = 35 chars for alignment
        if i % 3 == 0:               # wrap every 3 items
            print()
    print()  # final newline

    choice = input("Select option: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(funcs):
        funcs[int(choice)-1]()
    else:
        print("Invalid option. Enter a number between 1 and", len(funcs))