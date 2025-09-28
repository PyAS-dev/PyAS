# PyAS v0.5.27
import sympy as sp
import math

x, y = sp.symbols('x, y')

def PyAs():
    l = str(input())
    r = str(input())
    l = sp.sympify(l)
    r = sp.sympify(r)
    expr = sp.Eq(l, r)
    print(sp.solveset(expr))
    
def system_of_equations():
    a = str(input("a = "))
    b = str(input("b = "))
    c = str(input("c = "))
    d = str(input("d = "))
    e = str(input("e = "))
    f = str(input("f = "))
    
    a = sp.sympify(a)
    b = sp.sympify(b)
    c = sp.sympify(c)
    d = sp.sympify(d)
    e = sp.sympify(e)
    f = sp.sympify(f)
    
    solution = sp.linsolve([a * x + b * y - c, d * x + e * y - f], (x, y))
    print(solution)
# ---------- Hovedmeny ----------
meny = [
    '1 - En ukjent',
    '2 - To ukjente'
]
 
while True:
    print()
    for alternativ in meny:
        print(alternativ)
    valg = input() 
    if valg == '1':
        PyAs()
    elif valg == '2':
        system_of_equations()