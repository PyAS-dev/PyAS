version = "0.11.7"
import inspect
import sympy as sp
import math
import pyas_math
import pyas_algebra
import pyas_calculus
import pyas_linear_algebra
import ast
from sympy.parsing.sympy_parser import parse_expr

# -----------------------------
# Auto-register functions from any module
# -----------------------------
FUNCTIONS = {}

def register_module_functions(module):
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and not name.startswith("_"):
            FUNCTIONS[name] = obj

# register functions from both modules
register_module_functions(pyas_algebra)
register_module_functions(pyas_math)
register_module_functions(pyas_calculus)
register_module_functions(pyas_linear_algebra)

# -----------------------------
# Parser: converts string to function + arguments
# -----------------------------

def split_args(arg_str):
    args = []
    current = ""
    depth = 0
    for c in arg_str:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(current.strip())
            current = ""
            continue
        current += c
    if current:
        args.append(current.strip())
    return args

def parse_call(expr: str):
    expr = expr.strip()
    if "(" not in expr or not expr.endswith(")"):
        raise ValueError("Invalid syntax. Use f(args)")

    func_name, arg_str = expr.split("(", 1)
    func_name = func_name.strip()
    arg_str = arg_str[:-1]

    if not arg_str:
        return func_name, []

    args = []
    for a in split_args(arg_str):
        try:
            val = ast.literal_eval(a)  # parse Python literals
        except:
            val = parse_expr(a)        # fallback: SymPy expressions
        args.append(val)
    return func_name, args

# -----------------------------
# Executor: call functions dynamically
# -----------------------------
def execute(expr: str):
    func_name, args = parse_call(expr)
    if func_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {func_name}")
    return FUNCTIONS[func_name](*args)

# -----------------------------
# REPL
# -----------------------------
def repl():
    print(f"PyAS v{version} (type 'exit' to quit)")
    while True:
        user_input = input(">>> ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        try:
            print(execute(user_input))
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    repl()
