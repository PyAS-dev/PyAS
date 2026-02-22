version = "1.0.0"
import inspect
import sympy as sp
import math
import pyas_math
import pyas_algebra
import pyas_calculus
import pyas_linear_algebra
import ast
import ast
from sympy.parsing.sympy_parser import parse_expr

# -----------------------------
# Auto-register FUNCTIONS from any module
# -----------------------------
FUNCTIONS = {}

def register_module_FUNCTIONS(module):
    for name, object in inspect.getmembers(module):
        if inspect.isfunction(object) and not name.startswith("_"):
            FUNCTIONS[name] = object

# register FUNCTIONS from both modules
register_module_FUNCTIONS(pyas_algebra)
register_module_FUNCTIONS(pyas_math)
register_module_FUNCTIONS(pyas_calculus)
register_module_FUNCTIONS(pyas_linear_algebra)

# -----------------------------
# Parser: converts string to functiontion + arguments
# -----------------------------

def split_arguments(argument_string):
    arguments = []
    current = ""
    depth = 0
    for c in argument_string:
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            arguments.append(current.strip())
            current = ""
            continue
        current += c
    if current:
        arguments.append(current.strip())
    return arguments

def parse_call(expression: str):
    expression = expression.strip()
    if "(" not in expression or not expression.endswith(")"):
        raise ValueError("Invalid syntax. Use f(arguments)")

    function_name, argument_string = expression.split("(", 1)
    function_name = function_name.strip()
    argument_string = argument_string[:-1]

    if not argument_string:
        return function_name, []

    arguments = []
    for a in split_arguments(argument_string):
        try:
            value = ast.literal_eval(a)  # parse Python literals
        except:
            value = parse_expr(a)        # fallback: SymPy expressions
        arguments.append(value)
    return function_name, arguments

# -----------------------------
# Executor: call FUNCTIONS dynamically
# -----------------------------
def execute(expression: str):
    function_name, arguments = parse_call(expr)
    if function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function: {function_name}")
    return FUNCTIONS[function_name](*arguments)

# -----------------------------
# REPL
# -----------------------------
def repl():
    print(f"PyAS v{version}")
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
