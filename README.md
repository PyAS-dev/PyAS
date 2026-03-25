# Python Algebra System (PyAS)

**PyAS** is a modular Python-based algebra and calculus toolkit built on top of [SymPy](https://www.sympy.org/).  
It provides a user-friendly interface for solving equations, performing calculus operations, and evaluating mathematical functions, perfect for students, educators, and developers.

---

## Features

- **Algebra**: Solve equations, factor polynomials, find gcd/lcm, work with complex numbers, and more.  
- **Calculus**: Compute derivatives, integrals, limits, series expansions, and curve analysis.  
- **Functions**: Evaluate trigonometric, hyperbolic, logarithmic, exponential, and special functions.  
- **Systems**: Solve systems of equations and ODEs.
- **Linear Algebra**: Work with matrices, vectors and compute determinants, inverses, rref, rank and more.

---

## Modules

| Module | Description |
|--------|-------------|
| `pyas_core.py` | Core module which is contains the command line |
| `pyas_algebra.py` | Algebraic operations: factor, expand, solve, modular arithmetic and primes |
| `pyas_calculus.py` | Calculus toolkit: derivatives, integrals, limits, series and curve analysis |
| `pyas_math.py` | Evaluate mathematical functions: trigonometric, logarithms, gamma functions and error function |
| `pyas_linear_algebra.py` | Work with matrix and vector operations such as determinant, inverse, rank and rref |

---

## Installation

### Prerequisites
- Python 3.7+ as the program is written in python
- `sympy` for symbolic computation
- Optionally `mpmath` for arbitrary-presicion arithmetic

### Install SymPy
```bash
pip install sympy
