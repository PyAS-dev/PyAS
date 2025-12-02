# Python Algebra System (PyAS)

**PyAS** is a modular Python-based algebra and calculus toolkit built on top of [SymPy](https://www.sympy.org/).  
It provides a user-friendly interface for solving equations, performing calculus operations, and evaluating mathematical functions—perfect for students, educators, and developers.

---

## Features

- **Algebra**: Solve equations, factor polynomials, find gcd/lcm, work with complex numbers, and more.  
- **Calculus**: Compute derivatives, integrals, limits, series expansions, and curve analysis.  
- **Functions**: Evaluate trigonometric, hyperbolic, logarithmic, exponential, and special functions.  
- **Systems**: Solve systems of equations and ODEs.  
- **Interactive Menus**: Easy-to-navigate CLI menus for each module.

---

### Version History

# Python Algebra System - Versjonhistorie

---

## 0.0.1 – Alpha Release

### Nytt
- Støtte for førstegradslikninger `[ax+b=0]`:
  - Automatisk visning av desimaltall
- Støtte for andregradslikninger `[ax^2+bx+c=0]`:
  - Løser både reelle og komplekse løsninger
  - Viser løsninger i desimalform
- Interaktiv tekstbasert meny:
  - Valg mellom førstegrad- og andregradslikninger

### Bugs
- **B1:** Programmet kan kræsje ved feil input (f.eks. bokstaver i stedet for tall)

---

## 0.1

### Nytt
- Støtte for brøksvar
  - Automatisk visning av både desimaltall og brøkform
  - Bruk av `fraction.limit_denominator()` for rasjonal brøkform
- Input-validering
  - Avslutter korrekt ved spesielle tilfeller (for eksempel a=0 i førstegrad og andregrad)

---

## 0.1.15 – To Ukjente

### Nytt
- Likningssett med to ukjente (2x2 systemer)
  - Bruker determinanter for å finne løsning
  - Varsler hvis systemet ikke har entydig løsning `[det(A)=0]`

---

## 0.2.2 – Stiloppdateringen

### Nytt
- Automatisk visning av både desimaltall og brøkform (Q-løsning)
- Bruker Unicode til å vise “logisk og” og “logisk eller”

### Bugs
- **B2:** Små formateringsfeil mulig med -0.0 ved utskrift

---

## 0.2.5

### Nytt
- Viser versjonnummer øverst på linje 1

---

## 0.2.11

### Nytt
- Skriver nå andregradsuttrykk også på faktorisert form

---

## 0.5.27 – SymPy oppdateringen

### Nytt
- Full omskriving av koden
- Løser likninger og likningssystem symbolsk istedenfor numerisk
- Bruker `sympy` og `math`-biblioteket
- Andregradsuttrykk ikke lenger skrevet på faktorisert form
- Viser ikke lenger svar i desimalform

---

## 0.6.15 – Matematiske funksjoner

### Nytt
- Nytt modul: Matematiske funksjoner
- Implementert 54 funksjoner fra grunnleggende aritmetikk til avanserte spesialfunksjoner

#### Grunnleggende funksjoner
| Funksjon | Definisjon |
|----------|------------|
| `random(x)` | Genererer et tilfeldig tall mellom 0 og 1 |
| `sqrt(x)` | √x = x^(1/2) |
| `cbrt(x)` | ∛x = x^(1/3) |
| `nroot(x, n)` | n-te rot: √[n]{x} = x^(1/n) |
| `abs(x)` | Absoluttverdi: |x| |
| `sgn(x)` | Tegnfunksjon: 1 hvis x>0, 0 hvis x=0, -1 hvis x<0 |
| `floor(x)` | Gulvfunksjon: største heltall ≤ x |
| `ceil(x)` | Takfunksjon: minste heltall ≥ x |
| `round(x)` | Nærmeste heltall |
| `fractional_part(x)` | Fraksjonsdel: {x} = x - ⌊x⌋ |

#### Logaritmer og eksponentialfunksjoner
| Funksjon | Definisjon |
|----------|------------|
| `log(x, b)` | logaritme base b: log_b(x) |
| `ln(x)` | Naturlig logaritme: log_e(x) |
| `log10(x)` | Logaritme base 10 |
| `log2(x)` | Logaritme base 2 |
| `exp(x)` | Eksponentialfunksjon: e^x |

#### Trigonometriske funksjoner
| Funksjon | Definisjon |
|----------|------------|
| `sin(x)` | sinus |
| `cos(x)` | cosinus |
| `tan(x)` | tangent = sin(x)/cos(x) |
| `sec(x)` | sekans = 1/cos(x) |
| `csc(x)` | kosekans = 1/sin(x) |
| `cot(x)` | kotangent = cos(x)/sin(x) |
| `asin(x)` | arcsin |
| `acos(x)` | arccos |
| `atan(x)` | arctan |
| `atan2(y,x)` | arctan2(y,x) |

#### Hyperbolske funksjoner
| Funksjon | Definisjon |
|----------|------------|
| `sinh(x)` | (e^x - e^-x)/2 |
| `cosh(x)` | (e^x + e^-x)/2 |
| `tanh(x)` | sinh(x)/cosh(x) |
| `sech(x)` | 1/cosh(x) |
| `csch(x)` | 1/sinh(x) |
| `coth(x)` | cosh(x)/sinh(x) |
| `asinh(x)` | arcsinh(x) = ln(x + √(x^2+1)) |
| `acosh(x)` | arccosh(x) = ln(x + √(x^2-1)) |
| `atanh(x)` | arctanh(x) = 1/2 ln((1+x)/(1-x)) |

#### Gamma- og Beta-funksjoner
| Funksjon | Definisjon |
|----------|------------|
| `gamma(x)` | Γ(x) = ∫_0^∞ t^(x-1) e^(-t) dt |
| `lower_incomplete_gamma(a,x)` | γ(a,x) = ∫_0^x t^(a-1) e^(-t) dt |
| `gamma_regularized(a,x)` | P(a,x) = γ(a,x)/Γ(a) |
| `psi(x)` | ψ(x) = (ln(Γ(x)))' |
| `beta(a,b)` | B(a,b) = Γ(a)Γ(b)/Γ(a+b) |
| `incomplete_beta(a,b,x)` | B_x(a,b) = ∫_0^x t^(a-1)(1-t)^(b-1) dt |
| `beta_regularized(a,b,x)` | I_x(a,b) = B_x(a,b)/B(a,b) |

#### Spesialfunksjoner
| Funksjon | Definisjon |
|----------|------------|
| `erf(x)` | Feilfunksjon |
| `sin_integral(x)` | Si(x) = ∫_0^x sin(t)/t dt |
| `cos_integral(x)` | Ci(x) = -∫_x^∞ cos(t)/t dt |
| `exp_integral(x)` | Ei(x) = ∫_-∞^x e^t/t dt |
| `zeta(s)` | ζ(s) = ∑_n=1^∞ 1/n^s |
| `Dirac(x)` | δ(x), null overalt unntatt ∫_-∞^∞ δ(x) dx = 1 |
| `Heaviside(x)` | H(x) = 0 hvis x<0, 1 hvis x≥0 |

#### Kombinatorikk
| Funksjon | Definisjon |
|----------|------------|
| `nPr(n,r)` | Permutasjon nPr = n!/(n-r)! |
| `nCr(n,r)` | Kombinasjon nCr = n!/r!(n-r)! |

#### Komplekse funksjoner
| Funksjon | Definisjon |
|----------|------------|
| `arg(z)` | Argumentet θ av z |
| `conjugate(z)` | Kompleks konjugat: ¯z = a - bi hvis z = a + bi |
| `real(z)` | Reell del av z |
| `imaginary(z)` | Imag del av z |

---

### Forbedringer
- Standardisert input/output med `sympy` for numeriske og symbolske uttrykk
- Støtte for flytende punkt- og sympy-input, inkludert komplekse tall
- Menyvalg støtter tall med og uten ledende nuller (f.eks. 01 = 1)

---

## 0.6.22 – Algebra-modulen

### Nytt
- Nytt modul: Algebra
- Interaktivt konsollverktøy for algebraiske og numeriske operasjoner
- Symbolsk algebra via `SymPy`
- Funksjoner inkludert:
  - `common_denominator`, `complete_square`, `vector_cross_product`, `imaginary_solutions`, `division`, `divisors`, `divisors_list`, `sigma`
  - `vector_dot_product`, `expand`, `factorise`, `from_base`, `gcd`, `irrational_factor`, `is_factored`
  - `is_prime`, `lcm`, `left_side`, `maximum`, `minimum`, `mod`, `next_prime`, `nsolutions`
  - `previous_prime`, `prime_factors`, `right_side`, `simplify`, `to_base`

### Forbedringer
- Interaktiv meny med tre kolonner
- Støtte for både numeriske og symbolske beregninger
- Unngår avhengighet av eksterne biblioteker utover `sympy`

### Notater
- `left_side` og `right_side` under utvikling
- Kan brukes direkte fra terminalen via menyvalg

---

## 0.9.3 – Kalkulus-modulen

### Nytt
- Nytt modul: Kalkulus
- Konsollverktøy for symbolske og numeriske operasjoner
- Symbolsk analyse via `SymPy`
- Funksjoner inkludert: `asymptote`, `bezier_curve`, `coefficients`, `complex_roots`, `curvature`, `curvature_vector`, `parametric_curve`, `degree`, `denominator`, `derivative`, `factors`, `implicit_derivative`, `inflection_point`, `integral`, `integral_between`, `is_vertex_form`, `iteration`, `iteration_list`, `left_sum`, `limit`, `limit_above`, `limit_below`, `lower_sum`, `n_integral`, `normalize`, `n_solve_ode`, `numerator`, `osculating_circle`, `parametric_derivative`, `partial_fractions`, `path_parameter`, `polynomial`, `rectangle_sum`, `removable_discontinuity`, `root`, `root_list`, `solve_ode`, `spline`, `svd`, `taylor_polynomial`, `trapezoid_sum`, `trig_combine`, `trig_expand`, `trig_simplify`, `turning_point`, `upper_sum`

### Notater
- `bezier_curve` fungerer foreløpig ikke

## Modules

| Module | Description |
|--------|-------------|
| `PyAS.py` | Basic equation solving (1 or 2 unknowns) |
| `PyAS [ALGEBRA].py` | Algebraic operations: factor, expand, solve, modular arithmatic, primes, etc. |
| `PyAS [CALCULUS].py` | Calculus toolkit: derivatives, integrals, limits, series, curve analysis |
| `PyAS [MATHEMATICAL FUNCTIONS].py` | Evaluate mathematical functions: trigonometric, logarithms, gamma functions, error function, etc. |

---

## Installation

### Prerequisites
- Python 3.7+
- `sympy` library

### Install SymPy
```bash
pip install sympy
