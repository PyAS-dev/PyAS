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

    1T
    -1 Algebra
    ✅--1A Tallregning
    --1B Tallmengder
    ✅--1C Potensregning
    --1D Store og små tall
    ✅--1E Bokstavuttrykk
    ✅--1F Kvadratsetningene
    ✅--1G Faktorisering
    ✅--1H Brøkregning
    --1I n-te røtter
    ✅--1J Rotregning
    -2 Likninger
    ✅--2A Førstegradslikninger
    ✅--2B Fra tekst til likning
    ✅--2C Forhold
    ✅--2D Andregradslikninger
    ✅--2E Nullpunktsfaktorisering
    --2F Formler
    --2G Logaritmelikninger
    --2H Eksponentiallikninger
    ✅--2I Potenslikninger
    --2J Prosentregning med veksfaktor
    -3 Funksjoner
    ✅--3A Funksjonsbegrepet
    ✅--3B Lineære funksjoner
    --3C Praktisk bruk av lineære funksjoner
    --3D Lineær regresjon
    ✅--3E Polynomfunksjoner
    --3F Eksponentialfunksjoner
    --3G Rasjonale funksjoner
    ✅--3H Potensfunksjoner og Rotfunksjoner
    --3I Kombinasjoner av funksjoner
    -4 Likningssystemer og ulikheter
    ✅--4A Lineære likningssystemer
    ✅--4B Ikke-lineære likningssystemer
    --4C Førstegradsulikheter
    --4D Andregradsulikheter
    -5 Derivasjon
    --5A Gjennomsnittlig vekstfart
    --5B Momentan vekstfart
    --5C Derivasjon
    --5D Derivasjonsregler
    --5E Fortegnet for den deriverte
    --5F Funksjonsdrøfting
    --5G Praktisk bruk av derivasjon
    --5H Definisjonen av den deriverte
    -6 Geometri
    ✅--6A Pytagorassetningen
    ✅--6B Formlikhet
    --6C Tangens
    ✅--6D Sinus og cosinus
    --6E Enhetssirkelen
    --6F Arealsetningen
    --6G Sinussetningen
    --6H Cosinussetningen
    -7 Sannsynlighet
    --7A Sannsynlighet og relativ frekvens
    --7B Sannsynlighetsmodell
    --7C Uniforme sannsynlighetsmodeller
    --7D Addisjonssetningen
    --7E Produktsetningen for uavhengige hendelser
    --7F Produktsetningen for avhengige hendelser
    --7G Sammensatte forsøk
    R1
    -1 Algebra
    ✅--1A Faktorisering
    ✅--1B Polynomdivisjon
    --1C Implikajson og ekvivalens
    ✅--1D Likninger
    --1E Ulikheter
    --1F Bevistyper
    -2 Logaritmer
    ✅--2A Potenser
    ✅--2B Logaritmedefinisjoner
    ✅--2C Logaritmesetningene
    ✅--2D Logaritmelikninger
    ✅--2E Eksponentiallikninger
    --2F Ulikheter
    -3 Funksjoner
    ✅--3A Funksjonsbegrepet
    --3B Grenseverdier
    --3C Kontinuitet
    ✅--3D Rasjonale funksjoner
    ✅--3E Eksponentialfunksjoner
    ✅--3F Logaritmefunksjoner
    --3G Deriverbarhet
    -4 Funksjonsdrøfting
    --4A Derivasjonsregler
    --4B Tangenter
    --4C Størst og minst
    --4D Kjerneregelen
    --4E Produktregelen
    --4F Brøkregelen
    --4G Den andrederiverte
    -5 Geometri
    --5A Geometrisk sted
    --5B Periferivinkler og sentralvinkler
    ✅--5C Formlikhet og kongruens
    --5D Analyse og konstruksjon
    --5E Skjæringssetninger
    --5F Å bevise pytagorassetningen
    -6 Vektorer
    ✅--6A Hva er en vektor?
    ✅--6B Addisjon og subtraksjon av vektorer
    --6C Parallelle vektorer
    ✅--6D Mer om vektorkoordinater
    --6E Sirkellikningen
    ✅--6F Skalarproduktet
    ✅--6G Vektorer og geometri
    --6H Parameterframstilling
    --6I Vektorfunksjoner
    -7 Sannsynlighet
    ✅--7A Betinget sannsynlighet og uavhengige hendelser
    ✅--7B Produktsetningen
    --7C Total sannsynlighet og Bayes' setning
    ✅--7D Kombinatorikk
    ✅--7E Ordnet utvalg uten tilbakelegging
    ✅--7F Urdnet utvalg uten tilbakelegging
    --7G Hypergeometriske sannsynligheter
    ✅--7H Mer kombinatorikk
    --7I Binomiske sannsynligheter
    R2
    -1 Integrasjon
    --1A Bestemt integral
    --1B Ubestemt integral
    --1C Bestemt integral ved antiderivasjon
    --1D Volum ved integrasjon
    --1E Delvis integrasjon
    --1F Variabelskifte
    --1G Delbrøkoppspalting
    --1H Hvilken metode?
    -2 Trigonometri
    --2A Vinkelmål
    --2B Enhetssirkelen
    --2C Trigonometriske grunnlikninger
    --2D Enhetsformelen
    --2E Trigonometriske likninger
    --2F Sum av og differanse mellom vinkler
    -3 Funksjoner
    --3A Trigonometriske funksjoner
    --3B Harmoniske svinginger
    --3C Omforming til sinus
    --3D Derivasjon av trigonometriske funksjoner
    --3E Funksjonsdrøfting
    --3F Integrasjon av trigonometriske funksjoner
    --3G Modellering
    -4 Tredimensjonale vektorer
    --4A Fra 2D til 3D
    ✅--4B Skalarproduktet
    --4C Geometrisk representasjon
    --4D Linjer i rommet
    --4E Vektorproduktet
    --4F Areal og volum
    -5 Romgeometri
    --5A Plan
    --5B Plan og linjer
    --5C Avstand til linje
    --5D Avstand til plan
    --5E Kuleflater
    --5F Parameterframstilling for plan og kuleflate
    -6 Differensiallikninger
    --6A En ny type likninger
    --6B Seperable differensiallikninger
    --6C Lineære differensiallikninger av første orden
    --6D Integralkurver
    --6E Praktisk bruk av differensiallikninger
    --6F Differensiallikninger av andre orden
    --6G Frie svinginger
    -7 Følger og rekker
    ✅--7A Følger
    ✅--7B Rekker
    ✅--7C Induksjonsbevis
    ✅--7D Aritmetiske rekker
    ✅--7E Geometriske rekker
    --7F Uendelige geometriske rekker

- Optionally `mpmath` for arbitrary-presicion arithmetic

### Install SymPy
```bash
pip install sympy
