<p align="center">
  <img src="haarpy.svg">
</p>

<div align="center">

  <a href="https://pypi.org/project/haarpy">
    <img src="https://github.com/polyquantique/haarpy/actions/workflows/tests.yml/badge.svg" alt="Test"/>
  </a>

  <a href="https://codecov.io/gh/polyquantique/haarpy" > 
   <img src="https://codecov.io/gh/polyquantique/haarpy/graph/badge.svg?token=VYWOCW165M"/> 
  </a>

  <a href="https://pypi.org/project/haarpy">
    <img src="https://img.shields.io/pypi/pyversions/haarpy.svg?style=flat" alt="Python Versions"/>
  </a>

  <a href="https://pypi.python.org/pypi/haarpy">
    <img src="https://img.shields.io/pypi/v/haarpy.svg" alt="PyPI version"/>
  </a>

</div>

<br>

Haarpy is a Python library for the symbolic calculation of [Weingarten functions](https://en.wikipedia.org/wiki/Weingarten_function) and related averages of ensembles of unitary matrices under the [Haar measure](https://pennylane.ai/qml/demos/tutorial_haar_measure): these include the [classical compact groups](https://arxiv.org/abs/math-ph/0609050) namely the orthogonal, unitary and (complex-)symplectic groups, as well as the circular orthogonal and symplectic orthogonal ensembles. 


## Haarpy in action
To introduce the main functionality of Haarpy consider the following problem: imagine that you can generate unitary matrices at random, you would like to estimate the average of $|U_{i,j}|^2$ which we write mathematically as $\int dU |U_{i,j}|^2 = \int dU U_{i,j} U_{i,j}^*$. We could obtain this average by using the random matrix functionality of SciPy as follows:
```Python
import numpy as np
from scipy.stats import unitary_group

np.random.seed(137)

def average_mod_single_elem(dim, shots=10000):
    """Estimate the average of the modulus squared of a single element of a Haar random unitary matrix
    
    Args:
        d (int): size of the matrix
        shots (ind): number of shots

    Returns:
        (float): average
    """
    return np.mean([np.abs(unitary_group.rvs(dim = dim)[0,1])**2 for _ in range(shots)])


```
and use it to obtain values for different sizes of the unitary matrices
```
np.array([average_mod_single_elem(i) for i in range(2, 5)]) # Output: array([0.50464014, 0.33273422, 0.25036507])
```

Haarpy allows you to obtain this (and many other!) overages analytically. We first recall that the expression we are trying to calculate is $\int dU |U_{i,j}|^2 = \int dU U_{i,j} U_{i,j}^*$. With this expression in mind we can use Sympy to create a symbolic variable $d$ for the dimension of the unitary and simply write

```Python
from sympy import Symbol
from haarpy import haar_integral_unitary

d = Symbol("d")
haar_integral_unitary(("i","j","i","j"),d)   # Output: 1/d
```
Notice the order of the indices! The first `"i"` and `"j"` are the indices of $U$ while the second pair of `"i"` and `"j"` are the indices of $U^*$.

Imagine that now we want to calculate something like $\int dU U_{i,m} U_{j,n} U_{k,o} U_{i,m}^* U_{j,n}^* U_{k,p}^* = \int dU |U_{i,j,k}U_{m,n,o}|^2$ we simply do
```
haar_integral_unitary(("ijk","mno","ijk","mno"), d) # Output: (d**2 - 2)/(d*(d - 2)*(d - 1)*(d + 1)*(d + 2))
```


The main functions of Haarpy are *weingarten_class*, *weingarten_element* and *haar_integral* allowing for the calculation of Weingarten functions and integrals over unitaries sampled at random from the Haar measure. We recommend importing the following when working with Haarpy:
```Python
from sympy import Symbol
from sympy.combinatorics import Permutation

d = Symbol("d")
```
### *weingarten_class*
Takes a partition, labeling a conjugacy class of $S_p$, and a dimension $d$ as arguments. For the conjugacy class labeled by partition $\lbrace 3,1\rbrace$, the function returns
```Python
from haarpy import weingarten_class
weingarten_class((3,1),d)
```
```
(2*d**2 - 3)/(d**2*(d - 3)*(d - 2)*(d - 1)*(d + 1)*(d + 2)*(d + 3))
```
The previous can also be called with integer values as such
```Python
weingarten_class((3,1),4)
```
```
29/20160
```
### *weingarten_element*
Takes an element and the degree $p$ of the symmetric group $S_p,$ and a dimension $d$ as arguments, the conjugacy class being obtained from the first two.
```Python
from haarpy import weingarten_element
weingarten_element(Permutation(0,1,2), 4, d)
```
```
(2*d**2 - 3)/(d**2*(d - 3)*(d - 2)*(d - 1)*(d + 1)*(d + 2)*(d + 3))
```
Which yields the same result as before since $\lbrace 3,1\rbrace$ is the class of permutation $(0,1,2)$ in $S_4$.

### *haar_integral*
Takes in a tuple of sequences $((i_1,\dots,i_p),\ (j_1,\dots,j_p),\ (i\prime_1,\dots, i\prime_p),\ (j\prime_1,\dots,j\prime_p))$, and the dimension $d$ of the unitary group. Returns the value of the integral $\int dU \ U_{i_1j_1}\dots U_{i_pj_p}U^\ast_{i\prime_1 j\prime_1}\dots U^\ast_{i\prime_p j\prime_p}$.
```Python
from haarpy import haar_integral
haar_integral(((1,2), (1,2), (1,2), (1,2)), d)
```
```
1/((d-1)*(d+1))
```
Auxiliary functions include, but are not limited to, the following. For a comprehensive list of functionalities, please refer to the [documentation]().

### *murn_naka_rule*
Implementation of the [Murnaghan-Nakayama rule](https://en.wikipedia.org/wiki/Murnaghan%E2%80%93Nakayama_rule) for the characters irreducible representations of the symmetric group $S_p$. Takes a partition characterizing an irrep of $S_p$ and a conjugacy class and yields the associate character.
```Python
from haarpy import murn_naka_rule
murn_naka_rule((3,1), (1,1,1,1))
```
```
3
```
### *get_conjugacy_class*
Returns the class of a given element of $S_p$ when given the order and the element.
```Python
from haarpy import get_conjugacy_class
get_conjugacy_class(Permutation(0,1,2), 4)
```
```
(3,1)
```
### *irrep_dimension*
Takes a partition labeling an irrep of $S_p$ and returns the dimension of this irrep.
```Python
from haarpy import irrep_dimension
irrep_dimension((5,4,2,1,1,1))
```
```
63063
```
### *representation_dimension*
Takes a partition labeling a representation of the unitary group $U(d)$, as well as the dimension $d$, and returns the dimension of the representation.
```Python
from haarpy import representation_dimension
representation_dimension((5, 4, 2, 1, 1, 1),d)
```
```
d**2*(d - 5)*(d - 4)*(d - 3)*(d - 2)*(d - 1)**2*(d + 1)**2*(d + 2)**2*(d + 3)*(d + 4)/1382400
```
Which can also be done numerically.
```Python
ud_dimension((5, 4, 2, 1, 1, 1),8)
```
```
873180
```

## Tables of Weingarten functions for $n \le 5$
The following have been retrieved using the *weingarten_class* function. Weingarten functions of symmetric groups of higher degrees can just as easily be obtained.

### Symmetric group $S_2$
|Class| Weingarten |
|--|--|
|$\lbrace2\rbrace$ |$-\displaystyle\frac{1}{(d-1) d (d+1)}$|
| $\lbrace1,1\rbrace$ | $\displaystyle\frac{1}{(d-1)(d+1)}$ |

### Symmetric group $S_3$
|Class  | Weingarten |
|--|--|
|  $\lbrace 3\rbrace$ | $\displaystyle\frac{2}{(d-2) (d-1) d (d+1) (d+2)}$ |
|$\lbrace 2,1\rbrace$|$-\displaystyle\frac{1}{(d-2) (d-1) (d+1) (d+2)}$|
| $\lbrace 1,1,1\rbrace$  |$\displaystyle\frac{d^2-2}{(d-2) (d-1) d (d+1) (d+2)}$ |

### Symmetric group $S_4$
|Class| Weingarten |
|--|--|
| $\lbrace 4\rbrace$   |$-\displaystyle\frac{5}{(d-3) (d-2) (d-1) d (d+1) (d+2) (d+3)}$  |
|$\lbrace 3,1\rbrace$ |$\displaystyle\frac{2 d^2-3}{(d-3) (d-2) (d-1) d^2 (d+1) (d+2) (d+3)}$ |
|$\lbrace 2,2\rbrace$|$\displaystyle\frac{d^2+6}{(d-3) (d-2) (d-1) d^2 (d+1) (d+2) (d+3)}$|
| $\lbrace 2,1,1\rbrace$| $-\displaystyle\frac{1}{(d-3) (d-1) d (d+1) (d+3)}$|
|$\lbrace 1,1,1,1\rbrace$ |$\displaystyle\frac{d^4-8 d^2+6}{(d-3) (d-2) (d-1)d^2 (d+1) (d+2)(d+3)}$|

### Symmetric group $S_5$
|Class| Weingarten |
|--|--|
|  $\lbrace 5\rbrace$    |$\displaystyle\frac{14}{(d-4) (d-3) (d-2) (d-1) d (d+1) (d+2) (d+3)(d+4)}$  |
|$\lbrace 4,1\rbrace$  |$\displaystyle\frac{24-5 d^2}{(d-4) (d-3) (d-2) (d-1) d^2 (d+1) (d+2)(d+3)(d+4)}$ |
| $\lbrace 3,2\rbrace$|$-\displaystyle\frac{2 \left(d^2+12\right)}{(d-4) (d-3) (d-2) (d-1)d^2(d+1)(d+2)(d+3)(d+4)}$|
|  $\lbrace 3,1,1\rbrace$ | $\displaystyle\frac{2}{(d-4) (d-2) (d-1) d (d+1) (d+2) (d+4)}$|
| $\lbrace 2,2,1\rbrace$  |$\displaystyle\frac{-d^4+14 d^2-24}{(d-4) (d-3) (d-2) (d-1) d^2  (d+1) (d+2) (d+3) (d+4)}$|
| $\lbrace 1,1,1,1,1\rbrace$|$\displaystyle\frac{d^4-20 d^2+78}{(d-4) (d-3) (d-2) (d-1) d (d+1) (d+2) (d+3) (d+4)}$|

## Examples of integrals over Haar-random unitaries
Selected integrals of unitary groups ; $i,j,k$ and $\ell$ are assumed to take distinct integer values in the following.
|Integral| Result |
|--|--|
| $\int dU \ U_{ij}U^\ast_{ij}$ |$\displaystyle\frac{1}{d}$|
| $\int dU \ U_{ij}U_{kj}U^\ast_{ij}U^\ast_{kj}$ | $\displaystyle\frac{1}{d(d+1)}$|
| $\int dU \ U_{ik}U_{k\ell}U^\ast_{ij}U^\ast_{k\ell}$ |$\displaystyle\frac{1}{(d-1)(d+1)}$|
| $\int dU \ U_{ij}U_{k\ell}U^\ast_{i\ell}U^\ast_{kj}$ | $\displaystyle\frac{-1}{(d-1)d(d+1)}$ |
| $\int dU \ U_{ij}U_{k\ell}U_{mn}U^\ast_{ij}U^\ast_{k\ell}U^\ast_{mn}$ | $\displaystyle\frac{d^2-2}{(d-2)(d-1)d(d+1)(d+2)}$ |
| $\int dU \ U_{i\ell}U_{jm}U_{kn}U^\ast_{im}U^\ast_{jn}U^\ast_{k\ell}$ | $\displaystyle\frac{2}{(d-2)(d-1)d(d+1)(d+2)}$ |
| $\int dU \ U_{i\ell}U_{j\ell}U_{km}U^\ast_{i\ell}U^\ast_{jm}U^\ast_{k\ell}$ | $\displaystyle\frac{-1}{(d-1)d(d+1)(d+2)}$ |
| $\int dU  \ U_{i\ell}U_{j\ell}U_{k\ell}U^\ast_{i\ell}U^\ast_{j\ell}U^\ast_{k\ell}$ | $\displaystyle\frac{1}{d(d+1)(d+2)}$ |
| $\int dU \ U_{ij}U_{ik}U_{i\ell}U_{im}U^\ast_{ij}U^\ast_{ik}U^\ast_{i\ell}U^\ast_{im}$ | $\displaystyle\frac{1}{d(d + 1)(d + 2)(d + 3)}$ |
## Installation
Haarpy requires Python version 3.9 or later. Installation can be done through the pip command
```
pip install haarpy
```

## Compiling from source
Haarpy has the following dependencies:
* [Python](https://www.python.org/) >= 3.9
* [SymPy](https://www.sympy.org) >= 1.12


## Documentation
Haarpy documentation is available online on [Read the Docs]().


## How to cite this work
Please cite as:
```
@misc{cardin2024haarpy,
  author={Cardin, Yanic and de Guise, Hubert and Quesada, Nicol{\'a}s},
  title={Haarpy, a Python library for the symbolic calculation of Weingarten functions},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished = {\url{https://github.com/polyquantique/haarpy}},
  version = {0.0.5}
}
```


## Authors
* Yanic Cardin, Hubert de Guise, Nicolás Quesada.


## License
Haarpy is free and open source, released under the Apache License, Version 2.0.
