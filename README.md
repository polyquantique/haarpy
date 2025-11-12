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
shots = 1000

# For unitary matrices of size between 2 and 4 produce 1000 shots and calculate the average 
# value of the absolute value squared of the 0,1 entry
np.array(
    [
        np.mean([np.abs(unitary_group.rvs(dim=dim)[0, 1]) ** 2 for _ in range(shots)])
        for dim in range(2, 5)
    ]
)
# Output: array([0.4964599 , 0.32742463, 0.25429793])
```

Haarpy allows you to obtain this (and many other!) overages analytically. We first recall that the expression we are trying to calculate is $\int dU |U_{i,j}|^2 = \int dU U_{i,j} U_{i,j}^*$. With this expression in mind we can use Sympy to create a symbolic variable $d$ for the dimension of the unitary and write

```Python
from sympy import Symbol
from haarpy import haar_integral_unitary

d = Symbol("d")
haar_integral_unitary(("i","j","i","j"),d)   
# Output: 1/d
```
Notice the order of the indices! The first `"i"` and `"j"` are the indices of $U$ while the second pair of `"i"` and `"j"` are the indices of $U^*$.

Imagine that now we want to calculate something like $\int dU U_{i,m} U_{j,n} U_{k,o} U_{i,m}^* U_{j,n}^* U_{k,p}^*$ $= \int dU |U_{i,m} U_{j,n} U_{k,o}|^2$ we simply do
```
haar_integral_unitary(("ijk","mno","ijk","mno"), d) 
# Output: (d**2 - 2)/(d*(d - 2)*(d - 1)*(d + 1)*(d + 2))
```
The averages we are calculating are obtained by using so-called Weingarten calculus. Weingarten functions depend only on a class of symmetric group $S_p$ and on the dimension $d$ of the unitaries that are averaged.  A convenient closed form expression for averages of unitary matrices is given by

$$
\int dU \ U_{i_1j_1}\ldots U_{i_pj_p} \left(U_{i^\prime_1j^\prime_1}\ldots U_{i^\prime_p,j^\prime_p}\right)^{\ast}   =\sum_{\sigma,\tau\in S_p}\text{Wg}_U([\sigma\tau^{-1}];d)\, \tag{1} 
$$

where $\text{Wg}_U([\sigma\tau^{-1}];d)$ is the unitary Weingarten function,  $U$ is a Haar-random $d\times d$ unitary matrix, $dU$ is the Haar measure over $U(d)$, and  $[\sigma]$ is the class of element $\sigma$.  The sum in Eq. (1) is a sum over all $\sigma\in S_p$ and all the $\tau\in S_p$ so that

$$   (i^\prime_{\sigma(1)},\ldots,i^\prime_{\sigma(p)})=(i_1,\ldots,i_p)\, \\
(j^\prime_{\tau(1)},\ldots,j^\prime_{\tau(p)})=(j_1,\ldots,j_p)\, 
$$

with the integral $0$ if the $i',i$, $j'$ or $j$ strings have different lengths.  In other words, expectation of polynomials of entries of unitary matrices are given by a sum of Weingarten functions. 

One can access directly the Weingarten functions by calling
```Python
from haarpy import weingarten_unitary
weingarten_unitary((1,2,3),d)
# Output: (-2*d**2 - 13)/(d*(d - 5)*(d - 4)*(d - 2)*(d - 1)**2*(d + 1)**2*(d + 2)*(d + 4)*(d + 5))
```
Here the tuples `(1,2,3)` are used to represent an element of the symmetric group.



## Haarpy functionality

Haarpy implements Weingarten functions for all the classical compact groups, the circular orthogonal ensembles as well as the permutation and centered permutation groups

### Unitary group
Unitary matrices are complex-matrices $U$ that satisfy $U U^\dagger = I_d$ where $I_d$ is the identity matrices.
One can calculate averages over the unitary Haar measure using `haar_integral_unitary` and obtain their associated Weingarten function using `weingarten_unitary`. The later function takes as input a permutation, specified either by a tuple or SymPy `Permutation` object.

### Orthogonal group
Unitary matrices are real-matrices $O$ that satisfy $O O^T = I_d$ where $I_d$ is the identity matrices. One can calculate averages over the orthogonal Haar measure using `haar_integral_unitary` and obtain their associated Weingarten function using `weingarten_orthogonal`. The later function takes as input a ????


### Unitary-Symplectic group
### Circular Orthogonal ensemble
### Circular Symplectic ensemble
### Other useful functionality
List the other things that happen order the hood: murn_naka, young subgroup, accounting with tableaux etc etc...

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

## Acknowledgements
The authors thank the Natural Sciences and Engineering Research Council of Canada and the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec.
