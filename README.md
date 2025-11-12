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
np.array([average_mod_single_elem(i) for i in range(2, 5)]) 
# Output: array([0.50464014, 0.33273422, 0.25036507])
```

Haarpy allows you to obtain this (and many other!) overages analytically. We first recall that the expression we are trying to calculate is $\int dU |U_{i,j}|^2 = \int dU U_{i,j} U_{i,j}^*$. With this expression in mind we can use Sympy to create a symbolic variable $d$ for the dimension of the unitary and simply write

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
The averages we are calculating are obtained by using so-called Weingarten calculus. Averages over the unitary group can be written in quite general form as follows
$ \int dU U_{i_1j_1}\cdots U_{i_nj_n}U^*_{i^'_1 j^'_1}\cdots U^*_{i^'_n j^'_n} =  \sum_{\sigma,\tau\in S_n}\delta_{i_1 i^'_{\sigma(1)}}\cdots\delta_{i_ni^'_{\sigma(n)}} \delta_{j_1j'_{\tau(1)}}\cdots\delta_{j_nj^'_{\tau(n)}}W_{\sigma \tau^{-1}}(d)$
where $W_{\sigma, \tau}(d)$ is the (unitary) Weingarten function associated with the permutation $\sigma \tau^{-1}$ and $d$ is the dimensionality of the unitary matrices involved.

One can access directly the Weingarten functions by calling
```Python
from haarpy import weingarten_unitary
weingarten_unitary((1,2,3),d)
# Output: (-2*d**2 - 13)/(d*(d - 5)*(d - 4)*(d - 2)*(d - 1)**2*(d + 1)**2*(d + 2)*(d + 4)*(d + 5))
```
NEED TO EXPLAIN WHAT (1,2,3) is and how to use Sympy's Permutation!


## Haarpy functionality

For each item below, explain when the integral is zero and what each weingarten function eats (a pemutation or perfect matching permutation or whatever it is)
### Unitary group
### Orthogonal group
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