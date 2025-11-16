<p align="center">
  <img src="https://raw.githubusercontent.com/polyquantique/haarpy/master/haarpy.svg">
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




Haarpy is a Python library for the symbolic calculation of [Weingarten functions](https://en.wikipedia.org/wiki/Weingarten_function) and related averages of matrix ensembles under their [Haar measure](https://pennylane.ai/qml/demos/tutorial_haar_measure): these include the [classical compact groups](https://arxiv.org/abs/math-ph/0609050) namely the orthogonal, unitary and unitary-symplectic groups, the circular orthogonal and circular symplectic ensembles and the group of permutation matrices.


## Haarpy in action
To introduce the main functionality of Haarpy consider the following problem: imagine that you can generate unitary matrices at random (from the Haar measure); you would like to estimate the average of $|U_{i,j}|^2$ which we write mathematically as $\int dU |U_{i,j}|^2 = \int dU U_{i,j} U_{i,j}^*$. We could obtain this average by using the random matrix functionality of SciPy as follows:
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

Haarpy allows you to obtain this (and many other!) averages analytically. We first recall that the expression we are trying to calculate is $\int dU |U_{i,j}|^2 = \int dU U_{i,j} U_{i,j}^*$. With this expression in mind we can use Sympy to create a symbolic variable $d$ for the dimension of the unitary and write

```Python
from sympy import Symbol
from haarpy import haar_integral_unitary

d = Symbol("d")
haar_integral_unitary(("i", "j", "i", "j"), d)
# Output: 1/d
haar_integral_unitary(("i", "j", "i", "j"), 3)  # We can also put integers
# Output: 1/3
```
Notice the order of the indices! The first `"i"` and `"j"` are the indices of $U$ while the second pair of `"i"` and `"j"` are the indices of $U^*$.

Imagine that now we want to calculate something like $\int dU U_{i,m} U_{j,n} U_{k,o} U_{i,m}^* U_{j,n}^* U_{k,p}^*$ $= \int dU |U_{i,m} U_{j,n} U_{k,o}|^2$ we simply do
```
haar_integral_unitary(("ijk", "mno", "ijk", "mno"), d)
# Output: (d**2 - 2)/(d*(d - 2)*(d - 1)*(d + 1)*(d + 2))
```
The averages we are calculating are obtained by using so-called [Weingarten calculus](https://doi.org/10.1155/S107379280320917X). Unitary Weingarten functions depend only on a conjugacy class of the symmetric group $S_p$ and on the dimension $d$ of the unitaries that are averaged.  For four sequences $\mathbf{i}=\left(i_1,\dots,i_p\right)$, $\mathbf{j}=\left(j_1,\dots,j_p\right)$, $\mathbf{i}^\prime=\left(i_1^\prime,\dots,i_p^\prime\right)$ and $\mathbf{j}^\prime=\left(j_1^\prime,\dots,j_p^\prime\right)$, a convenient closed form expression for averages of unitary matrices is given by

$$
\int dU \ U_{i_1j_1}\ldots U_{i_pj_p} \left(U_{i^\prime_1j^\prime_1}\ldots U_{i^\prime_p,j^\prime_p}\right)^{\ast}   =\sum_{\sigma,\tau\in S_p}\delta_\sigma(\mathbf{i},\mathbf{i}^\prime)\delta_\tau(\mathbf{j},\mathbf{j}^\prime)\text{Wg}_U([\sigma\tau^{-1}];d)\, \quad [1]
$$

where $\text{Wg}_U([\sigma\tau^{-1}];d)$ is the unitary Weingarten function,  $U$ is a Haar-random $d\times d$ unitary matrix, $dU$ is the Haar measure over $U(d)$, $[\sigma]$ is the conjugacy class of element $\sigma$, and

$$\delta_\sigma(\mathbf{i},\mathbf{i}^\prime) = \prod_{s=1}^p\delta_{i_{\sigma(s)},i_s^\prime}.$$

In other words, expectation of polynomials of entries of unitary matrices are given by a sum of Weingarten functions. Note that the integral of Eq. (1) is trivially $0$ if sequences $\mathbf{i}$ and $\mathbf{i}^\prime$ (or equivalently $\mathbf{j}$ and $\mathbf{j}^\prime$) are of different lengths. 

The general average above in Eq. (1) can be compute in Haarpy as 
```Python
haar_integral_unitary(("i_1 i_2 ... i_p","j_1 j_2 ... j_p","i_1' i_2' ... i_p'","j_1' j_2' ... j_p'"), d)
```

One can access directly the Weingarten functions of a given permutation of the symmetric group by calling
```Python
from sympy.combinatorics import Permutation
from haarpy import weingarten_unitary

weingarten_unitary(Permutation(5)(4,3)(2,1,0), d)
# Output: (-2*d**2 - 13)/(d*(d - 5)*(d - 4)*(d - 2)*(d - 1)**2*(d + 1)**2*(d + 2)*(d + 4)*(d + 5))
```
Equivalently, since the unitary Weingarten function is a class function, one can call the previous using the cycle-type of any given permutation, i.e., the partition that labels its conjugacy class.
```Python
weingarten_unitary((3,2,1), d)
# Output: (-2*d**2 - 13)/(d*(d - 5)*(d - 4)*(d - 2)*(d - 1)**2*(d + 1)**2*(d + 2)*(d + 4)*(d + 5))
```


## Haarpy functionality

Haarpy implements Weingarten functions for all the classical compact groups, the circular orthogonal and circular symplectic ensembles as well as the permutation and centered permutation groups. 


### Unitary group
Unitary matrices $U$ are complex-matrices that satisfy $U U^\dagger = I_d$ where $I_d$ is the identity matrix. Here we use $U^\dagger$ to indicate the conjugate-transpose of the matrix $U$.
One can calculate averages over the unitary Haar measure using `haar_integral_unitary` and obtain their associated [Weingarten function](https://doi.org/10.1155/S107379280320917X) using `weingarten_unitary`. The latter function takes as input a permutation specified by a SymPy `Permutation` object or a conjugacy class specified by a tuple, as well as the dimension of the unitary group specified either by a SymPy `Symbol` or by an integer.

### Orthogonal group
Orthogonal matrices $O$ are real-matrices $O$ that satisfy $O O^T = I_d$. Here we use $O^T$ to indicate the transpose of the matrix $O$.
One can calculate averages over the orthogonal Haar measure using `haar_integral_orthogonal` and obtain their associated [Weingarten function](https://doi.org/10.1007/s00220-006-1554-3) using `weingarten_orthogonal`. The latter function takes as input a permutation specified by a SymPy `Permutation` object or a coset-type specified by a tuple, as well as the dimension of the orthogonal group specified either by a SymPy `Symbol` or by an integer.


### Unitary-Symplectic group
Unitary-symplectic matrices $S$ are complex-unitary matrices of even size that are also symplectic, that is they satisfy $S \Omega S^T = \Omega$ where 

$$
\Omega = \left(\begin{array}{cc} 0_d & I_d \\ 
-I_d & 0_d \end{array}\right)
$$ 

is the symplectic form. [Weingarten functions](https://doi.org/10.1007/s00220-006-1554-3) of this group take as input an element of the symmetric group as well as the dimension of the symplectic group and can be calculated using `weingarten_symplectic`. Functionality to calculate averages will be added in the near term.


### Circular Orthogonal ensemble
Circular orthogonal matrices $V$ are simply symmetric unitary matrices. If $U$ is a Haar-random unitary matrix, then $V = U U^T$ is a COE random matrix. One can calculate averages over the COE using `haar_integral_circular_unitary` and obtain their associated [Weingarten function](https://doi.org/10.1142/S2010326313500019) using `weingarten_circular_orthogonal`. The latter function takes as input a permutation specified by a SymPy `Permutation` object or a coset-type specified by a tuple, as well as the dimension of the ensemble matrices specified either by a SymPy `Symbol` or by an integer.


### Circular Symplectic ensemble
Circular symplectic matrices $R$ are obtained by drawing a Haar-random unitary $U$ of even size and calculating $R = -U \Omega U^T \Omega$. [Weingarten functions](https://doi.org/10.1142/S2010326313500019) of this ensemble take as input an element of the symmetric group as well as the dimension of the symplectic group and can be calculated using `weingarten_circular_symplectic`. Functionality to calculate averages will be added in the near term.


### Permutation and centered permutation groups
Weingarten functions associated with these group have been recently introduced. Integration over this discrete group can be performed using `haar_integral_permutation` and `haar_integral_centered_permutation` and the associated [Weingarten function](https://doi.org/10.48550/arXiv.2503.18453) can be accessed as `weingarten_permutation` and `weingarten_centered_permutation`.


### Other useful functionality
Under the hood, Haarpy implements a number of group-theoretic machinery that can be useful in other contexts, including 
* The [Murnaghan-Nakayama rule](https://en.wikipedia.org/wiki/Murnaghan%E2%80%93Nakayama_rule) as `murn_naka_rule` for the characters of the irreducible representations of the symmetric group;
* The dimension of the symmetric group irreps as `irrep_dimension`;
* The dimension of the representations of the unitary group as `representation_dimension`;
* The [hyperoctahedral group](https://en.wikipedia.org/wiki/Hyperoctahedral_group) as `HyperoctahedralGroup`;
* The [Young subgroup](https://en.wikipedia.org/wiki/Young_subgroup) as `YoungSubgroup`;
* The set of all permutations $\sigma\in S_p$ such that $\mathbf{i}^\sigma = \mathbf{i}^\prime$ as `stabilizer_coset`.


## Installation
Haarpy requires Python version 3.9 or later. Installation can be done through the pip command
```
pip install haarpy
```

## Compiling from source
Haarpy has the following dependencies:
* [Python](https://www.python.org/) >= 3.9
* [SymPy](https://www.sympy.org) >= 1.12


## How to cite this work
Please cite as:
```
@misc{cardin2024haarpy,
  author={Cardin, Yanic and de Guise, Hubert and Quesada, Nicol{\'a}s},
  title={Haarpy, a Python library for Weingarten calculus and integration of classical compact groups and ensembles},
  year={2024},
  publisher={GitHub},
  journal={GitHub repository},
  howpublished = {\url{https://github.com/polyquantique/haarpy}},
  version = {0.0.6}
}
```

## Authors
* Yanic Cardin, Hubert de Guise, Nicolás Quesada.


## License
Haarpy is free and open source, released under the Apache License, Version 2.0.


## Acknowledgements
The authors thank the Natural Sciences and Engineering Research Council of Canada and the Ministère de l'Économie, de l'Innovation et de l'Énergie du Québec.
