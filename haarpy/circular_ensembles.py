# Copyright 2025 Polyquantique

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Circular ensembles Python interface
"""

from fractions import Fraction
from functools import lru_cache
from sympy import Symbol, fraction, factor, simplify
from sympy.combinatorics import Permutation, SymmetricGroup
from haarpy import weingarten_orthogonal, weingarten_symplectic


@lru_cache
def weingarten_circular_orthogonal(
    permutation: Permutation, coe_dimension: Symbol
) -> Symbol:
    """Returns the circular orthogonal ensembles Weingarten functions

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        coe_dimension (int): The dimension of the COE

    Returns:
        Symbol : The Weingarten function
    """
    return weingarten_orthogonal(permutation, coe_dimension + 1)


@lru_cache
def weingarten_circular_symplectic(
    permutation: Permutation, cse_dimension: Symbol
) -> Symbol:
    """Returns the circular symplectic ensembles Weingarten functions

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        cse_dimension (int): The dimension of the CSE

    Returns:
        Symbol : The Weingarten function
    """
    symplectic_dimension = (
        (2 * cse_dimension - 1) / 2
        if isinstance(cse_dimension, Symbol)
        else Fraction(2 * cse_dimension - 1, 2)
    )
    return weingarten_symplectic(permutation, symplectic_dimension)


@lru_cache
def haar_integral_circular_orthogonal(
    sequences: tuple[tuple[int]], group_dimension: int
) -> Symbol:
    """Returns integral over circular orthogonal ensemble polynomial
    sampled at random from the Haar measure

    Args:
        sequences (tuple(tuple(int))) : Indices of matrix elements
        orthogonal_dimension (int) : Dimension of the orthogonal group

    Returns:
        Symbol : Integral under the Haar measure

    Raise:
        ValueError : If sequences doesn't contain 2 tuples
        ValueError : If tuples i and j are of odd size
    """
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j = sequences
    degree = len(seq_i)

    if degree % 2 or len(seq_j) % 2:
        raise ValueError("Wrong tuple format")
    
    if len(seq_j) != degree:
        return 0

    if sorted(seq_i) != sorted(seq_j):
        return 0
    
    seq_j = list(seq_j)
    integral = sum(
        weingarten_circular_orthogonal(perm, group_dimension)
        for perm in SymmetricGroup(degree).generate()
        if perm(seq_i) == seq_j
    )

    if isinstance(group_dimension, Symbol):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral