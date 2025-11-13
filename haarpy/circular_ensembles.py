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

from math import prod
from fractions import Fraction
from functools import lru_cache
from typing import Union
from collections import Counter
from sympy import Symbol, Expr, fraction, factor, simplify
from sympy.combinatorics import Permutation
from haarpy import (
    weingarten_orthogonal,
    weingarten_symplectic,
    coset_type,
    stabilizer_coset,
)


@lru_cache
def weingarten_circular_orthogonal(
    permutation: Union[Permutation, tuple[int]],
    coe_dimension: Symbol,
) -> Expr:
    """Returns the circular orthogonal ensembles Weingarten functions

    Args:
        permutation (Permutation): A permutation of S_2k or its coset-type
        coe_dimension (int): The dimension of the COE

    Returns:
        Expr: The Weingarten function
    """
    return weingarten_orthogonal(permutation, coe_dimension + 1)


@lru_cache
def weingarten_circular_symplectic(
    permutation: Permutation, cse_dimension: Symbol
) -> Expr:
    """Returns the circular symplectic ensembles Weingarten functions

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        cse_dimension (int): The dimension of the CSE

    Returns:
        Expr: The Weingarten function
    """
    symplectic_dimension = (
        (2 * cse_dimension - 1) / 2
        if isinstance(cse_dimension, Expr)
        else Fraction(2 * cse_dimension - 1, 2)
    )
    return weingarten_symplectic(permutation, symplectic_dimension)


@lru_cache
def haar_integral_circular_orthogonal(
    sequences: tuple[tuple[int]], group_dimension: Symbol
) -> Expr:
    """Returns integral over circular orthogonal ensemble polynomial
    sampled at random from the Haar measure

    Args:
        sequences (tuple[tuple[int]]) : Indices of matrix elements
        group_dimension (Symbol) : Dimension of the orthogonal group

    Returns:
        Expr : Integral under the Haar measure

    Raise:
        ValueError : If sequences doesn't contain 2 tuples
        ValueError : If tuples i and j are of odd size
    """
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j = sequences

    if len(seq_i) % 2 or len(seq_j) % 2:
        raise ValueError("Wrong tuple format")

    coset_mapping = Counter(
        coset_type(permutation) for permutation in stabilizer_coset(seq_i, seq_j)
    )

    integral = sum(
        count * weingarten_circular_orthogonal(coset, group_dimension)
        for coset, count in coset_mapping.items()
    )

    if isinstance(group_dimension, Expr):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral


@lru_cache
def haar_integral_circular_symplectic(
    sequences: tuple[tuple[int]], group_dimension: int
) -> Fraction:
    """Returns integral over circular symplectic ensemble polynomial
    sampled at random from the Haar measure

    Args:
        sequences (tuple[tuple[int]]) : Indices of matrix elements
        group_dimension (Symbol) : Dimension of the orthogonal group

    Returns:
        Expr : Integral under the Haar measure

    Raise:
        ValueError : If sequences doesn't contain 2 tuples
        ValueError : If tuples i and j are of odd size
        TypeError : If group_dimension is not an int
        ValueError: If all sequence indices are not between 1 and dimension
    """
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j = sequences

    if len(seq_i) % 2 or len(seq_j) % 2:
        raise ValueError("Wrong tuple format")

    if not isinstance(group_dimension, int):
        raise TypeError(
            "Unlike other ensembles, "
            "the CSE dimension must be an integer to compute the integral."
        )

    if not (
        all(1 <= i <= group_dimension for i in seq_i)
        and all(1 <= j <= group_dimension for j in seq_j)
    ):
        raise ValueError

    coefficient = prod(
        -1 if 1 <= i <= group_dimension else 1 for i in (seq_i + seq_j)[::2]
    )

    shifted_i = tuple(
        (
            i + group_dimension
            if 1 <= i <= group_dimension and index % 2 == 0
            else i - group_dimension if index % 2 == 0 else i
        )
        for index, i in enumerate(seq_i)
    )

    shifted_j = tuple(
        (
            i + group_dimension
            if 1 <= i <= group_dimension and index % 2 == 0
            else i - group_dimension if index % 2 == 0 else i
        )
        for index, i in enumerate(seq_j)
    )

    return coefficient * sum(
        weingarten_circular_symplectic(permutation, group_dimension)
        for permutation in stabilizer_coset(shifted_i, shifted_j)
    )
