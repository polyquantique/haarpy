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

References
----------
    [1] Matsumoto, S. (2013). Weingarten calculus for matrix ensembles associated with compact
        symmetric spaces. arXiv preprint arXiv:1301.5401.
"""

from math import prod
from fractions import Fraction
from functools import lru_cache
from typing import Union
from collections import Counter
from sympy import Symbol, Expr, fraction, factor, cancel, together
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.core.numbers import Integer
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
    """Returns the circular orthogonal ensemble's Weingarten functions

    Parameters
    ----------
        permutation (Permutation) : A permutation of S_2k or its coset-type
        coe_dimension (Symbol) : The dimension of the COE

    Returns
    -------
        Expr : The Weingarten function

    Examples
    --------
        >>> from sympy import Symbol
        >>> from sympy.combinatorics import Permutation
        >>> from haarpy import weingarten_circular_orthogonal
        >>> d = Symbol("d")
        >>> weingarten_circular_orthogonal(Permutation(3)(0,1), 4)
        Fraction(3, 70)
        >>> weingarten_circular_orthogonal(Permutation(3)(0,1), d)
        (d + 2)/(d*(d + 1)*(d + 3))
        >>> weingarten_circular_orthogonal((1,1), d)
        (d + 2)/(d*(d + 1)*(d + 3))

        Where (1,1) is the coset-type of Permutation(3)(0,1)

    See Also
    --------
        coset_type, weingarten_orthogonal
    """
    return weingarten_orthogonal(permutation, coe_dimension + 1)


@lru_cache
def weingarten_circular_symplectic(permutation: Permutation, cse_dimension: Symbol) -> Expr:
    """Returns the circular symplectic ensembles Weingarten functions

    Parameters
    ----------
        permutation (Permutation) : A permutation of the symmetric group S_2k
        cse_dimension (int) : The dimension of the CSE

    Returns
    -------
        Expr : The Weingarten function

    Examples
    --------
        >>> from sympy import Symbol
        >>> from sympy.combinatorics import Permutation
        >>> from haarpy import weingarten_circular_symplectic
        >>> d = Symbol("d")
        >>> weingarten_circular_symplectic(Permutation(3)(0,1), 4)
        Fraction(-3, 140)
        >>> weingarten_circular_symplectic(Permutation(3)(0,1), d)
        (1 - d)/(d*(2*d - 3)*(2*d - 1))

    See Also
    --------
        weingarten_symplectic
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

    Parameters
    ----------
        sequences (tuple[tuple[int]]) : Indices of matrix elements
        group_dimension (Symbol) : Dimension of the orthogonal group

    Returns
    -------
        Expr : Integral under the Haar measure

    Raise
    -----
        ValueError : if sequences do not contain 2 tuples
        ValueError : if tuples i and j are of odd size

    Examples
    --------
        >>> from sympy import Symbol
        >>> from haarpy import haar_integral_circular_orthogonal
        >>> d = Symbol("d")
        >>> seq_i, seq_j = (0, 0, 1, 2), (1, 0, 0, 2)
        >>> haar_integral_circular_orthogonal((seq_i, seq_j), 7)
        Fraction(-1, 280)
        >>> haar_integral_circular_orthogonal((seq_i, seq_j), d)
        -2/(d*(d + 1)*(d + 3))

    See Also
    --------
        coset_type, stabilizer_coset, weingarten_circular_orthogonal
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
        numerator, denominator = fraction(together(integral))
        integral = factor(numerator) / factor(denominator)

    return integral


@lru_cache
def haar_integral_circular_symplectic(sequences: tuple[tuple[Expr]], half_dimension: Expr) -> Expr:
    """Returns integral over circular symplectic ensemble polynomial
    sampled at random from the Haar measure

    Parameters
    ----------
        sequences (tuple[tuple[int]]) : Indices of matrix elements
        half_dimension (Symbol) : Half the dimension of the unitary group

    Returns
    -------
        Expr : Integral under the Haar measure

    Raise
    -----
        ValueError : if sequences do not contain 2 tuples
        ValueError : if tuples i and j are of odd size
        TypeError : if dimension is int and sequence is not
        TypeError : if the half_dimension is neither int nor Symbol
        ValueError : if all sequence indices are not between 0 and 2*dimension - 1
        TypeError : if sequence contains something other than Expr
        TypeError : if symbolic sequences have the wrong format

    Examples
    --------
        >>> from sympy import Symbol
        >>> from haarpy import haar_integral_circular_symplectic
        >>> d = Symbol("d")
        >>> seq_i_num, seq_j_num = (0, 3, 2, 1), (0, 1, 2, 3)
        >>> haar_integral_circular_symplectic((seq_i_num, seq_j_num), 2)
        Fraction(1, 6)
        >>> seq_i_symb, seq_j_symb = (0, d+1, d, 1), (0, 1, d, d + 1)
        >>> haar_integral_circular_symplectic((seq_i_symb, seq_j_symb), d)
        -1/(2*d*(2*d - 3)*(2*d - 1))

    See Also
    --------
        weingarten_circular_symplectic
    """
    if len(sequences) != 2:
        raise ValueError("Wrong sequence format")

    seq_i, seq_j = sequences

    degree = len(seq_i)

    if degree % 2 or len(seq_j) % 2:
        raise ValueError("Wrong sequence format")

    if isinstance(half_dimension, int):
        if not all(isinstance(i, int) for i in seq_i + seq_j):
            raise TypeError
        if not all(0 <= i <= 2 * half_dimension - 1 for i in seq_i + seq_j):
            raise ValueError("The matrix indices are outside the dimension range")
        if degree != len(seq_j):
            return 0
        coefficient = prod(-1 if i < half_dimension else 1 for i in (seq_i + seq_j)[::2])
        shifted_i = [
            (
                i + half_dimension
                if i < half_dimension and index % 2 == 0
                else i - half_dimension if index % 2 == 0 else i
            )
            for index, i in enumerate(seq_i)
        ]
        shifted_j = [
            (
                i + half_dimension
                if i < half_dimension and index % 2 == 0
                else i - half_dimension if index % 2 == 0 else i
            )
            for index, i in enumerate(seq_j)
        ]

    elif isinstance(half_dimension, Symbol):
        if not all(isinstance(i, (int, Expr)) for i in seq_i + seq_j):
            raise TypeError

        if not all(
            (
                len(xpr.as_ordered_terms()) == 2
                and xpr.as_ordered_terms()[0] == half_dimension
                and isinstance(xpr.as_ordered_terms()[1], Integer)
                and xpr.as_ordered_terms()[1] > 0
            )
            or xpr == half_dimension
            for xpr in seq_i + seq_j
            if isinstance(xpr, Expr)
        ):
            raise TypeError
        if degree != len(seq_j):
            return 0
        coefficient = prod(-1 if isinstance(i, int) else 1 for i in (seq_i + seq_j)[::2])
        shifted_i = [
            (
                i + half_dimension
                if isinstance(i, int) and index % 2 == 0
                else (
                    0
                    if i == half_dimension and index % 2 == 0
                    else i.as_ordered_terms()[1] if index % 2 == 0 else i
                )
            )
            for index, i in enumerate(seq_i)
        ]
        shifted_j = [
            (
                i + half_dimension
                if isinstance(i, int) and index % 2 == 0
                else (
                    0
                    if i == half_dimension and index % 2 == 0
                    else i.as_ordered_terms()[1] if index % 2 == 0 else i
                )
            )
            for index, i in enumerate(seq_j)
        ]

    else:
        raise TypeError

    permutation_tuple = (
        permutation
        for permutation in SymmetricGroup(degree).generate()
        if permutation(shifted_i) == shifted_j
    )

    integral = coefficient * sum(
        weingarten_circular_symplectic(permutation, half_dimension)
        for permutation in permutation_tuple
    )

    if isinstance(half_dimension, Expr):
        numerator, denominator = fraction(cancel(together(integral)))
        integral = factor(numerator) / factor(denominator)

    return integral
