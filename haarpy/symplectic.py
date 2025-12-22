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
Symplectic group Python interface

References
----------
    [1] Collins, B., & Śniady, P. (2006). Integration with respect to the Haar measure on unitary,
    orthogonal and symplectic group. Communications in Mathematical Physics, 264(3), 773-795.
    [2] Matsumoto, S. (2013). Weingarten calculus for matrix ensembles associated with compact
    symmetric spaces. arXiv preprint arXiv:1301.5401.
    [3] Macdonald, I. G. (1998). Symmetric functions and Hall polynomials. Oxford university press.
"""

from math import prod
from fractions import Fraction
from functools import lru_cache
from itertools import product
from sympy import Symbol, factorial, factor, fraction, simplify, Expr
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from sympy.core.numbers import Integer
from haarpy import (
    get_conjugacy_class,
    murn_naka_rule,
    HyperoctahedralGroup,
    irrep_dimension,
    hyperoctahedral_transversal,
)


@lru_cache
def twisted_spherical_function(permutation: Permutation, partition: tuple[int]) -> Fraction:
    """Returns the twisted spherical function of the Gelfand pair (S_2k, H_k)

    Parameters
    ----------
        permutation (Permutation) : a permutation of the symmetric group S_2k
        partition (tuple[int]) : a partition of k

    Returns
    -------
        Fraction : the twisted spherical function of the given permutation

    Raise
    -----
        TypeError : if partition is not a tuple
        TypeError : if permutation argument is not a permutation.
        ValueError : if the degree of the permutation is not a factor of 2
        ValueError : if the degree of the partition and the permutation are incompatible

    Examples
    --------
        >>> from sympy.combinatorics import Permutation
        >>> from haarpy import twisted_spherical_function
        >>> twisted_spherical_function(Permutation(5, 0, 1, 2), (2, 1))
        Fraction(1, 4)

    See Also
    --------
        murn_naka_rule
    """
    if not isinstance(partition, tuple):
        raise TypeError

    if not isinstance(permutation, Permutation):
        raise TypeError

    degree = permutation.size

    if degree % 2:
        raise ValueError("degree should be a factor of 2")
    if degree != 2 * sum(partition):
        raise ValueError("Incompatible partition and permutation")

    duplicate_partition = tuple(part for part in partition for _ in range(2))
    hyperocta = HyperoctahedralGroup(degree // 2)
    numerator = sum(
        murn_naka_rule(duplicate_partition, get_conjugacy_class(~zeta * permutation, degree))
        * zeta.signature()
        for zeta in hyperocta.generate()
    )
    return Fraction(numerator, hyperocta.order())


@lru_cache
def weingarten_symplectic(permutation: Permutation, half_dimension: Symbol) -> Expr:
    """Returns the symplectic Weingarten function

    Parameters
    ----------
        permutation (Permutation) : a permutation of the symmetric group S_2k
        half_dimension (Symbol) : half the dimension of the symplectic group

    Returns
    -------
        Expr : the Weingarten function

    Raise
    -----
        ValueError : if the degree 2k of the symmetric group S_2k is not a factor of 2

    Examples
    --------
        >>> from sympy import Symbol
        >>> from sympy.combinatorics import Permutation
        >>> from haarpy import weingarten_symplectic
        >>> d = Symbol("d")
        >>> weingarten_symplectic(Permutation(0, 1, 2, 3, 4, 5), 7)
        Fraction(-1, 201600)
        >>> weingarten_symplectic(Permutation(0, 1, 2, 3, 4, 5), d)
        -1/(8*d*(d - 2)*(d - 1)*(d + 1)*(2*d + 1))

    See Also
    --------
        twisted_spherical_function, sympy.utilities.iterables.partitions
    """
    degree = permutation.size
    if degree % 2:
        raise ValueError("The degree of the symmetric group S_2k should be even")
    half_degree = degree // 2

    partition_tuple = tuple(
        sum((value * (key,) for key, value in part.items()), ()) for part in partitions(half_degree)
    )
    duplicate_partition_tuple = tuple(
        tuple(part for part in partition for _ in range(2)) for partition in partition_tuple
    )
    irrep_dimension_gen = (irrep_dimension(partition) for partition in duplicate_partition_tuple)
    twisted_spherical_gen = (
        twisted_spherical_function(permutation, partition) for partition in partition_tuple
    )
    coefficient_gen = (
        prod(
            2 * half_dimension - 2 * i + j
            for i in range(len(partition))
            for j in range(partition[i])
        )
        for partition in partition_tuple
    )

    if isinstance(half_dimension, (int, Fraction)):
        weingarten = sum(
            Fraction(
                irrep_dim * zonal_spherical,
                coefficient,
            )
            for irrep_dim, zonal_spherical, coefficient in zip(
                irrep_dimension_gen, twisted_spherical_gen, coefficient_gen
            )
            if coefficient
        ) * Fraction(
            2**half_degree * factorial(half_degree),
            factorial(degree),
        )
    else:
        weingarten = (
            sum(
                irrep_dim * zonal_spherical / coefficient
                for irrep_dim, zonal_spherical, coefficient in zip(
                    irrep_dimension_gen, twisted_spherical_gen, coefficient_gen
                )
                if coefficient
            )
            * 2**half_degree
            * factorial(half_degree)
            / factorial(degree)
        )
        numerator, denominator = fraction(simplify(weingarten))
        weingarten = simplify(factor(numerator) / factor(denominator))

    return weingarten


@lru_cache
def haar_integral_symplectic(
    sequences: tuple[tuple[Expr]],
    half_dimension: Symbol,
) -> Expr:
    """Returns integral over symplectic group polynomial sampled at random from the Haar measure

    Parameters
    ----------
        sequences (tuple[tuple[Expr]]) : indices of matrix elements
        half_dimension (Symbol) : half the dimension of the symplectic group

    Returns
    -------
        Expr : integral under the Haar measure

    Raise
    -----
        ValueError : if sequences do not contain 2 tuples
        ValueError : if tuples i and j are of different length
        TypeError : if the half_dimension is not int nor Symbol
        TypeError : if dimension is int and sequence is not
        ValueError : if all sequence indices are not between 0 and 2*dimension - 1
        TypeError : if sequence containt something else than Expr
        TypeError : if symbolic sequences have the wrong format

    Examples
    --------
        >>> from sympy import Symbol
        >>> from haarpy import haar_integral_symplectic

        >>> d = Symbol("d")
        >>> sequence_1, sequence_2 = (0, 1, 2, d, d+1, d+2), (0, 0, 0, d, d, d)
        >>> haar_integral_symplectic((sequence_1, sequence_2), d)
        1/(4*d*(d + 1)*(2*d + 1))

        >>> d = 4
        >>> sequence_1, sequence_2 = (0, 1, 2, d, d+1, d+2), (0, 0, 0, d, d, d)
        >>> haar_integral_symplectic((sequence_1, sequence_2), d)
        Fraction(1, 720)

    See Also
    --------
        hyperoctahedral_transversal, weingarten_symplectic
    """
    if len(sequences) != 2:
        raise ValueError("Wrong sequence format")

    seq_i, seq_j = sequences

    degree = len(seq_i)

    if degree != len(seq_j):
        raise ValueError("Wrong sequence format")

    if isinstance(half_dimension, int):
        if not all(isinstance(i, int) for i in seq_i + seq_j):
            raise TypeError
        if not all(0 <= i <= 2 * half_dimension - 1 for i in seq_i + seq_j):
            raise ValueError("The matrix indices are outside the dimension range")
        if degree % 2:
            return 0
        seq_i_position = tuple(0 if i < half_dimension else 1 for i in seq_i)
        seq_j_position = tuple(0 if j < half_dimension else 1 for j in seq_j)
        seq_i_value = tuple(i if i < half_dimension else i - half_dimension for i in seq_i)
        seq_j_value = tuple(j if j < half_dimension else j - half_dimension for j in seq_j)
    elif isinstance(half_dimension, Symbol):
        if not all(isinstance(i, (int, Expr)) for i in seq_i + seq_j):
            raise TypeError

        if not all(
            (
                len(xpr.as_ordered_terms()) == 2
                and xpr.as_ordered_terms()[0] == half_dimension
                and isinstance(xpr.as_ordered_terms()[1], Integer)
            )
            or xpr == half_dimension
            for xpr in seq_i + seq_j
            if isinstance(xpr, Expr)
        ):
            raise TypeError
        if degree % 2:
            return 0
        seq_i_position = tuple(0 if isinstance(i, int) else 1 for i in seq_i)
        seq_j_position = tuple(0 if isinstance(j, int) else 1 for j in seq_j)

        seq_i_value = tuple(
            (i if isinstance(i, int) else 0 if i == half_dimension else i.as_ordered_terms()[1])
            for i in seq_i
        )
        seq_j_value = tuple(
            (j if isinstance(j, int) else 0 if j == half_dimension else j.as_ordered_terms()[1])
            for j in seq_j
        )
    else:
        raise TypeError

    def twisted_delta(seq_value, seq_pos, perm):
        return (
            0
            if not all(i1 == i2 for i1, i2 in zip(perm(seq_value)[::2], perm(seq_value)[1::2]))
            else prod(i2 - i1 for i1, i2 in zip(perm(seq_pos)[::2], perm(seq_pos)[1::2]))
        )

    permutation_i_tuple = tuple(
        (perm, twisted_delta(seq_i_value, seq_i_position, perm))
        for perm in hyperoctahedral_transversal(degree)
    )
    permutation_j_tuple = tuple(
        (perm, twisted_delta(seq_j_value, seq_j_position, perm))
        for perm in hyperoctahedral_transversal(degree)
    )

    integral = sum(
        perm_i[1] * perm_j[1] * weingarten_symplectic(perm_j[0] * ~perm_i[0], half_dimension)
        for perm_i, perm_j in product(permutation_i_tuple, permutation_j_tuple)
        if perm_i[1] * perm_j[1]
    )

    if isinstance(half_dimension, Expr):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral
