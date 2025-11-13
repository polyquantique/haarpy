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
"""

from math import prod
from fractions import Fraction
from itertools import product
from functools import lru_cache
from sympy import Symbol, factorial, factor, fraction, simplify
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from haarpy import (
    get_conjugacy_class,
    murn_naka_rule,
    hyperoctahedral,
    irrep_dimension,
    hyperoctahedral_transversal,
)


@lru_cache
def twisted_spherical_function(
    permutation: Permutation, partition: tuple[int]
) -> float:
    """Returns the twisted spherical function of the Gelfand pair (S_2k, H_k)
    as seen in Macdonald's "Symmetric Functions and Hall Polynomials" chapter VII

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        partition (tuple[int]): A partition of k

    Returns:
        (float): The twisted spherical function of the given permutation

    Raise:
        TypeError: If partition is not a tuple
        TypeError: If permutation argument is not a permutation.
        ValueError: If the degree of the permutation is not a factor of 2
        ValueError: If the degree of the partition and the permutation are incompatible
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
    hyperocta = hyperoctahedral(degree // 2)
    numerator = sum(
        murn_naka_rule(
            duplicate_partition, get_conjugacy_class(~zeta * permutation, degree)
        )
        * zeta.signature()
        for zeta in hyperocta.generate()
    )
    return Fraction(numerator, hyperocta.order())


@lru_cache
def weingarten_symplectic(
    permutation: Permutation, symplectic_dimension: Symbol
) -> Symbol:
    """Returns the symplectic Weingarten function

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        symplectic_dimension (int): The dimension of the symplectic group

    Returns:
        Symbol : The Weingarten function

    Raise:
        ValueError : If the degree 2k of the symmetric group S_2k is not a factor of 2
    """
    degree = permutation.size
    if degree % 2:
        raise ValueError("The degree of the symmetric group S_2k should be even")
    half_degree = degree // 2

    partition_tuple = tuple(
        sum((value * (key,) for key, value in part.items()), ())
        for part in partitions(half_degree)
    )
    duplicate_partition_tuple = tuple(
        tuple(part for part in partition for _ in range(2))
        for partition in partition_tuple
    )
    irrep_dimension_gen = (
        irrep_dimension(partition) for partition in duplicate_partition_tuple
    )
    twisted_spherical_gen = (
        twisted_spherical_function(permutation, partition)
        for partition in partition_tuple
    )
    coefficient_gen = (
        prod(
            2 * symplectic_dimension - 2 * i + j
            for i in range(len(partition))
            for j in range(partition[i])
        )
        for partition in partition_tuple
    )

    if isinstance(symplectic_dimension, int):
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
    sequences: tuple[tuple[int]],
    symplectic_dimension: int,
) -> Fraction:
    """Returns integral over symplectic group polynomial sampled at random from the Haar measure

    Args:
        sequences (tuple[tuple[int]]): Indices of matrix elements
        symplectic_dimension (int): Dimension of the symplectic group

    Returns:
        Expr: Integral under the Haar measure

    Raise:
        ValueError: If sequences doesn't contain 2 tuples
        ValueError: If tuples i and j are of different length
        TypeError: If the symplectic_dimension is not int
        ValueError: If all sequence indices are not between 1 and dimension
    """
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j = sequences
    degree = len(seq_i)

    if degree != len(seq_j):
        raise ValueError("Wrong tuple format")

    if not isinstance(symplectic_dimension, int):
        raise TypeError(
            "Unlike other compact groups, "
            "the symplectic group dimension must be an integer to compute the integral."
        )

    if not (
        all(1 <= i <= symplectic_dimension for i in seq_i)
        and all(1 <= j <= symplectic_dimension for j in seq_j)
    ):
        raise ValueError

    if degree % 2:
        return 0

    def twisted_delta(seq, permutation, n):
        return prod(
            (
                1
                if 1 <= i <= n and j == i + n
                else -1 if 1 <= j <= n and i == j + n else 0
            )
            for i, j in zip(permutation(seq)[::2], permutation(seq)[1::2])
        )

    permutation_i = (
        (twisted_delta(seq_i, permutation, symplectic_dimension), permutation)
        for permutation in hyperoctahedral_transversal(degree)
    )

    permutation_j = (
        (twisted_delta(seq_j, permutation, symplectic_dimension), permutation)
        for permutation in hyperoctahedral_transversal(degree)
    )

    return sum(
        perm_i[0]
        * perm_j[0]
        * weingarten_symplectic(perm_j[1] * ~perm_i[1], symplectic_dimension)
        for perm_i, perm_j in product(permutation_i, permutation_j)
        if perm_i[0] * perm_j[0]
    )
