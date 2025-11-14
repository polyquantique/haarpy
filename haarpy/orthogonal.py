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
Orthogonal group Python interface
"""

from math import prod
from fractions import Fraction
from itertools import product
from functools import lru_cache
from typing import Union
from collections import Counter
from sympy import Symbol, Expr, factorial, factor, fraction, simplify
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from haarpy import (
    murn_naka_rule,
    get_conjugacy_class,
    irrep_dimension,
    HyperoctahedralGroup,
    hyperoctahedral_transversal,
    coset_type,
    coset_type_representative,
)


@lru_cache
def zonal_spherical_function(permutation: Permutation, partition: tuple[int]) -> float:
    """Returns the zonal spherical function of the Gelfand pair (S_2k, H_k)
    as seen in Macdonald's "Symmetric Functions and Hall Polynomials" chapter VII

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        partition (tuple[int]): A partition of k

    Returns:
        (float): The zonal spherical function of the given permutation

    Raise:
        TypeError: If partition argument is not a tuple
        TypeError: If permutation argument is not a permutation
    """
    if not isinstance(partition, tuple):
        raise TypeError

    if not isinstance(permutation, Permutation):
        raise TypeError

    degree = permutation.size

    if degree % 2:
        raise ValueError("degree should be a factor of 2")
    if degree != 2 * sum(partition):
        raise ValueError("Invalid partition and cyle-type")

    double_partition = tuple(2 * part for part in partition)
    hyperocta = HyperoctahedralGroup(degree // 2)
    numerator = sum(
        murn_naka_rule(
            double_partition, get_conjugacy_class(permutation * zeta, degree)
        )
        for zeta in hyperocta.generate()
    )
    return Fraction(numerator, hyperocta.order())


@lru_cache
def weingarten_orthogonal(
    permutation: Union[Permutation, tuple[int]], orthogonal_dimension: Symbol
) -> Expr:
    """Returns the orthogonal Weingarten function

    Args:
        permutation (Permutation, tuple[int]): A permutation of S_2k or its coset-type
        orthogonal_dimension (int): Dimension of the orthogonal group

    Returns:
        Symbol: The Weingarten function

    Raise:
        TypeError: if unitary_dimension has the wrong type
        TypeError: if permutation has the wrong type
        ValueError: if the degree 2k of the symmetric group S_2k is not a factor of 2
    """
    if not isinstance(orthogonal_dimension, (Expr, int)):
        raise TypeError(
            "orthogonal_dimension must be an instance of int or sympy.Symbol"
        )

    if isinstance(permutation, (tuple, list)) and all(
        isinstance(value, int) for value in permutation
    ):
        permutation = coset_type_representative(permutation)
    elif not isinstance(permutation, Permutation):
        raise TypeError

    degree = permutation.size
    if degree % 2:
        raise ValueError("The degree of the symmetric group S_2k should be even")

    half_degree = degree // 2

    partition_tuple = tuple(
        sum((value * (key,) for key, value in part.items()), ())
        for part in partitions(half_degree)
    )
    double_partition_tuple = tuple(
        tuple(2 * part for part in partition) for partition in partition_tuple
    )
    irrep_dimension_gen = (
        irrep_dimension(partition) for partition in double_partition_tuple
    )
    zonal_spherical_gen = (
        zonal_spherical_function(permutation, partition)
        for partition in partition_tuple
    )
    coefficient_gen = (
        prod(
            orthogonal_dimension + 2 * j - i
            for i in range(len(partition))
            for j in range(partition[i])
        )
        for partition in partition_tuple
    )

    if isinstance(orthogonal_dimension, int):
        weingarten = sum(
            Fraction(
                irrep_dim * zonal_spherical,
                coefficient,
            )
            for irrep_dim, zonal_spherical, coefficient in zip(
                irrep_dimension_gen, zonal_spherical_gen, coefficient_gen
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
                    irrep_dimension_gen, zonal_spherical_gen, coefficient_gen
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
def haar_integral_orthogonal(
    sequences: tuple[tuple[int]], orthogonal_dimension: Symbol
) -> Expr:
    """Returns integral over orthogonal group polynomial sampled at random from the Haar measure

    Args:
        sequences (tuple[tuple[int]]): Indices of matrix elements
        orthogonal_dimension (int): Dimension of the orthogonal group

    Returns:
        Expr: Integral under the Haar measure

    Raise:
        ValueError: If sequences doesn't contain 2 tuples
        ValueError: If tuples i and j are of different length
    """
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j = sequences
    degree = len(seq_i)

    if degree != len(seq_j):
        raise ValueError("Wrong tuple format")

    if degree % 2:
        return 0

    permutation_i = (
        perm
        for perm in hyperoctahedral_transversal(degree)
        if perm(seq_i)[::2] == perm(seq_i)[1::2]
    )

    permutation_j = (
        perm
        for perm in hyperoctahedral_transversal(degree)
        if perm(seq_j)[::2] == perm(seq_j)[1::2]
    )

    coset_mapping = Counter(
        coset_type(cycle_j * ~cycle_i)
        for cycle_i, cycle_j in product(permutation_i, permutation_j)
    )

    integral = sum(
        count * weingarten_orthogonal(coset, orthogonal_dimension)
        for coset, count in coset_mapping.items()
    )

    if isinstance(orthogonal_dimension, Expr):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral
