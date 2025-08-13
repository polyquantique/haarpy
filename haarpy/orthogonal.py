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

from typing import Generator
from math import prod
from fractions import Fraction
from functools import lru_cache
from sympy import Symbol, factorial, factor, fraction, simplify
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.utilities.iterables import partitions
from haarpy import murn_naka_rule, get_conjugacy_class, irrep_dimension


@lru_cache
def hyperoctahedral(degree: int) -> PermutationGroup:
    """Return the hyperoctahedral group

    Args:
        degree (int): The degree k of the hyperoctahedral group H_k

    Returns:
        (PermutationGroup): The hyperoctahedral group

    Raise:
        TypeError: If degree is not of type int
    """
    if not isinstance(degree, int):
        raise TypeError
    transpositions = tuple(
        Permutation(2 * degree - 1)(2 * i, 2 * i + 1) for i in range(degree)
    )
    double_transpositions = tuple(
        Permutation(2 * degree - 1)(2 * i, 2 * j)(2 * i + 1, 2 * j + 1)
        for i in range(degree)
        for j in range(i + 1, degree)
    )
    return PermutationGroup(transpositions + double_transpositions)


def perfect_matchings(seed: tuple[int]) -> Generator[tuple[tuple[int]], None, None]:
    """Returns the partitions of a tuple in terms of perfect matchings.

    Args:
        seed (tuple[int]): a tuple representing the (multi-)set that will be partitioned.
            Note that it must hold that ``len(s) >= 2``.

    Returns:
        generator: a generators that goes through all the single-double
        partitions of the tuple
    """
    if len(seed) == 2:
        yield seed

    for idx1 in range(1, len(seed)):
        item_partition = (seed[0], seed[idx1])
        rest = seed[1:idx1] + seed[idx1 + 1 :]
        rest_partitions = perfect_matchings(rest)
        for p in rest_partitions:
            if isinstance(p[0], tuple):
                yield ((item_partition),) + p
            else:
                yield (item_partition, p)


def hyperoctahedral_transversal(degree: int) -> Generator[Permutation, None, None]:
    """Returns a generator with the permutations of M_2k, the complete set of coset
    representatives of S_2k/H_k as seen in Macdonald's "Symmetric Functions and Hall
    Polynomials" chapter VII

    Args:
        degree (int): Degree 2k of the set M_2k

    Returns:
        (Generator[Permutation]): The permutations of M_2k
    """
    if degree % 2:
        raise ValueError("degree should be a factor of 2")
    if degree == 2:
        return (Permutation(1),)
    flatten_pmp = (
        tuple(i for pair in pmp for i in pair)
        for pmp in perfect_matchings(tuple(range(degree)))
    )
    return (Permutation(pmp) for pmp in flatten_pmp)


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
    hyperocta = hyperoctahedral(degree // 2)
    numerator = sum(
        murn_naka_rule(
            double_partition, get_conjugacy_class(permutation * zeta, degree)
        )
        for zeta in hyperocta.generate()
    )
    return Fraction(numerator, hyperocta.order())


@lru_cache
def weingarten_orthogonal(
    permutation: Permutation, orthogonal_dimension: Symbol
) -> Symbol:
    """Returns the orthogonal Weingarten function

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        orthogonal_dimension (int): Dimension of the orthogonal group

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
