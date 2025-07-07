# Copyright 2024 Polyquantique

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
from fractions import Fraction
from functools import lru_cache
from sympy import Symbol
from sympy.combinatorics import Permutation, SymmetricGroup, PermutationGroup
from unitary import murn_naka_rule, get_conjugacy_class, irrep_dimension


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


@lru_cache
def zonal_spherical_function(permutation_or_partition, partition: tuple[int]) -> float:
    """Returns the zonal spherical function of the Gelfand pair (S_2k, H_k)

    Args:
        perm (Permutation): A permutation of the symmetric group S_2k
        partition (tuple[int]): A partition of k

    Returns:
        (float): The zonal spherical function of the given permutation
    """
    if isinstance(permutation_or_partition, Permutation):
        permutation = permutation_or_partition
        degree = permutation.size
    else:
        degree = sum(permutation_or_partition)
        permutation = Permutation(
            tuple(
                tuple(i + sum(permutation_or_partition[:index]) for i in range(part))
                for index, part in enumerate(permutation_or_partition)
            )
        )
    if degree % 2:
        raise ValueError("degree should be a factor of 2")
    if degree != 2*sum(partition):
        raise ValueError("Invalid partition and cyle-type")

    double_partition = tuple(2 * part for part in partition)
    hyperocta = hyperoctahedral(degree // 2)
    numerator = sum(
        murn_naka_rule(double_partition, get_conjugacy_class(permutation * zeta, degree))
        for zeta in hyperocta.generate()
    )
    return Fraction(numerator, hyperocta.order())


@lru_cache
def weingarten_orthogonal(cycle_type: Permutation, order: int, orthogonal_dimension: int) -> Symbol:
    return


@lru_cache
def haar_integral_orthogonal():
    return