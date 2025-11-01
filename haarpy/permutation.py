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
Permutation matrices Python interface
"""

from math import factorial, prod
from functools import lru_cache
from itertools import product
from collections.abc import Sequence
from sympy import Symbol, simplify, binomial, factor, fraction
from haarpy import set_partitions, meet_operation, join_operation, partial_order


@lru_cache
def mobius_function(
    partition_1: tuple[tuple[int]], partition_2: tuple[tuple[int]]
) -> int:
    """Return the Möbius function
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_

    Args:
        partition_1 (tuple[tuple[int]): The intersected partition
        partition_2 (tuple[tuple[int]]): The partition summed over

    Returns:
        int: The value of the Möbius function
    """
    partition_set_1 = tuple(set(block) for block in partition_1)
    partition_set_2 = tuple(set(block) for block in partition_2)

    partition_intersection = tuple(
        sum(1 for block_1 in partition_set_1 if block_1 & block_2)
        for block_2 in partition_set_2
    )

    return prod(
        (-1) ** (block_count - 1) * factorial(block_count - 1)
        for block_count in partition_intersection
    )


@lru_cache
def weingarten_permutation(
    first_partition: tuple[tuple[int]],
    second_partition: tuple[tuple[int]],
    dimension: Symbol,
) -> Symbol:
    """Returns the Weingarten function for random permutation matrices
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_

    Args:
        first_partition (tuple(tuple(int))): a set partition of integer k
        second_partition (tuple(tuple(int))): a set partition of integer k
        dimension (Symbol): Dimension of the random permutation matrices

    Returns:
        Symbol : The Weingarten function
    """
    disjoint_partition_tuple = tuple(
        (partition for partition in set_partitions(block))
        for block in meet_operation(first_partition, second_partition)
    )

    inferieur_partition_tuple = (
        tuple(block for partition in partition_tuple for block in partition)
        for partition_tuple in product(*disjoint_partition_tuple)
    )

    weingarten = sum(
        mobius_function(partition, first_partition)
        * mobius_function(partition, second_partition)
        / prod(dimension - i for i, _ in enumerate(partition))
        for partition in inferieur_partition_tuple
    )

    return weingarten if isinstance(dimension, int) else simplify(weingarten)


@lru_cache
def weingarten_centered_permutation(
    first_partition: tuple[tuple[int]],
    second_partition: tuple[tuple[int]],
    dimension: Symbol,
) -> Symbol:
    """Returns the Weingarten function for centered random permutation matrices
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_

    Args:
        first_partition (tuple(tuple(int))): a set partition of integer k
        second_partition (tuple(tuple(int))): a set partition of integer k
        dimension (Symbol): Dimension of the centered random permutation matrices

    Returns:
        Symbol : The Weingarten function
    """
    singleton_set_size = sum(
        1
        for block in join_operation(first_partition, second_partition)
        if len(block) == 1
    )

    disjoint_partition_tuple = tuple(
        (partition for partition in set_partitions(block))
        for block in meet_operation(first_partition, second_partition)
    )

    inferieur_partition_tuple = tuple(
        tuple(block for partition in partition_tuple for block in partition)
        for partition_tuple in product(*disjoint_partition_tuple)
    )

    weingarten = sum(
        (-1) ** i
        * binomial(singleton_set_size, i)
        * sum(
            mobius_function(partition, first_partition)
            * mobius_function(partition, second_partition)
            / dimension**i
            / prod(dimension - j for j in range(len(partition) - i))
            for partition in inferieur_partition_tuple
        )
        for i in range(singleton_set_size + 1)
    )

    if isinstance(dimension, Symbol):
        num, denum = fraction(simplify(weingarten))
        weingarten = factor(num) / factor(denum)

    return weingarten


@lru_cache
def haar_integral_permutation(
    row_indices: tuple[int],
    column_indices: tuple[int],
    dimension: Symbol,
) -> Symbol:
    """Returns the integral over Haar random permutation matrices
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_

    Args:
        row_indices (tuple(int)) : sequence of row indices
        column_indices (tuple(int)) : sequence of column indices

    Returns:
        Symbol : Integral under the Haar measure

    Raise:
        TypeError : If row_indices and column_indices are not Sequence
        ValueError : If row_indices and column_indices are of different length
    """
    if not (isinstance(row_indices, Sequence) and isinstance(column_indices, Sequence)):
        raise TypeError

    if len(row_indices) != len(column_indices):
        raise ValueError("Wrong tuple format")

    def sequence_to_partition(sequence: tuple) -> tuple[tuple[int]]:
        return sorted(
            sorted(index for index, value in enumerate(sequence) if value == unique)
            for unique in set(sequence)
        )

    row_partition = sequence_to_partition(row_indices)
    column_partition = sequence_to_partition(column_indices)

    return (
        1 / prod(dimension - i for i, _ in enumerate(row_partition))
        if row_partition == column_partition
        else 0
    )


@lru_cache
def haar_integral_centered_permutation(
    row_indices: tuple[int],
    column_indices: tuple[int],
    dimension: Symbol,
) -> Symbol:
    """Returns the integral over Haar random centered permutation matrices
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_

    Args:
        row_indices (tuple(int)) : sequence of row indices
        column_indices (tuple(int)) : sequence of column indices

    Returns:
        Symbol : Integral under the Haar measure

    Raise:
        TypeError : If row_indices and column_indices are not Sequence
        ValueError : If row_indices and column_indices are of different length
    """
    if not (isinstance(row_indices, Sequence) and isinstance(column_indices, Sequence)):
        raise TypeError

    if len(row_indices) != len(column_indices):
        raise ValueError("Wrong tuple format")

    row_partition = tuple(
        tuple(index for index, value in enumerate(row_indices) if value == unique)
        for unique in set(row_indices)
    )
    column_partition = tuple(
        tuple(index for index, value in enumerate(column_indices) if value == unique)
        for unique in set(column_indices)
    )

    integral = sum(
        weingarten_centered_permutation(
            partition_sigma,
            partition_tau,
            dimension,
        )
        for partition_sigma, partition_tau in product(
            set_partitions(tuple(i for i, _ in enumerate(row_indices))),
            set_partitions(tuple(i for i, _ in enumerate(column_indices))),
        )
        if partial_order(partition_sigma, row_partition)
        and partial_order(partition_tau, column_partition)
    )

    if isinstance(dimension, Symbol):
        num, denum = fraction(simplify(integral))
        integral = factor(num) / factor(denum)

    return integral
