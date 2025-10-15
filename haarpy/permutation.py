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

from typing import Generator
from itertools import product
from collections.abc import Sequence
from math import factorial, prod
from sympy import Symbol, simplify, binomial, factor, fraction


def set_partition(collection: Sequence) -> Generator[tuple[tuple], None, None]:
    """Returns the partitionning of a given collection (set) of objects
    into non-empty subsets.

    Args:
        collection (Sequence): An indexable iterable to be partitionned

    Returns:
        generator(tuple(tuple)): all partitions of the input collection

    Raise:
        ValueError: if the collection not an indexable iterable
    """
    if not isinstance(collection, Sequence) or isinstance(collection, range):
        raise TypeError("collection must be an indexable iterable")

    if len(collection) == 1:
        yield (collection,)
        return

    first = collection[0]
    for smaller in set_partition(collection[1:]):
        for index, subset in enumerate(smaller):
            yield ((first,) + subset,) + smaller[:index] + smaller[index + 1 :]
        yield ((first,),) + smaller


def partial_order(partition_1: tuple[tuple], partition_2: tuple[tuple]) -> bool:
    """Returns True if parition_1 <= partition_2 in terms of partial order

    Args:
        partition_1 (tuple(tuple)): The partition of lower order
        partition_2 (tuple(tuple)): The partition of higher order

    Returns:
        bool: True if parition_1 <= partition_2

    Raise:
        ValueError: If both partitions are not composed of unique elements
    """
    flatten_partitions = (
        tuple(i for j in partition for i in j)
        for partition in (partition_1, partition_2)
    )
    if any(len(flatten) != len(set(flatten)) for flatten in flatten_partitions):
        raise ValueError("The partitions must be composed of unique elements")

    for part in partition_1:
        if all(not set(part).issubset(bigger_part) for bigger_part in partition_2):
            return False

    return True


def meet_operation(
    partition_1: tuple[tuple], partition_2: tuple[tuple]
) -> tuple[tuple]:
    """Returns the greatest lower bound of the two input partitions

    Args:
        partition_1 (tuple(tuple)): partition of a set
        partition_2 (tuple(tuple)): partition of a set

    Return:
        tuple(tuple): Greatest lower bound
    """
    partition_1 = tuple(set(part) for part in partition_1)
    partition_2 = tuple(set(part) for part in partition_2)

    meet_list = []
    for block_1 in partition_1:
        for block_2 in partition_2:
            if block_1 & block_2:
                meet_list.append(block_1 & block_2)

    return tuple(tuple(block) for block in meet_list)


def join_operation(
    partition_1: tuple[tuple],
    partition_2: tuple[tuple]
) -> tuple[tuple]:
    """Returns the least upper bound of the two input partitions

    Args:
        partition_1 (tuple(tuple)): partition of a set
        partition_2 (tuple(tuple)): partition of a set

    Return:
        tuple(tuple): Least upper bound
    """
    parent = [
        {
            index 
            for value in block1
            for index, block2 in enumerate(partition_2)
            if value in block2
        }
        for block1 in partition_1
    ]

    merged = [
        {
            index
            for index_set2 in parent
            for index in index_set2
            if index_set1 & index_set2
        }
        for index_set1 in parent
    ]

    block_indices = {
        tuple(index_set) for index_set in merged
    }

    return tuple(sorted(
        tuple(sorted(
            value
            for index in block
            for value in partition_2[index]
        ))
        for block in block_indices
    ))


def mobius_function(partition_1: tuple[tuple], partition_2: tuple[tuple]) -> int:
    """Return the Möbius function as seen in Collin & Nagatsu's
    "Weingarten calculus for centered random permutation matrices"

    Args:
        partition_1 (tuple(tuple)): The intersected partition
        partition_2 (tuple(tuple)): The partition summed over

    Returns:
        int: The value of the Möbius function
    """
    flatten_partitions = (
        tuple(i for j in partition for i in j)
        for partition in (partition_1, partition_2)
    )
    if any(len(flatten) != len(set(flatten)) for flatten in flatten_partitions):
        raise ValueError("The partitions must be composed of unique elements")

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
        (partition for partition in set_partition(block))
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
        (partition for partition in set_partition(block))
        for block in meet_operation(first_partition, second_partition)
    )

    inferieur_partition_tuple = (
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


def haar_integral_permutation(
    row_indices: tuple[int],
    column_indices: tuple[int],
    dimension: Symbol,
) -> Symbol:
    """
    """
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


def haar_integral_centered_permutation(
    row_indices: tuple[int],
    column_indices: tuple[int],
    dimension: Symbol,
) -> Symbol:
    """
    """
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
            set_partition(tuple(i for i, _ in enumerate(row_indices))),
            set_partition(tuple(i for i, _ in enumerate(column_indices))),
        )
        if partial_order(partition_sigma, row_partition)
        and partial_order(partition_tau, column_partition)
    )

    if isinstance(dimension, Symbol):
        num, denum = fraction(simplify(integral))
        integral = factor(num) / factor(denum)

    return integral
