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
from sympy import Symbol, simplify


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
        raise TypeError('collection must be an indexable iterable')

    if len(collection) == 1:
        yield (collection,)
        return
    
    first = collection[0]
    for smaller in set_partition(collection[1:]):
        for index, subset in enumerate(smaller):
            yield ((first,) + subset,) + smaller[:index] + smaller[index + 1:]
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


def meet_operation(partition_1: tuple[tuple], partition_2: tuple[tuple]) -> tuple[tuple]:
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


def join_operation(partition_1: tuple[tuple], partition_2: tuple[tuple]) -> tuple[tuple]:
    """Returns the least upper bound of the two input partitions

    Args:
        partition_1 (tuple(tuple)): partition of a set
        partition_2 (tuple(tuple)): partition of a set

    Return:
        tuple(tuple): Least upper bound
    """
    partition_1 = tuple(set(part) for part in partition_1)
    partition_2 = list(set(part) for part in partition_2)

    joined_list = []
    for block_1 in partition_1:
        for index_2, block_2 in enumerate(partition_2):
            if block_1 & block_2:
                block_1.update(block_2)
                partition_2.pop(index_2)
            joined = False
            for index, block in enumerate(joined_list):
                if block_1 & block:
                    joined_list[index].update(block_1)
                    joined = True
                    break
            if not joined:
                joined_list.append(block_1)

    return tuple(tuple(block) for block in joined_list)


def mobius_function(
    partition_1: tuple[tuple], partition_2: tuple[tuple]
) -> int:
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
        (-1)**(block_count -1) * factorial(block_count - 1)
        for block_count in partition_intersection
    )


def weingarten_permutation(
    first_partition: tuple[tuple[int]],
    second_partition: tuple[tuple[int]],
    permutation_size: Symbol,
) -> Symbol:
    """Returns the Weingarten function for random permutation matrices
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_

    Args:
        first_partition (tuple(tuple(int))): a set partition of integer k
        second_partition (tuple(tuple(int))): a set partition of integer k
        permutation_size (Symbol): Dimension of the random permutation matrices

    Returns:
        Symbol : The Weingarten function
    """
    disjoint_partition_tuple = tuple(
        (partition for partition in set_partition(block))
        for block in meet_operation(first_partition, second_partition)
    )

    inferieur_partition_tuple = (
        tuple(
            block
            for partition in partition_tuple
            for block in partition
        )
        for partition_tuple in product(*disjoint_partition_tuple)
    )

    weingarten = sum(
        mobius_function(partition, first_partition)
        * mobius_function(partition, second_partition)
        * prod(permutation_size - i for i, _ in enumerate(partition))
        for partition in inferieur_partition_tuple
    )

    return (
        weingarten
        if isinstance(permutation_size, int)
        else simplify(weingarten)
    )


def weingarten_centered_random_permutation():
    """
    """
    return


def haar_integral_permutation(
    sequence_i: tuple[int],
    sequence_j: tuple[int],
    permutation_size: Symbol,
) -> Symbol:
    """
    """
    def sequence_to_partition(sequence: tuple) -> tuple[tuple[int]]:
        return sorted(
            sorted(index for index, value in enumerate(sequence) if value == unique)
            for unique in set(sequence)
        )
    
    partition_i = sequence_to_partition(sequence_i)
    partition_j = sequence_to_partition(sequence_j)

    return (
        prod(permutation_size - i for i, _ in enumerate(partition_i))
        if partition_i == partition_j
        else 0
    )


def haar_integral_centered_random_permutation():
    """
    """
    return
