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
from collections.abc import Sequence
from math import factorial, prod

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
            yield smaller[:index] + ((first,) + subset,) + smaller[index + 1:]
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
        if not any(set(part).issubset(bigger_part) for bigger_part in partition_2):
            return False
    
    return True


def meet_operation(partition_1: tuple[tuple], partition_2: tuple[tuple]) -> tuple[tuple]:
    """Meet operation of two partitions

    Args:
        partition_1 (tuple(tuple)): A partition of a set
        partition_2 (tuple(tuple)): A partition of a set

    Return:   
    """
    #RAISE VALUE ERROR IF NOT SAME PARTITION K
    return


def join_operation(partition_1: tuple[tuple], partition_2: tuple[tuple]) -> tuple[tuple]:
    """
    """
    partition_1 = tuple(set(part) for part in partition_1)
    partition_2 = tuple(set(part) for part in partition_2)

    joined_list = []
    for block_1 in partition_1:
        intersection = block_1
        for block_2 in partition_2:
            if intersection & block_2:
                intersection.update(block_2)
            joined = False
            for index, block in enumerate(joined_list):
                if intersection & block:
                    joined_list[index].update(intersection)
                    joined = True
                    break
            if not joined:
                joined_list.add(intersection)

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
    permutation_size: int,
):
    """
    """
    #1. generate sigma^tau
    #2. Generate set partitions of each idividual block
    #3. Itertool product all block to find all partial order
    return


def weingarten_centered_random_permutation():
    return


def haar_integral_permutation():
    return


def haar_integral_centered_random_permutation():
    return
