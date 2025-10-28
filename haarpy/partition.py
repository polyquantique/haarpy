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
Partition Python interface
"""

from typing import Generator
from collections.abc import Sequence


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

    return tuple(sorted(tuple(block) for block in meet_list))


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

    merged = []
    for index_set in parent:
        overlap = [m for m in merged if m & index_set]
        for m in overlap:
            index_set |= m
            merged.remove(m)
        merged.append(index_set)

    return tuple(sorted(
        tuple(sorted(
            value
            for index in block
            for value in partition_2[index]
        ))
        for block in merged
    ))
