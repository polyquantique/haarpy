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

from functools import lru_cache
from typing import Generator
from itertools import product


def set_partitions(collection: tuple) -> Generator[tuple[tuple], None, None]:
    """Returns the partitionning of a given collection (set) of objects
    into non-empty subsets.

    Args:
        collection (tuple): An indexable iterable to be partitionned

    Returns:
        generator(tuple[tuple]): all partitions of the input collection

    Raise:
        ValueError: if the collection is not a tuple
    """
    if not isinstance(collection, tuple):
        raise TypeError("collection must be a tuple")

    if len(collection) == 1:
        yield (collection,)
        return

    first = collection[0]
    for smaller in set_partitions(collection[1:]):
        for index, subset in enumerate(smaller):
            yield ((first,) + subset,) + smaller[:index] + smaller[index + 1 :]
        yield ((first,),) + smaller


def perfect_matchings(
    seed: tuple[int],
) -> Generator[tuple[tuple[int]], None, None]:
    """Returns the partitions of a tuple in terms of perfect matchings.

    Args:
        seed (tuple[int]): a tuple representing the (multi-)set that will be partitioned.
            Note that it must hold that ``len(s) >= 2``.

    Returns:
        generator: a generators that goes through all the single-double
        partitions of the tuple

    Raise:
        TypeError: if the seed is not a tuple
    """
    if not isinstance(seed, tuple):
        raise TypeError("seed must be a tuple")

    if len(seed) == 2:
        yield (seed,)

    for idx1 in range(1, len(seed)):
        item_partition = (seed[0], seed[idx1])
        rest = seed[1:idx1] + seed[idx1 + 1 :]
        rest_partitions = perfect_matchings(rest)
        for p in rest_partitions:
            yield ((item_partition),) + p


@lru_cache
def partial_order(
    partition_1: tuple[tuple[int]], partition_2: tuple[tuple[int]]
) -> bool:
    """Returns True if parition_1 <= partition_2 in terms of partial order

    For parition_1 and partition_2, two partitions of the same set, we call
    parition_1 <= partition_2 if and only if each block of parition_1 is
    contained in some block of partition_2

    Ex.
    ((0,1), (2,3), (4,)) < ((0,1), (2,3,4))

    Args:
        partition_1 tuple[tuple[int]]: The partition of lower order
        partition_2 tuple[tuple[int]]: The partition of higher order

    Returns:
        bool: True if parition_1 <= partition_2
    """
    for part in partition_1:
        if all(not set(part).issubset(bigger_part) for bigger_part in partition_2):
            return False

    return True


@lru_cache
def meet_operation(
    partition_1: tuple[tuple[int]], partition_2: tuple[tuple[int]]
) -> tuple[tuple]:
    """Returns the greatest lower bound of the two input partitions

    For parition_1 and partition_2, two partitions of the same set,
    the meet operation yields the greatest lower bound of both partitions

    Ex.
    ((0,1), (2,3), (4,)) ∧ ((0,1,2), (3,4)) = ((0,1), (2,), (3,), (4,))

    Args:
        partition_1 (tuple[tuple[int]]): partition of a set
        partition_2 (tuple[tuple[int]]): partition of a set

    Return:
        tuple[tuple]: Greatest lower bound
    """
    partition_1 = tuple(set(part) for part in partition_1)
    partition_2 = tuple(set(part) for part in partition_2)

    meet_list = [
        block_1 & block_2
        for block_1, block_2 in product(partition_1, partition_2)
        if block_1 & block_2
    ]

    return tuple(sorted(tuple(block) for block in meet_list))


@lru_cache
def join_operation(
    partition_1: tuple[tuple[int]], partition_2: tuple[tuple[int]]
) -> tuple[tuple]:
    """Returns the least upper bound of the two input partitions

    For parition_1 and partition_2, two partitions of the same set,
    the join operation yields the least upper bound of both partitions

    Ex.
    ((0,1), (2,), (3,4)) ∨ ((0,2), (1,), (3,), (4,)) = ((0,1,2), (3,4))

    Args:
        partition_1 (tuple[tuple[int]]): partition of a set
        partition_2 (tuple[tuple[int]]): partition of a set

    Return:
        tuple[tuple[int]]: Least upper bound
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

    return tuple(
        sorted(
            tuple(sorted(value for index in block for value in partition_2[index]))
            for block in merged
        )
    )


@lru_cache
def is_crossing_partition(partition: tuple[tuple[int]]) -> bool:
    """
    Checks if the partition is a crossing partition

    Ex.
    Crossing partition : ((0,2,4), (1,3))
    Non crossing partition : ((0,3,4), (1,2))

    Args:
        partition (tuple[tuple[int]])): partition of a set

    Returns:
        bool: True if the partition is crossing, False otherwise
    """
    filtered_partition = tuple(
        block for block in partition if len(block) != 1 and block[0] + 1 != block[-1]
    )

    for index, previous_block in enumerate(filtered_partition[:-1]):
        for next_block in filtered_partition[index + 1 :]:
            if previous_block[-1] < next_block[0]:
                break
            for value in previous_block[1:]:
                if next_block[0] < value < next_block[-1]:
                    return True

    return False
