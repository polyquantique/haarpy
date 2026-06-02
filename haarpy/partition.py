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

References
----------
    [1] Collins, B., & Nagatsu, M. (2025). Weingarten calculus for centered random permutation
    matrices. arXiv preprint arXiv:2503.18453.
    [2] Matsumoto, S. (2013). Weingarten calculus for matrix ensembles associated with compact
    symmetric spaces. arXiv preprint arXiv:1301.5401.
    [3] Nica, A., & Speicher, R. (2006). Lectures on the combinatorics of free probability
    (Vol. 13). Cambridge University Press.
"""

from functools import lru_cache
from collections.abc import Iterator
from itertools import product
from sympy import Symbol, Matrix, Expr


def set_partitions(collection: tuple) -> Iterator[tuple[tuple, ...]]:
    """Returns the partitionning of a given collection (set) of objects
    into non-empty subsets.

    Parameters
    ----------
        collection (tuple) : an indexable iterable to be partitionned

    Returns
    -------
        Iterator(tuple[tuple]) : all partitions of the input collection

    Raise
    -----
        ValueError : if the collection is not a tuple

    Examples
    --------
        >>> from haarpy import set_partitions
        >>> for partition in set_partitions((0,1,2)):
        >>>     print(partition)
        ((0, 1, 2),)
        ((0,), (1, 2))
        ((0, 1), (2,))
        ((0, 2), (1,))
        ((0,), (1,), (2,))
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


def pair_partitions(
    seed: tuple[int, ...],
) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Returns the pair partitions of a given set.

    Parameters
    ----------
        seed (tuple[int]) : a tuple representing the (multi-)set that will be partitioned.
            Note that it must hold that ``len(s) >= 2``

    Returns
    -------
        Iterator[tuple[tuple[int]]] : all the single-double partitions of the tuple

    Raise
    -----
        TypeError : if the seed is not a tuple

    Examples
    --------
        >>> from haarpy import pair_partitions
        >>> for matching in pair_partitions((0,1,2,3)):
        >>>     print(matching)
        ((0, 1), (2, 3))
        ((0, 2), (1, 3))
        ((0, 3), (1, 2))
    """
    if not isinstance(seed, tuple):
        raise TypeError("seed must be a tuple")

    if len(seed) == 2:
        yield (seed,)

    for idx1 in range(1, len(seed)):
        item_partition = (seed[0], seed[idx1])
        rest = seed[1:idx1] + seed[idx1 + 1 :]
        rest_partitions = pair_partitions(rest)
        for p in rest_partitions:
            yield ((item_partition),) + p


@lru_cache
def partial_order(
    partition_1: tuple[tuple[int, ...], ...], partition_2: tuple[tuple[int, ...], ...]
) -> bool:
    """Checks if parition_1 <= partition_2 in terms of partial order

    For parition_1 and partition_2, two partitions of the same set, we call
    partition_1 <= partition_2 if and only if each block of partition_1 is
    contained in some block of partition_2

    Parameters
    ----------
        partition_1 (tuple[tuple[int]]) : the partition of lower order
        partition_2 (tuple[tuple[int]]) : the partition of higher order

    Returns
    -------
        bool : True if partition_1 <= partition_2

    Examples
    --------
        >>> from haarpy import partial_order
        >>> partition_1 = ((0, 1), (2, 3), (4,))
        >>> partition_2 = ((0, 1), (2, 3, 4))
        >>> partition_3 = ((0, 4), (1, 2, 3))
        >>> partial_order(partition_1, partition_2)
        True
        >>> partial_order(partition_1, partition_3)
        False
    """
    for part in partition_1:
        if all(not set(part).issubset(bigger_part) for bigger_part in partition_2):
            return False

    return True


@lru_cache
def meet_operation(
    partition_1: tuple[tuple[int, ...], ...], partition_2: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """Returns the greatest lower bound of the two input partitions

    For parition_1 and partition_2, two partitions of the same set,
    the meet operation yields the greatest lower bound of both partitions

    The meet operation symbol is ∧, for instance
    ((0, 1), (2, 3), (4,)) ∧ ((0, 1, 2), (3, 4)) = ((0, 1), (2,), (3,), (4,))

    Parameters
    ----------
        partition_1 (tuple[tuple[int]]) : partition of a set
        partition_2 (tuple[tuple[int]]) : partition of a set

    Returns
    -------
        tuple[tuple] : greatest lower bound

    Examples
    --------
        >>> from haarpy import meet_operation
        >>> partition_1 = ((0, 1), (2, 3), (4,))
        >>> partition_2 = ((0, 1, 2), (3, 4))
        >>> meet_operation(partition_1, partition_2)
        ((0, 1), (2,), (3,), (4,))
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
    partition_1: tuple[tuple[int, ...], ...], partition_2: tuple[tuple[int, ...], ...]
) -> tuple[tuple[int, ...], ...]:
    """Returns the least upper bound of the two input partitions

    For parition_1 and partition_2, two partitions of the same set,
    the join operation yields the least upper bound of both partitions

    The join operation symbol is ∨, for instance
    ((0, 1), (2,), (3, 4)) ∨ ((0, 2), (1,), (3,), (4,)) = ((0, 1, 2), (3, 4))

    Parameters
    ----------
        partition_1 (tuple[tuple[int]]) : partition of a set
        partition_2 (tuple[tuple[int]]) : partition of a set

    Returns
    ------
        tuple[tuple[int]] : least upper bound

    Examples
    --------
        >>> from haarpy import join_operation
        >>> partition_1 = ((0, 1), (2,), (3, 4))
        >>> partition_2 = ((0, 2), (1,), (3,), (4,))
        >>> join_operation(partition_1, partition_2)
        ((0, 1, 2), (3, 4))
    """
    parent = [
        {index for value in block1 for index, block2 in enumerate(partition_2) if value in block2}
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


def non_crossing_partitions(n: int, pair: bool = False) -> Iterator[tuple[tuple[int, ...], ...]]:
    """Yields non crossing partitions of [n] = {1,2,...,n}

    Parameters
    ----------
        n (int) : the size of the partitioned set [n]
        pair (bool) : True if limited to pair partitions

    Returns
    -------
        Iterator : yields the non-crossing partitions

    Raise
    -----
        TypeError : if n is not int
        ValueError : if n < 0 or if pair is True and n is odd

    Examples
    --------
        >>> from haarpy import non_crossing_partitions
        >>> for partition in non_crossing_partitions(3):
        >>>     print(partition)
        ((0,), (1,), (2,))
        ((0,), (1, 2))
        ((0, 2), (1,))
        ((0, 1), (2,))
        ((0, 1, 2),)
        >>> for partition in non_crossing_partitions(6, pair = True):
        >>>     print(partition)
        ((0, 5), (1, 4), (2, 3))
        ((0, 5), (1, 2), (3, 4))
        ((0, 3), (1, 2), (4, 5))
        ((0, 1), (2, 5), (3, 4))
        ((0, 1), (2, 3), (4, 5))
    """
    if not isinstance(n, int):
        raise TypeError
    if n < 0 or (pair and n % 2):
        raise ValueError

    def recursion_partitions(elements, active_partitions, inactive_partitions, pair):
        if not elements:
            if not pair or all(
                len(block) == 2 for block in active_partitions + inactive_partitions
            ):
                yield active_partitions + inactive_partitions
            return

        element = elements.pop()

        # pair partitions pruning
        if pair:
            open_blocks = sum(1 for b in active_partitions if len(b) == 1)
            if open_blocks > len(elements) + 1:
                elements.append(element)
                return

        active_partitions.append([element])
        yield from recursion_partitions(elements, active_partitions, inactive_partitions, pair)
        active_partitions.pop()

        size = len(active_partitions)
        for block in active_partitions[::-1]:
            if not pair or len(block) < 2:
                block.append(element)
                yield from recursion_partitions(
                    elements, active_partitions, inactive_partitions, pair
                )
                block.pop()

            # remove potential crossing
            inactive_partitions.append(active_partitions.pop())

        for _ in range(size):
            active_partitions.append(inactive_partitions.pop())

        elements.append(element)

    for partition in recursion_partitions(list(range(n - 1, -1, -1)), [], [], pair):
        yield tuple(sorted(map(tuple, partition), key=lambda x: x[0]))


@lru_cache
def is_crossing_partition(partition: tuple[tuple[int, ...], ...]) -> bool:
    """Checks if a given partition is crossing

    Parameters
    ----------
        partition (tuple[tuple[int]])) : partition of a set

    Returns
    -------
        bool : True if the partition is crossing, False otherwise

    Examples
    --------
        >>> from haarpy import is_crossing_partition
        >>> is_crossing_partition(((0,2,4), (1,3)))
        True
        >>> is_crossing_partition(((0,3,4), (1,2)))
        False
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


@lru_cache
def gram_matrix(
    partition_tuple: tuple[tuple[tuple[int, ...], ...], ...],
    group_dimension: Symbol,
) -> Matrix:
    """Generates the Gram matrix of a given input set of partitions

    Parameters
    ----------
        partition_tuple (tuple[tuple[tuple[int]])) : set of partitions
        group_dimension (Symbol) : the dimension of the underlying group

    Returns
    -------
        Matrix : the symbolic Gram matrix

    Raise
    -----
        TypeError : group_dimension is neither a symbol or an integer

    Examples
    --------
        >>> from haarpy import gram_matrix
        >>> from sympy import Symbol
        >>> n = Symbol('n')
        >>> pair_partition_tuple = (((0, 1), (2, 3)), ((0, 3), (1, 2)))
        >>> gram_matrix(pair_partition_tuple, n)
        Matrix([
        [n**2,    n],
        [   n, n**2]])
    """
    if not isinstance(group_dimension, (Expr, int)):
        raise TypeError

    return Matrix(
        tuple(
            tuple(
                group_dimension ** len(join_operation(partition_row, partition_col))
                for partition_col in partition_tuple
            )
            for partition_row in partition_tuple
        )
    )
