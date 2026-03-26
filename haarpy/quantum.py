# Copyright 2026 Polyquantique

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
Quantum groups Python interface

References
----------
    [1] Banica, T., & Collins, B. (2007). Integration over compact quantum groups.
    Publications of the Research Institute for Mathematical Sciences, 43(2), 277-302.
"""

from functools import lru_cache
from itertools import product
from sympy import Expr, Symbol, fraction, together, cancel, factor
from haarpy import non_crossing_partitions, gram_matrix


@lru_cache
def haar_integral_quantum(
    sequences: tuple[tuple[int, ...], ...],
    group_dimension: Symbol,
    pair: bool,
) -> Expr:
    """ """
    # RAISE ERROR IF GROUP NOT EXPR OR INT
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")
    if len(sequences[0]) != len(sequences[1]):
        raise

    degree = len(sequences[0])
    partition_tuple = tuple(partition for partition in non_crossing_partitions(degree, pair))
    weingarten_matrix = gram_matrix(partition_tuple, group_dimension).inv()

    def is_elligible_partition(partition, sequence):
        block_values = tuple(sequence[block[0]] for block in partition)
        return all(
            sequence[i] == block_values[index]
            for index, block in enumerate(partition)
            for i in block
        )

    elligible_row_partitions = (
        partition
        for partition in partition_tuple
        if is_elligible_partition(partition, sequences[0])
    )
    elligible_col_partitions = (
        partition
        for partition in partition_tuple
        if is_elligible_partition(partition, sequences[1])
    )

    integral = sum(
        weingarten_matrix[row_index, col_index]
        for row_index, col_index in product(elligible_row_partitions, elligible_col_partitions)
    )

    if isinstance(group_dimension, Symbol):
        num, denum = fraction(cancel(together(integral)))
        integral = factor(num) / factor(denum)

    return integral


@lru_cache
def haar_integral_free_symmetric(
    sequences: tuple[tuple[int, ...], ...],
    group_dimension: Symbol,
) -> Expr:
    """ """
    return haar_integral_quantum(sequences, group_dimension, False)


@lru_cache
def haar_integral_free_orthogonal(
    sequences: tuple[tuple[int, ...], ...],
    group_dimension: Symbol,
) -> Expr:
    """ """
    return haar_integral_quantum(sequences, group_dimension, True)
