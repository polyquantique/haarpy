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
from sympy import Expr, Symbol
from haarpy import non_crossing_partitions, gram_matrix
from ._utils import _simplify


@lru_cache
def _haar_integral_quantum(
    sequences: tuple[tuple[int, ...], ...],
    group_dimension: Symbol,
    pair: bool,
) -> Expr:
    """Returns the integral of a quantum group, either the free symmetric group or
    the free orthogonal group

    Parameters
    ----------
        sequences (tuple[tuple[int]]) : indices of matrix elements
        group_dimension (Symbol) : dimension of the quantum group
        pair (bool) : True for free orthogonal group, False for free symmetric group

    Returns
    -------
        Expr : integral under the Haar measure

    Raise
    -----
        TypeError : if the dimension is neither int nor Symbol
        ValueError : if sequences do not contain 2 tuples or if they are of different length 
        ValueError : 

    See Also
    --------
        gram_matrix
    """
    if not isinstance(group_dimension, (Expr, int)):
        raise TypeError
    if len(sequences) != 2 or len(sequences[0]) != len(sequences[1]):
        raise ValueError("Wrong tuple format")

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

    elligible_row_indices = (
        idx
        for idx, partition in enumerate(partition_tuple)
        if is_elligible_partition(partition, sequences[0])
    )
    elligible_col_indices = (
        idx
        for idx, partition in enumerate(partition_tuple)
        if is_elligible_partition(partition, sequences[1])
    )

    integral_gen = (
        weingarten_matrix[row_index, col_index]
        for row_index, col_index in product(elligible_row_indices, elligible_col_indices)
    )

    return sum(integral_gen) if isinstance(group_dimension, int) else _simplify(integral_gen)


@lru_cache
def haar_integral_free_symmetric(
    sequences: tuple[tuple[int, ...], ...],
    group_dimension: Symbol,
) -> Expr:
    """Returns the integral of the free symmetric group under the Haar measure

    Parameters
    ----------
        sequences (tuple[tuple[int]]) : indices of matrix elements
        group_dimension (Symbol) : dimension of the quantum group

    Returns
    -------
        Expr : integral under the Haar measure

    Examples
    --------
        >>> from sympy import Symbol
        >>> from haarpy import haar_integral_free_symmetric
        >>> d = Symbol("d")
        >>> sequences = ((0, 1, 2), (2, 1, 0))
        >>> haar_integral_free_symmetric(sequences, d)
        1/(d*(d - 2)*(d - 1))
        >>> haar_integral_free_symmetric(sequences, 4)
        1/24

    See Also
    --------
        _haar_integral_quantum
    """
    return _haar_integral_quantum(sequences, group_dimension, False)


@lru_cache
def haar_integral_free_orthogonal(
    sequences: tuple[tuple[int, ...], ...],
    group_dimension: Symbol,
) -> Expr:
    """Returns the integral of the free orthogonal group under the Haar measure

    Parameters
    ----------
        sequences (tuple[tuple[int]]) : indices of matrix elements
        group_dimension (Symbol) : dimension of the quantum group

    Returns
    -------
        Expr : integral under the Haar measure

    Examples
    --------
        >>> from sympy import Symbol
        >>> from haarpy import haar_integral_free_symmetric
        >>> d = Symbol("d")
        >>> sequences = ((0, 1, 1, 0), (0, 0, 1, 1))
        >>> haar_integral_free_symmetric(sequences, d)
        -1/(d*(d - 1)*(d + 1))
        >>> haar_integral_free_symmetric(sequences, 4)
        -1/60

    See Also
    --------
        _haar_integral_quantum
    """
    return _haar_integral_quantum(sequences, group_dimension, True)
