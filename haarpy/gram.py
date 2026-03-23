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
Gram and Weingarten matrices Python interface

References
----------
    [1]
"""

from functools import lru_cache
from sympy import Symbol, Matrix
from sympy.combinatorics import SymmetricGroup
from haarpy import join_operation, perfect_matchings, non_crossing_partitions, set_partitions


@lru_cache
def gram_matrix(
    partition_tuple: tuple[tuple[tuple[int]]],
    group_dimension: Symbol,
) -> Matrix:
    """ Generates the Gram matrix of a given input set of partitions

    Parameters
    ----------
        partition_tuple (tuple[tuple[tuple[int]]]) : a set of partitions
        group_dimension (Symbol) : the dimension of the underlying group

    Returns
    -------
        Matrix : the symbolic Gram matrix

    Raise
    -----
        TypeError :

    Examples
    --------
        >>> from sympy import Symbol
        >>> from haarpy import gram_matrix
        >>> d = Symbol("d")
        >>>
    """
    #
    return Matrix(
        tuple(
            tuple(
                group_dimension ** len(join_operation(partition_column, partition_row))
                for partition_column in partition_tuple
            )
            for partition_row in partition_tuple
        )
    )


@lru_cache
def weingarten_matrix_unitary(degree: int, group_dimension: Symbol) -> Matrix:
    """ """
    #RAISE ERROR IF DEGREE IS NOT INT GREATER THAN 0
    #RAISE ERROR IF GROUP IS NOT SYMBOL OR INT GREATER THAN O
    #RETURN THE SET OF PARTITIONS
    unitary_partitions = tuple(
        tuple((i, j) for i, j in enumerate(permutation(range(degree, 2 * degree))))
        for permutation in SymmetricGroup(degree).generate()
    )
    return gram_matrix(unitary_partitions, group_dimension).inv()


@lru_cache
def weingarten_matrix_orthogonal(degree: int, group_dimension: Symbol) -> Matrix:
    """ """
    orthogonal_partitions = tuple(
        partition for partition in perfect_matchings(tuple(i for i in range(degree)))
    )
    return gram_matrix(orthogonal_partitions, group_dimension).inv()


@lru_cache
def weingarten_matrix_free_symmetric(degree: int, group_dimension: Symbol) -> Matrix:
    """ """
    free_symmetric_partitions = tuple(partition for partition in non_crossing_partitions(degree))
    return gram_matrix(free_symmetric_partitions, group_dimension).inv()


@lru_cache
def weingarten_matrix_free_orthogonal(degree: int, group_dimension: Symbol) -> Matrix:
    """ """
    free_orthogonal_partitions = tuple(partition for partition in non_crossing_partitions(degree, pair = True))
    return gram_matrix(free_orthogonal_partitions, group_dimension).inv()
