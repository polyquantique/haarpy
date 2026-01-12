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
"""

from functools import lru_cache
from sympy import Symbol, Matrix
from sympy.combinatorics import SymmetricGroup
from haarpy import join_operation, perfect_matchings


@lru_cache
def gram_matrix(
    partition_tuple: tuple[tuple[tuple[int]]],
    group_dimension: Symbol,
) -> Matrix:
    """
    """
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
    """
    """
    unitary_partitions = tuple(
        tuple(
            (i, j)
            for i, j in enumerate(permutation(range(degree,2*degree)))
        )
        for permutation in SymmetricGroup(degree).generate()
    )
    return gram_matrix(unitary_partitions, group_dimension).inv()


@lru_cache
def weingarten_matrix_orthogonal(degree: int, group_dimension: Symbol) -> Matrix:
    """
    """
    orthogonal_partitions = tuple(
        matching for matching in perfect_matchings(tuple(i for i in range(degree)))
    )
    return gram_matrix(orthogonal_partitions, group_dimension).inv()


@lru_cache
def weingarten_matrix_free_symmetric(degree: int, group_dimension: Symbol) -> Matrix:
    """
    """
    return


@lru_cache
def weingarten_matrix_free_orthogonal(degree: int, group_dimension: Symbol) -> Matrix:
    """
    """
    return
