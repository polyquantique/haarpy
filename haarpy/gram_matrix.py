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
Gram and Weingarten matrices Python interface
"""

from functools import lru_cache
from sympy import Symbol, Matrix
from haarpy import join_operation


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
                group_dimension ** len(join_operation(partition_1, partition_2))
                for partition_1 in partition_tuple
            )
            for partition_2 in partition_tuple
        )
    )


@lru_cache
def weingarten_matrix(
    partition_tuple: tuple[tuple[tuple[int]]],
    group_dimension: Symbol,
) -> Matrix:
    """
    """
    return gram_matrix(partition_tuple, group_dimension).inv()


@lru_cache
def weingarten_matrix_unitary(degree: int, group_dimension: Symbol) -> Matrix:
    """
    """
    return


@lru_cache
def weingarten_matrix_orthogonal(degree: int, group_dimension: Symbol) -> Matrix:
    """
    """
    return


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


@lru_cache
def haar_integral_free_symmetric():
    return


@lru_cache
def haar_integral_free_orthogonal():
    return