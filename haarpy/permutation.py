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

from typing import Union
from functools import lru_cache
from math import factorial, prod
from itertools import product
from collections.abc import Sequence
from sympy import Symbol, simplify, binomial, factor, fraction
from haarpy import Partition, set_partition


@lru_cache
def mobius_function(
    first_partition: Union[Partition, Sequence[Sequence[int]]],
    second_partition: Union[Partition, Sequence[Sequence[int]]],
) -> int:
    if isinstance(first_partition, Sequence) and isinstance(second_partition, Sequence):
        first_partition, second_partition = Partition(*first_partition), Partition(*second_partition)
    elif not (isinstance(first_partition, Partition) and isinstance(second_partition, Partition)):
        raise TypeError("Partitions must be either an instance of haarpy.Partition or Sequence.")
    
    return


@lru_cache
def weingarten_permutation(
    first_partition: Union[Partition, Sequence[Sequence[int]]],
    second_partition: Union[Partition, Sequence[Sequence[int]]],
    dimension: Symbol,
) -> Symbol:
    if isinstance(first_partition, Sequence) and isinstance(second_partition, Sequence):
        first_partition, second_partition = Partition(*first_partition), Partition(*second_partition)
    elif not (isinstance(first_partition, Partition) and isinstance(second_partition, Partition)):
        raise TypeError("Partitions must be either an instance of haarpy.Partition or Sequence.")
    
    return


@lru_cache
def weingarten_centered_permutation(
    first_partition: Union[Partition, Sequence[Sequence[int]]],
    second_partition: Union[Partition, Sequence[Sequence[int]]],
    dimension: Symbol,
) -> Symbol:
    if isinstance(first_partition, Sequence) and isinstance(second_partition, Sequence):
        first_partition, second_partition = Partition(*first_partition), Partition(*second_partition)
    elif not (isinstance(first_partition, Partition) and isinstance(second_partition, Partition)):
        raise TypeError("Partitions must be either an instance of haarpy.Partition or Sequence.")
    
    return


@lru_cache
def haar_integral_permutation(
    row_indices: Sequence[int],
    column_indices: Sequence[int],
    dimension: Symbol,
) -> Symbol:
    if not (isinstance(row_indices, Sequence) and isinstance(column_indices, Sequence)):
        raise TypeError
    
    if len(row_indices) != len(column_indices):
        raise ValueError("Wrong tuple format")
    
    return


@lru_cache
def haar_integral_centered_permutation(
    row_indices: Sequence[int],
    column_indices: Sequence[int],
    dimension: Symbol,
) -> Symbol:
    if not (isinstance(row_indices, Sequence) and isinstance(column_indices, Sequence)):
        raise TypeError
    
    if len(row_indices) != len(column_indices):
        raise ValueError("Wrong tuple format")
    return