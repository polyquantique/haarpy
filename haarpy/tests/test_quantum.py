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
Quantum tests
"""

import pytest
from random import randint, seed
from itertools import product
from sympy import Symbol, diag, fraction, factor, sqrt, simplify
import haarpy as ap

seed(137)
d = Symbol('d')


@pytest.mark.parametrize(
    "sequences",
    [
        ((0,1), (0,0)),
        ((0,1,0), (0,1,1)),
        ((0,1,2), (0,1,1)),
        ((1,0,0), (0,1,2)),
        ((0,1,0), (0,1,2)),
        ((0,1,2), (0,1,0)),
        ((0,1,0,1), (0,1,0,0)),
        ((1,0,0,2), (0,1,2,1)),
        ((0,0,0,0), (0,1,2,3)),
        ((0,0,0,1), (0,1,2,3)),
        ((0,0,1,1), (0,1,2,3)),
        ((1,2,0,1), (0,1,2,3)),
        ((1,2,1,2), (1,2,0,1)),
        ((1,2,0,1), (1,2,1,2)),
    ],
)
def test_free_symmetric_trivially_zero(sequences):
    """Integrals are trivially 0 if two succesive matrix elements
    (including first and last) are on the same row or column
    """
    assert not ap.haar_integral_free_symmetric(sequences, d)
    assert not ap.haar_integral_free_symmetric(sequences, randint(5,20))


@pytest.mark.parametrize(
    "sequences",
    [
        ((0,1), (0,1)),
        ((0,1,2), (2,1,0)),
        ((0,1,0), (0,1,0)),
        ((0,1,2,3), (0,1,2,3)),
        ((0,1,0,1), (0,1,0,1)),
        ((0,1,2,3), (0,1,2,1)),
        ((0,1,2,1), (0,1,2,3)),
        ((0,1,0,1), (0,1,2,3)),
    ],
)
def test_free_symmetric_none_zero(sequences):
    "Test non-zero moments of free symmetric group"
    assert ap.haar_integral_free_symmetric(sequences, d)
    assert ap.haar_integral_free_symmetric(sequences, randint(5,20))


@pytest.mark.parametrize(
    "sequences",
    [
        ((0,1), (0,0)),
        ((0,1,0,1), (0,0,0,0)),
        ((1,0,0,1), (0,1,0,1)),
        ((0,1,2,3), (0,0,0,0)),
        ((0,0,1,1,2,2), (0,0,0,1,1,1)),
        ((0,1,2,0,1,2), (0,0,0,0,1,1)),
        ((1,0,1,0,2,2), (0,0,1,1,1,1)),
    ],
)
def test_free_orthogonal_trivially_zero(sequences):
    """Integrals are if crossing
    """
    assert not ap.haar_integral_free_orthogonal(sequences, d)
    assert not ap.haar_integral_free_orthogonal(sequences, randint(5,20))


@pytest.mark.parametrize(
    "sequences",
    [
        ((0,0), (0,0)),
        ((0,0,0,0), (0,1,1,0)),
        ((1,0,0,1), (0,1,1,0)),
        ((1,0,0,1), (0,0,1,1)),
        ((0,1,1,0,2,2), (0,0,0,0,0,0)),
        ((0,1,2,2,1,0), (0,0,1,1,2,2)),
    ],
)
def test_free_orthogonal_none_zero(sequences):
    "Test non-zero moments of free symmetric group"
    assert ap.haar_integral_free_orthogonal(sequences, d)
    assert ap.haar_integral_free_orthogonal(sequences, randint(5,20))



@pytest.mark.parametrize(
    "sequences",
    [
        ((0,1), (0,1)),
        ((0,1,2), (2,1,0)),
        ((0,1,0), (0,1,0)),
        ((0,1,2,3), (0,1,2,3)),
        ((0,1,0,1), (0,1,0,1)),
        ((0,1,2,3), (0,1,2,1)),
        ((0,1,2,1), (0,1,2,3)),
        ((0,1,0,1), (0,1,2,3)),
        ((0,1), (0,0)),
        ((0,1,0), (0,1,1)),
        ((0,1,2), (0,1,0)),
        ((0,1,0,1), (0,1,0,0)),
        ((1,0,0,2), (0,1,2,1)),
        ((0,0,0,0), (0,1,2,3)),
        ((1,2,1,2), (1,2,0,1)),
        ((1,2,0,1), (1,2,1,2)),
    ],
)
def test_free_quantum_relation(sequences):
    "Test relation between moments of free symmetric and free orthogonal"
    def partition_fattening(partition):
        def block_fattening(block):
            yield (2 * block[0], 2 * block[-1] + 1)
            if len(block) == 1:
                return
            for i, j in zip(block[:-1], block[1:]):
                yield (2 * i + 1, 2 * j)

        return tuple(
            sorted(
                (pair for block in partition for pair in block_fattening(block)), key=lambda x: x[0]
            )
        )
    
    degree = len(sequences[0])
    partition_tuple = tuple(partition for partition in ap.non_crossing_partitions(degree))

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

    fat_partitions = tuple(
        partition_fattening(partition)
        for partition in partition_tuple
    )

    d_sqrt = Symbol('n')
    weingarten_orthogonal_matrix = ap.gram_matrix(fat_partitions, d_sqrt**2).inv()
    diagonal_matrix = diag(
        *(
            d_sqrt ** (degree - 2*len(partition))
            for partition in partition_tuple
        )
    )
    weingarten_matrix = diagonal_matrix @ weingarten_orthogonal_matrix @ diagonal_matrix

    integral_gen = sum(
        weingarten_matrix[row_index, col_index]
        for row_index, col_index in product(elligible_row_indices, elligible_col_indices)
    )

    num, denum = fraction(simplify(integral_gen))
    integral = (
        factor(num.subs({d_sqrt : sqrt(sqrt(d))}))
        / factor(denum.subs({d_sqrt : sqrt(sqrt(d))}))
    )

    assert integral == ap.haar_integral_free_symmetric(sequences, d)