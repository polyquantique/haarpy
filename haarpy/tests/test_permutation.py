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
Permutation matrices tests
"""

import pytest
from math import factorial
from itertools import product
from sympy import Symbol, simplify, factor, fraction
import haarpy as ap

d = Symbol('d')


@pytest.mark.parametrize("size", range(1,7))
def test_mobius_function_trivial(size):
    """Test the first trivial relation for the Mobius funciton as seen
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_
    """
    minimum_partition = tuple((i,) for i in range(size))
    maximum_partition = (tuple(range(size)),)
    assert (
        ap.mobius_function(
            minimum_partition,
            maximum_partition,
        )
        == (-1)**(size-1) * factorial(size-1)
    )
    

@pytest.mark.parametrize("size", range(1,7))
def test_mobius_inversion_formula(size):
    """Test Mobius inversion formula
    as seen in `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_
    """
    collection = tuple(range(size))
    for partition_1, partition_2 in product(
        ap.set_partition(collection),
        ap.set_partition(collection),
    ):
        kronecker = int(partition_1 == partition_2)
        convolution = sum(
            ap.mobius_function(
                partition_1,
                partition_3,
            )
            * int(
                ap.partial_order(
                    partition_3,
                    partition_2
                )
            )
            for partition_3 in ap.set_partition(collection)
            if (
                ap.partial_order(partition_1, partition_3)
                and ap.partial_order(partition_3, partition_2)
            )
        )
        
        assert convolution == kronecker


@pytest.mark.parametrize(
        "partition1, partition2",
        [
            (((0,1),(2,3)), ((0,2),(1,3))),
        ]
)
def test_weingarten_permutation_hand_calculated(partition1, partition2):
    "Test Weingarten permutation function against hand calculated cases"
    assert False


@pytest.mark.parametrize(
        "partition1, partition2, result",
        [
            (((0,), (1,)), ((0,), (1,)), 1/d**2/(d-1)),
            (((0,), (1,)), ((0,1),), -1/d/(d-1)),
            (((0,1),), ((0,1),), 1/(d-1)),

            (((0,), (1,), (2,)), ((0,), (1,), (2,)), 4/d**3/(d-1)/(d-2)),
            (((0,), (1,), (2,)), ((0,2), (1,)), -2/d**2/(d-1)/(d-2)),
            (((0,), (1,), (2,)), ((0,1,2),), 2/d/(d-1)/(d-2)),
            (((0,1,), (2,)), ((0,2), (1,)), 1/d/(d-1)/(d-2)),
            (((0,1,2),), ((0,1), (2,)), -1/(d-1)/(d-2)),
            (((0,1,2),), ((0,1,2),), d/(d-1)/(d-2)),

            (((0,), (1,), (2,), (3,)), ((0,), (1,), (2,), (3,)), factor(3*(d+6))/d**4/(d-1)/(d-2)/(d-3)),
            (((0,), (1,2,), (3,)), ((0,), (1,), (2,), (3,)), -(d+6)/d**3/(d-1)/(d-2)/(d-3)),
            (((0,2), (1,3)), ((0,), (1,), (2,), (3,)), 1/d/(d-1)/(d-2)/(d-3)),
            (((0,1,3), (2,)), ((0,), (1,), (2,), (3,)), 6/d**2/(d-1)/(d-2)/(d-3)),
            (((0,1,2,3),), ((0,), (1,), (2,), (3,)), -6/d/(d-1)/(d-2)/(d-3)),

            (((0,1), (2,), (3,)), ((0,), (1,3), (2,)), 3/d**2/(d-1)/(d-2)/(d-3)),
            (((0,2), (1,), (3,)), ((0,), (1,3), (2,)), 1/d/(d-1)/(d-2)/(d-3)),

            (((0,3), (1,), (2,)), ((0,2), (1,3)), -1/d/(d-1)/(d-2)/(d-3)),
            (((0,3), (1,), (2,)), ((0,3), (1,2)), -1/d/(d-1)/(d-3)),

            (((0,), (1,), (2,3)), ((0,2,3), (1,)), -2/d/(d-1)/(d-2)/(d-3)),

            (((0,), (1,), (2,3)), ((0,1,2,3),), 2/(d-1)/(d-2)/(d-3)),

            (((0,), (1,), (2,3)), ((0,2,3), (1,)), -2/d/(d-1)/(d-2)/(d-3)),

            (((0,2), (1,3)), ((0,3), (1,2)), 1/d/(d-1)/(d-2)/(d-3)),
            (((0,3), (1,2)), ((0,3), (1,2)), (d**2-3*d+1)/d/(d-1)/(d-2)/(d-3)),

            (((0,3), (1,2)), ((0,), (1,2,3)), 1/d/(d-2)/(d-3)),

            (((0,3), (1,2)), ((0,1,2,3),), -1/(d-2)/(d-3)),

            (((0,1,2), (3,)), ((0,1,2), (3,)), (d+1)/d/(d-1)/(d-2)/(d-3)),
            (((0,1,2), (3,)), ((0,1,2,3),), -(d+1)/(d-1)/(d-2)/(d-3)),

            (((0,1,2,3),), ((0,1,2,3),), d*(d+1)/(d-1)/(d-2)/(d-3)),
        ]
)
def test_weingarten_centered_permutation_hand_calculated(partition1, partition2, result):
    "Test Weingarten centered permutation function against hand calculated cases"
    assert ap.weingarten_centered_permutation(partition1, partition2, d) == result


@pytest.mark.parametrize(
    "row_indices, column_indices",
    [
        ((1,2,3,4),(1,2,3,4)),
        ((3,2,2,1),(2,2,1,3)),
        ((3,2,2,1),(3,2,2,1)),
        ((3,2,2,1,2),(3,2,2,1,2)),
        ((3,3,2,2,1,2),(3,3,2,2,1,2)),
        ((3,3,2,2,3,2),(3,3,2,2,3,2)),
        ((3,3,2,2,3,2),(3,3,2,2,1,2)),
    ]
)
def test_haar_integral_permutation_weingarten(row_indices, column_indices):
    """Test haar integral for permutation matrices against the Weingarten
    sum as seen in Eq.(2.2) and (2.4) of `Collins and Nagatsu. Weingarten
    Calculus for Centered Random Permutation Matrices <https://arxiv.org/abs/2503.18453>`_
    """
    partition_row = tuple(
        tuple(index for index, value in enumerate(row_indices) if value == unique)
        for unique in set(row_indices)
    )
    partition_column = tuple(
        tuple(index for index, value in enumerate(column_indices) if value == unique)
        for unique in set(column_indices)
    )

    weingarten_integral = sum(
        ap.weingarten_permutation(
            partition_sigma,
            partition_tau,
            d,
        )
        for partition_sigma, partition_tau in product(
            ap.set_partition(tuple(i for i, _ in enumerate(row_indices))),
            ap.set_partition(tuple(i for i, _ in enumerate(column_indices))),
        )
        if ap.partial_order(partition_sigma, partition_row)
        and ap.partial_order(partition_tau, partition_column)
    )

    num, denum = fraction(simplify(weingarten_integral))
    weingarten_integral = factor(num)/factor(denum)

    assert (
        ap.haar_integral_permutation(row_indices, column_indices, d)
        == weingarten_integral
    )


@pytest.mark.parametrize(
    "row_indices, column_indices",
    [
        ((1,2,3,4),(1,2,3,4)),
        ((3,2,2,1),(2,2,1,3)),
        ((3,2,2,1),(3,2,2,1)),
        ((3,2,2,1,2),(3,2,2,1,2)),
        ((3,3,2,2,1,2),(3,3,2,2,1,2)),
        ((3,3,2,2,3,2),(3,3,2,2,3,2)),
        ((3,3,2,2,3,2),(3,3,2,2,1,2)),
    ]
)
def test_haar_integral_centered_permutation_weingarten(row_indices, column_indices):
    """Test haar integral for centered permutation matrices
    against the Weingarten sum as seen in Eq.(2.2)
    and (2.4) of `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_
    """
    assert False
