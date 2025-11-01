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

hand_calculated_weingarten = {
    21: 1/d**2/(d-1),
    22: -1/d/(d-1),
    23: 1/(d-1),

    31: 4/d**3/(d-1)/(d-2),
    32: -2/d**2/(d-1)/(d-2),
    33: 2/d/(d-1)/(d-2),
    34: 1/d/(d-1)/(d-2),
    35: -1/(d-1)/(d-2),
    36: d/(d-1)/(d-2),

    41: factor(3*(d+6))/d**4/(d-1)/(d-2)/(d-3),
    42: -(d+6)/d**3/(d-1)/(d-2)/(d-3),
    43: 1/d/(d-1)/(d-2)/(d-3),
    44: 6/d**2/(d-1)/(d-2)/(d-3),
    45: -6/d/(d-1)/(d-2)/(d-3),
    46: 3/d**2/(d-1)/(d-2)/(d-3),
    47: -1/d/(d-1)/(d-3),
    48: -1/d/(d-1)/(d-2)/(d-3),
    49: -2/d/(d-1)/(d-2)/(d-3),
    410: 2/(d-1)/(d-2)/(d-3),
    411: (d**2-3*d+1)/d/(d-1)/(d-2)/(d-3),
    412: 1/d/(d-2)/(d-3),
    413: -1/(d-2)/(d-3),
    414: (d+1)/d/(d-1)/(d-2)/(d-3),
    415: -(d+1)/(d-1)/(d-2)/(d-3),
    416: d*(d+1)/(d-1)/(d-2)/(d-3),
}

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
        ap.set_partitions(collection),
        ap.set_partitions(collection),
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
            for partition_3 in ap.set_partitions(collection)
            if (
                ap.partial_order(partition_1, partition_3)
                and ap.partial_order(partition_3, partition_2)
            )
        )
        
        assert convolution == kronecker


@pytest.mark.parametrize(
        "partition1, partition2, result_key",
        [
            (((0,), (1,)), ((0,), (1,)), 21),
            (((0,), (1,)), ((0,1),), 22),
            (((0,1),), ((0,1),), 23),

            (((0,), (1,), (2,)), ((0,), (1,), (2,)), 31),
            (((0,), (1,), (2,)), ((0,2), (1,)), 32),
            (((0,), (1,), (2,)), ((0,1,2),), 33),
            (((0,1,), (2,)), ((0,2), (1,)), 34),
            (((0,1,2),), ((0,1), (2,)), 35),
            (((0,1,2),), ((0,1,2),), 36),

            (((0,), (1,), (2,), (3,)), ((0,), (1,), (2,), (3,)), 41),
            (((0,), (1,2,), (3,)), ((0,), (1,), (2,), (3,)), 42),
            (((0,2), (1,3)), ((0,), (1,), (2,), (3,)), 43),
            (((0,1,3), (2,)), ((0,), (1,), (2,), (3,)), 44),
            (((0,1,2,3),), ((0,), (1,), (2,), (3,)), 45),

            (((0,1), (2,), (3,)), ((0,), (1,3), (2,)), 46),
            (((0,2), (1,), (3,)), ((0,), (1,3), (2,)), 43),

            (((0,3), (1,), (2,)), ((0,2), (1,3)), 48),
            (((0,3), (1,), (2,)), ((0,3), (1,2)), 47),

            (((0,), (1,), (2,3)), ((0,2,3), (1,)), 49),

            (((0,), (1,), (2,3)), ((0,1,2,3),), 410),

            (((0,2), (1,3)), ((0,3), (1,2)), 43),
            (((0,3), (1,2)), ((0,3), (1,2)), 411),

            (((0,3), (1,2)), ((0,), (1,2,3)), 412),

            (((0,3), (1,2)), ((0,1,2,3),), 413),

            (((0,1,2), (3,)), ((0,1,2), (3,)), 414),
            (((0,1,2), (3,)), ((0,1,2,3),), 415),

            (((0,1,2,3),), ((0,1,2,3),), 416),
        ]
)
def test_weingarten_centered_permutation_hand_calculated(partition1, partition2, result_key):
    "Test Weingarten centered permutation function against hand calculated cases"
    assert (
        ap.weingarten_centered_permutation(partition1, partition2, d) 
        == hand_calculated_weingarten[result_key]
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
            ap.set_partitions(tuple(i for i, _ in enumerate(row_indices))),
            ap.set_partitions(tuple(i for i, _ in enumerate(column_indices))),
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
        ((1,2,3), 12),
        ({3,4,5}, (3,4,5)),
    ]
)
def test_haar_integral_permutation_type_error(row_indices, column_indices):
    "test haar_integral_permutation_weingarten type error"
    with pytest.raises(TypeError):
        ap.haar_integral_permutation(row_indices, column_indices, d)


@pytest.mark.parametrize(
    "row_indices, column_indices",
    [
        ((1,2,3), (1,2,3,4)),
        ('abcd', 'abc'),
    ]
)
def test_haar_integral_permutation_value_error(row_indices, column_indices):
    "test haar_integral_permutation_weingarten value error"
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral_permutation(row_indices, column_indices, d)


@pytest.mark.parametrize(
    "row_indices, column_indices, result_dict",
    [
        ((1,2),(1,2), {21:1}),
        ((1,1),(2,2), {21:1, 22:2, 23:1}),
        ((1,1),(1,2), {21:1, 22:1}),

        ((1,2,3), (1,2,3), {31:1}),
        ((1,2,3), (1,2,2), {31:1, 32:1}),
        ((1,2,3), (1,1,1), {31:1, 32:3, 33:1}),
        ((1,2,2), (1,1,1), {31:1, 32:4, 33:1, 34:3, 35:1}),
        ((1,2,2), (2,2,1), {31:1, 32:2, 34:1}),
        ((1,1,1), (1,1,1), {31:1, 32:6, 33:2, 34:9, 35:6, 36:1}),

        ((1,2,3,4), (1,2,3,4), {41:1}),
        ((1,2,3,2), (1,1,1,1), {41:1, 42:7, 43:5, 44:4, 45:1, 46:4, 47:1, 48:2, 49:4, 410:1}),
        (
            (1,1,1,1),
            (1,1,1,1),
            {
                41:1,
                42:12,
                43:24,
                44:8,
                45:2,
                46:24,
                47:12,
                48:24,
                49:48,
                410:12,
                411:3,
                412:24,
                413:6,
                414:16,
                415:8,
                416:1,
            }
        ),
    ]
)
def test_haar_integral_centered_permutation_weingarten(row_indices, column_indices, result_dict):
    """Test haar integral for centered permutation matrices
    against the Weingarten sum as seen in Eq.(2.2)
    and (2.4) of `Collins and Nagatsu. Weingarten Calculus for Centered Random
    Permutation Matrices <https://arxiv.org/abs/2503.18453>`_
    """
    result = simplify(sum(value*hand_calculated_weingarten[key] for key, value in result_dict.items()))
    num, denum = fraction(result)
    assert (
        ap.haar_integral_centered_permutation(row_indices, column_indices, d)
        == factor(num)/factor(denum)
    )


@pytest.mark.parametrize(
    "row_indices, column_indices",
    [
        ((1,2,3), 12),
        ({3,4,5}, (3,4,5)),
    ]
)
def test_haar_integral_centered_permutation_type_error(row_indices, column_indices):
    "test haar_integral_centered_permutation_weingarten type error"
    with pytest.raises(TypeError):
        ap.haar_integral_centered_permutation(row_indices, column_indices, d)


@pytest.mark.parametrize(
    "row_indices, column_indices",
    [
        ((1,2,3), (1,2,3,4)),
        ('abcd', 'abc'),
    ]
)
def test_haar_integral_centered_permutation_value_error(row_indices, column_indices):
    "test haar_integral_centered_permutation_weingarten value error"
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral_centered_permutation(row_indices, column_indices, d)
