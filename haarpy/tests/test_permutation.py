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
from collections import Counter
from itertools import product
from sympy import bell
import haarpy as ap


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_size(size):
    "Assert the number of partitions is given by the Bell number"
    assert sum(1 for _ in ap.set_partition(tuple(range(size)))) == bell(size)


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_maximum_partition(size):
    "Assert that there is a single maximum parition"
    assert sum(
        1 for partition in ap.set_partition(tuple(range(size)))
        if len(partition) == 1
        and len(partition[0]) == size
    ) == 1


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_minimum_partition(size):
    "Assert that there is a single minimum partition"
    assert sum(
        1 for partition in ap.set_partition(tuple(range(size)))
        if len(partition) == size
        and all(len(part) == 1 for part in partition)
    ) == 1


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_unique(size):
    "Assert that all partitions are unique"
    partition_tuple = tuple(
        partition for partition in ap.set_partition(tuple(range(size)))
    )
    assert len(partition_tuple) == len(set(partition_tuple))


@pytest.mark.parametrize(
        "collection",
        [
            (set()),
            (dict()),
            (range(7)),
            (11),
        ]
)
def test_set_partition_type_error(collection):
    "Raise TypeError for wrong type input"
    with pytest.raises(
        TypeError,
        match = 'collection must be an indexable iterable'
    ):
        tuple(ap.set_partition(collection))


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_maximum_partition_in(size):
    "All partitions are contained within the maximal partition"
    maximum_partition = (tuple(range(size)),)
    assert all(
        ap.partial_order(partition, maximum_partition)
        for partition in ap.set_partition(tuple(range(size)))
    )


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_maximum_partition_out(size):
    "The maximal partition is contained within no partition but itself"
    maximum_partition = (tuple(range(size)),)
    assert sum(
        ap.partial_order(maximum_partition, partition)
        for partition in ap.set_partition(tuple(range(size)))
    ) == 1


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_minimum_partition_out(size):
    "No partition but itself is contained within the minimmal partition"
    minimum_partition = tuple((i,) for i in range(size))
    assert sum(
        ap.partial_order(partition, minimum_partition)
        for partition in ap.set_partition(tuple(range(size)))
    ) == 1


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_minimum_partition_in(size):
    "The minimal partition is contained within all partitions"
    minimum_partition = tuple((i,) for i in range(size))
    assert all(
        ap.partial_order(minimum_partition, partition)
        for partition in ap.set_partition(tuple(range(size)))
    )


@pytest.mark.parametrize(
    "partition1, partition2",
    [
        (((1,2),(3,4,5)), ((1,2),(3,4,5))),
        (((1,),(2,),(3,4,5)), ((1,2),(3,4,5))),
        (((1,2),(3,),(4,5)), ((1,2),(3,4,5))),
        (((3,2,1),(5,4),(8,),(9,7)), ((4,5,2,3,1,8),(7,9))),
    ]
)
def test_partial_order_true(partition1, partition2):
    "Partial orders such that partition1 <= partition2 is True"
    assert ap.partial_order(partition1, partition2)


@pytest.mark.parametrize(
    "partition1, partition2",
    [
        (((1,2,3),(4,5)), ((1,2),(3,4,5))),
        (((1,),(2,3),(4,5)), ((1,2),(3,4,5))),
        (((1,3),(2,4,5)), ((1,2),(3,4,5))),
        (((1,2),(3,4,5)), ((1,2),(3,),(4,5))),
        (((3,2,1),(5,4),(8,),(9,7)), ((5,2,3,1,8),(4,),(7,9))),
    ]
)
def test_partial_order_false(partition1, partition2):
    "Partial orders such that partition1 <= partition2 is False"
    assert not ap.partial_order(partition1, partition2)


@pytest.mark.parametrize(
    "partition1, partition2",
    [
        (((1,1),(3,4,5)), ((1,1),(3,4,5))),
        (((1,),(2,),(3,2,5)), ((1,2),(3,2,5))),
        (((1,2),(3,),(4,4)), ((1,2),(3,5,5))),
        (((3,2,1),(5,4),(8,),(9,7)), ((4,5,2,3,1,1),(7,9))),
    ]
)
def test_partial_order_value_error(partition1, partition2):
    "Partial orders value error for repetition"
    with pytest.raises(
        ValueError,
        match = "The partitions must be composed of unique elements",
    ):
        ap.partial_order(partition1, partition2)


@pytest.mark.parametrize("size" , range(1,7))
def test_meet_operation_minimum_partition(size):
    "Meet operation with minimum partition returns minimum partition"
    minimum_partition = tuple((i,) for i in range(size))
    minimum_counter = Counter(minimum_partition)
    for partition in ap.set_partition(tuple(range(size))):
        assert (
            Counter(ap.meet_operation(partition, minimum_partition))
            == minimum_counter
        )


@pytest.mark.parametrize("size" , range(1,7))
def test_meet_operation_maximum_partition(size):
    "Meet operation with maximum partition returns the same partition"
    maximum_partition = (tuple(range(size)),)
    for partition in ap.set_partition(tuple(range(size))):
        assert (
            ap.meet_operation(partition, maximum_partition)
            == partition
        )


@pytest.mark.parametrize(
    "partition1, partition2, expected_result",
    [
        (((0,1,3), (2,4,5), (6,)), ((0,1), (2,3,4), (5,6)), ((0,1), (2,4), (3,), (5,), (6,))),
        (((0,4,5), (1,3,2), (6,7)), ((2,1,3,4), (0,5,6,7)), ((0,5), (1,2,3), (4,), (6,7))),
    ]
)
def test_meet_operation_additional(partition1, partition2, expected_result):
    "additional hand calculated tests for the meet operation"
    meet_partition = Counter(
        tuple(sorted(block))
        for block in ap.meet_operation(partition1, partition2)
    )
    meet_expected = Counter(
        tuple(sorted(block))
        for block in expected_result
    )
    assert meet_partition == meet_expected


@pytest.mark.parametrize("size" , range(1,7))
def test_join_operation_minimum_partition(size):
    "Join operation with minimum partition returns same partition"
    minimum_partition = tuple((i,) for i in range(size))
    for partition in ap.set_partition(tuple(range(size))):
        assert (
            ap.join_operation(partition, minimum_partition)
            == partition
        )


@pytest.mark.parametrize("size" , range(1,7))
def test_join_operation_maximum_partition(size):
    "Join operation with maximum partition returns maximum partition"
    maximum_partition = (tuple(range(size)),)
    for partition in ap.set_partition(tuple(range(size))):
        assert (
            ap.join_operation(partition, maximum_partition)
            == maximum_partition
        )


@pytest.mark.parametrize(
    "partition1, partition2, expected_result",
    [
        (((0,1), (3,), (2,4), (5,), (6,)), ((0,), (1,5), (2,3), (4,6)), ((0,1,5),(2,3,4,6))),
        (((1,2), (0,4,5), (3,)), ((5,), (4,3), (1,2), (0,)), ((0,3,4,5), (1,2))),
    ]
)
def test_join_operation_additional(partition1, partition2, expected_result):
    "additional hand calculated tests for the join operation"
    join_partition = Counter(
        tuple(sorted(block))
        for block in ap.join_operation(partition1, partition2)
    )
    join_expected = Counter(
        tuple(sorted(block))
        for block in expected_result
    )
    assert join_partition == join_expected


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


#test trivial case for mobius function on page 4

#test_mobuis_function_orthgonality Eq(2.1)

#test Eq(2.4) against Eq(2.2)

#Use corollary 2.2 to test_weingarten_permutation

#Use corollary 2.3 to test_weingarten_permutation