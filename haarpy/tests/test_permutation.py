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
        (1,1),
    ]
)
def test_partial_order_true(partition1, partition2):
    "Partial orders  such that partition1 <= partition2"
    assert True


@pytest.mark.parametrize(
    "partition1, partition2",
    [
        (1,1),
    ]
)
def test_partial_order_true(partition1, partition2):
    "Partial orders  such that partition1 <= partition2"
    assert True



#test trivial case for mobius function on page 4

#test_mobuis_function_orthgonality Eq(2.1)

#Use corollary 2.2 to test_weingarten_permutation

#Use corollary 2.3 to test_weingarten_permutation