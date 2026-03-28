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
Partition tests
"""

import pytest
from collections import Counter
from random import seed, choice
from sympy import bell, factorial2, catalan, Symbol, factor, fraction
from sympy.combinatorics import SymmetricGroup
import haarpy as ap
from haarpy._utils import _simplify

seed(137)
d = Symbol("d")


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_size(size):
    "Assert the number of partitions is given by the Bell number"
    assert sum(1 for _ in ap.set_partitions(tuple(range(size)))) == bell(size)


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_maximum_partition(size):
    "Assert that there is a single maximum partition"
    assert (
        sum(
            1
            for partition in ap.set_partitions(tuple(range(size)))
            if len(partition) == 1 and len(partition[0]) == size
        )
        == 1
    )


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_minimum_partition(size):
    "Assert that there is a single minimum partition"
    assert (
        sum(
            1
            for partition in ap.set_partitions(tuple(range(size)))
            if len(partition) == size and all(len(part) == 1 for part in partition)
        )
        == 1
    )


@pytest.mark.parametrize("size", range(1, 7))
def test_set_partition_unique(size):
    "Assert that all partitions are unique"
    partition_tuple = tuple(partition for partition in ap.set_partitions(tuple(range(size))))
    assert len(partition_tuple) == len(set(partition_tuple))


@pytest.mark.parametrize(
    "collection",
    [
        (set()),
        (dict()),
        (range(7)),
        (11),
        ([0, 1, 2, 3]),
    ],
)
def test_set_partition_type_error(collection):
    "Raise TypeError for wrong type input"
    with pytest.raises(
        TypeError,
        match="collection must be a tuple",
    ):
        tuple(ap.set_partitions(collection))


@pytest.mark.parametrize("size", range(2, 14))
def test_pair_partitions_order(size):
    "test size of perfect matching partitions"
    assert sum(1 for _ in ap.pair_partitions(tuple(range(size)))) == (
        factorial2(size - 1) if not size % 2 else 0
    )


@pytest.mark.parametrize(
    "seed",
    [
        [1, 2],
        "a",
        range(4),
        12,
    ],
)
def test_pair_partitions_type_error(seed):
    "test perfect matching type error"
    with pytest.raises(
        TypeError,
        match="seed must be a tuple",
    ):
        [_ for _ in ap.pair_partitions(seed)]


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_maximum_partition_in(size):
    "All partitions are contained within the maximal partition"
    maximum_partition = (tuple(range(size)),)
    assert all(
        ap.partial_order(partition, maximum_partition)
        for partition in ap.set_partitions(tuple(range(size)))
    )


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_maximum_partition_out(size):
    "The maximal partition is contained within no partition but itself"
    maximum_partition = (tuple(range(size)),)
    assert (
        sum(
            ap.partial_order(maximum_partition, partition)
            for partition in ap.set_partitions(tuple(range(size)))
        )
        == 1
    )


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_minimum_partition_out(size):
    "No partition but itself is contained within the minimmal partition"
    minimum_partition = tuple((i,) for i in range(size))
    assert (
        sum(
            ap.partial_order(partition, minimum_partition)
            for partition in ap.set_partitions(tuple(range(size)))
        )
        == 1
    )


@pytest.mark.parametrize("size", range(1, 7))
def test_partial_order_minimum_partition_in(size):
    "The minimal partition is contained within all partitions"
    minimum_partition = tuple((i,) for i in range(size))
    assert all(
        ap.partial_order(minimum_partition, partition)
        for partition in ap.set_partitions(tuple(range(size)))
    )


@pytest.mark.parametrize(
    "partition1, partition2",
    [
        (((1, 2), (3, 4, 5)), ((1, 2), (3, 4, 5))),
        (((1,), (2,), (3, 4, 5)), ((1, 2), (3, 4, 5))),
        (((1, 2), (3,), (4, 5)), ((1, 2), (3, 4, 5))),
        (((3, 2, 1), (5, 4), (8,), (9, 7)), ((4, 5, 2, 3, 1, 8), (7, 9))),
    ],
)
def test_partial_order_true(partition1, partition2):
    "Partial orders such that partition1 <= partition2 is True"
    assert ap.partial_order(partition1, partition2)


@pytest.mark.parametrize(
    "partition1, partition2",
    [
        (((1, 2, 3), (4, 5)), ((1, 2), (3, 4, 5))),
        (((1,), (2, 3), (4, 5)), ((1, 2), (3, 4, 5))),
        (((1, 3), (2, 4, 5)), ((1, 2), (3, 4, 5))),
        (((1, 2), (3, 4, 5)), ((1, 2), (3,), (4, 5))),
        (((3, 2, 1), (5, 4), (8,), (9, 7)), ((5, 2, 3, 1, 8), (4,), (7, 9))),
    ],
)
def test_partial_order_false(partition1, partition2):
    "Partial orders such that partition1 <= partition2 is False"
    assert not ap.partial_order(partition1, partition2)


@pytest.mark.parametrize("size", range(1, 7))
def test_meet_operation_minimum_partition(size):
    "Meet operation with minimum partition returns minimum partition"
    minimum_partition = tuple((i,) for i in range(size))
    minimum_counter = Counter(minimum_partition)
    for partition in ap.set_partitions(tuple(range(size))):
        assert Counter(ap.meet_operation(partition, minimum_partition)) == minimum_counter


@pytest.mark.parametrize("size", range(1, 7))
def test_meet_operation_maximum_partition(size):
    "Meet operation with maximum partition returns the same partition"
    maximum_partition = (tuple(range(size)),)
    for partition in ap.set_partitions(tuple(range(size))):
        assert ap.meet_operation(partition, maximum_partition) == partition


@pytest.mark.parametrize(
    "partition1, partition2, expected_result",
    [
        (
            ((0, 1, 3), (2, 4, 5), (6,)),
            ((0, 1), (2, 3, 4), (5, 6)),
            ((0, 1), (2, 4), (3,), (5,), (6,)),
        ),
        (
            ((0, 4, 5), (1, 3, 2), (6, 7)),
            ((2, 1, 3, 4), (0, 5, 6, 7)),
            ((0, 5), (1, 2, 3), (4,), (6, 7)),
        ),
    ],
)
def test_meet_operation_additional(partition1, partition2, expected_result):
    "additional hand calculated tests for the meet operation"
    meet_partition = Counter(
        tuple(sorted(block)) for block in ap.meet_operation(partition1, partition2)
    )
    meet_expected = Counter(tuple(sorted(block)) for block in expected_result)
    assert meet_partition == meet_expected


@pytest.mark.parametrize("size", range(5, 10))
def test_meet_operation_symmetry(size):
    "Test the meet operation yields the same output both ways"
    sample_size = int(1e5)
    partition_tuple = tuple(partition for partition in ap.set_partitions(tuple(range(size))))

    sample_partition_1 = (choice(partition_tuple) for _ in range(sample_size))
    sample_partition_2 = (choice(partition_tuple) for _ in range(sample_size))

    for party1, party2 in zip(sample_partition_1, sample_partition_2):
        assert ap.meet_operation(party1, party2) == ap.meet_operation(party2, party1)


@pytest.mark.parametrize("size", range(5, 10))
def test_meet_operation_size(size):
    "Test the meet operation yields a partition with correct format"
    sample_size = int(1e5)
    partition_tuple = tuple(partition for partition in ap.set_partitions(tuple(range(size))))

    sample_partition_1 = (choice(partition_tuple) for _ in range(sample_size))
    sample_partition_2 = (choice(partition_tuple) for _ in range(sample_size))

    for party1, party2 in zip(sample_partition_1, sample_partition_2):
        flatten_joined_partition = tuple(
            value for block in ap.meet_operation(party1, party2) for value in block
        )

        assert len(flatten_joined_partition) == size and all(
            i in flatten_joined_partition for i in range(size)
        )


@pytest.mark.parametrize("size", range(1, 7))
def test_join_operation_minimum_partition(size):
    "Join operation with minimum partition returns same partition"
    minimum_partition = tuple((i,) for i in range(size))
    for partition in ap.set_partitions(tuple(range(size))):
        assert ap.join_operation(partition, minimum_partition) == partition


@pytest.mark.parametrize("size", range(1, 7))
def test_join_operation_maximum_partition(size):
    "Join operation with maximum partition returns maximum partition"
    maximum_partition = (tuple(range(size)),)
    for partition in ap.set_partitions(tuple(range(size))):
        assert ap.join_operation(partition, maximum_partition) == maximum_partition


@pytest.mark.parametrize(
    "partition1, partition2, expected_result",
    [
        (
            ((0, 1), (3,), (2, 4), (5,), (6,)),
            ((0,), (1, 5), (2, 3), (4, 6)),
            ((0, 1, 5), (2, 3, 4, 6)),
        ),
        (((1, 2), (0, 4, 5), (3,)), ((5,), (4, 3), (1, 2), (0,)), ((0, 3, 4, 5), (1, 2))),
    ],
)
def test_join_operation_additional(partition1, partition2, expected_result):
    "additional hand calculated tests for the join operation"
    join_partition = Counter(
        tuple(sorted(block)) for block in ap.join_operation(partition1, partition2)
    )
    join_expected = Counter(tuple(sorted(block)) for block in expected_result)
    assert join_partition == join_expected


@pytest.mark.parametrize("size", range(5, 10))
def test_join_operation_symmetry(size):
    "Test the join operation yields the same output both ways"
    sample_size = int(1e5)
    partition_tuple = tuple(partition for partition in ap.set_partitions(tuple(range(size))))

    sample_partition_1 = (choice(partition_tuple) for _ in range(sample_size))
    sample_partition_2 = (choice(partition_tuple) for _ in range(sample_size))

    for party1, party2 in zip(sample_partition_1, sample_partition_2):
        assert ap.join_operation(party1, party2) == ap.join_operation(party2, party1)


@pytest.mark.parametrize("size", range(5, 10))
def test_join_operation_size(size):
    "Test the join operation yields a partition with correct format"
    sample_size = int(1e5)
    partition_tuple = tuple(partition for partition in ap.set_partitions(tuple(range(size))))

    sample_partition_1 = (choice(partition_tuple) for _ in range(sample_size))
    sample_partition_2 = (choice(partition_tuple) for _ in range(sample_size))

    for party1, party2 in zip(sample_partition_1, sample_partition_2):
        flatten_joined_partition = tuple(
            value for block in ap.join_operation(party1, party2) for value in block
        )

        assert len(flatten_joined_partition) == size and all(
            i in flatten_joined_partition for i in range(size)
        )


@pytest.mark.parametrize("degree", range(10))
def test_non_crossing_partition_size(degree):
    "assert the number of non-crossing partitions coincides with the catatlan number"
    assert sum(1 for _ in ap.non_crossing_partitions(degree)) == catalan(degree)


@pytest.mark.parametrize("degree", range(6))
def test_non_crossing_partition_repetition(degree):
    "assert that non non-crossing partitions are repeated"
    assert len({partition for partition in ap.non_crossing_partitions(degree)}) == catalan(degree)


@pytest.mark.parametrize("degree", range(10))
def test_non_crossing_partition_is_non_crossing(degree):
    "assert that all non-crossing partitions are non-crossing"
    for partition in ap.non_crossing_partitions(degree):
        assert not ap.is_crossing_partition(partition)


@pytest.mark.parametrize("degree", range(0, 10, 2))
def test_non_crossing_partition_size(degree):
    "assert the number of non-crossing pair partitions coincides with the catatlan number"
    assert sum(1 for _ in ap.non_crossing_partitions(degree, pair=True)) == catalan(degree // 2)


@pytest.mark.parametrize("degree", range(0, 8, 2))
def test_non_crossing_pair_partition_repetition(degree):
    "assert that non non-crossing partitions are repeated"
    assert len(
        {partition for partition in ap.non_crossing_partitions(degree, pair=True)}
    ) == catalan(degree // 2)


@pytest.mark.parametrize("degree", range(0, 10, 2))
def test_non_crossing_pair_partition_is_non_crossing(degree):
    "assert that all non-crossing partitions are non-crossing"
    for partition in ap.non_crossing_partitions(degree, pair=True):
        assert not ap.is_crossing_partition(partition)


@pytest.mark.parametrize("degree", range(10))
def test_non_crossing_partition_bijection(degree):
    "test the bijection between non-crossing partitions and pair partitions"

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

    bijection = {partition_fattening(partition) for partition in ap.non_crossing_partitions(degree)}

    assert {
        pair_partition for pair_partition in ap.non_crossing_partitions(2 * degree, pair=True)
    } == bijection


@pytest.mark.parametrize(
    "partition",
    [
        ((0, 4), (1,), (2,), (3,)),
        ((0, 5), (1, 4), (2, 3)),
        ((0, 3), (1, 2), (4, 7), (5, 6)),
        ((0, 8), (1, 2), (3, 7), (4,), (5, 6)),
        ((0, 11), (1, 2), (3, 7, 8), (4, 6), (5,), (9, 10)),
    ],
)
def test_crossing_partition_false(partition):
    "Non crossing partitions"
    assert not ap.is_crossing_partition(partition)


@pytest.mark.parametrize(
    "partition",
    [
        ((0, 4), (1,), (2, 5), (3,)),
        ((0, 3, 6), (1, 5), (2, 4)),
        ((0, 7, 8), (1, 2), (3, 6), (4, 9), (5,)),
        ((0, 5), (1, 2), (3, 4), (6, 8), (7, 9, 10)),
        ((0, 7), (1,), (2, 3), (4,), (5, 12), (6,), (8, 9), (10, 11)),
        ((0, 11, 15), (1, 2), (3, 7), (4,), (5, 6), (8, 9), (10, 12), (13, 14)),
    ],
)
def test_crossing_partition_true(partition):
    "Crossing partitions"
    assert ap.is_crossing_partition(partition)


@pytest.mark.parametrize("degree", range(1,4))
def test_gram_matrix_weingarten_unitary(degree):
    "Asserts that the inverse of the Gram matrix yield the unitary Weingarten functions"
    partition_tuple = tuple(
        tuple(
            (i, j)
            for i, j in enumerate(perm(range(degree,2*degree)))
        )
        for perm in SymmetricGroup(degree).generate()
    )
    weingarten_matrix = ap.gram_matrix(partition_tuple, d).inv()

    for conjugacy in SymmetricGroup(degree).conjugacy_classes():
        representative = conjugacy.pop()
        partition = tuple(
            (idx, degree + representative.array_form.index(idx))
            for idx, _ in enumerate(representative.array_form)
        )

        col = partition_tuple.index(partition)
        num, denum = fraction(weingarten_matrix[0, col])
        weingarten = factor(num) / factor(denum)
        assert weingarten == ap.weingarten_unitary(representative, d)


@pytest.mark.parametrize("half_degree", range(1,3))
def test_gram_matrix_weingarten_orthogonal(half_degree):
    "Asserts that the inverse of the Gram matrix yield the orthogonal Weingarten functions"
    partition_tuple = tuple(
        perfect for perfect in ap.pair_partitions(tuple(i for i in range(2 * half_degree)))
    )
    weingarten_matrix = ap.gram_matrix(partition_tuple, d).inv()

    already_tested = set()
    for conjugacy in SymmetricGroup(2 * half_degree).conjugacy_classes():
        representative = conjugacy.pop()
        partition = (
            (representative.array_form.index(2*idx), representative.array_form.index(2*idx+1))
            for idx, _ in enumerate(representative.array_form[::2])
        )
        partition = sorted((map(sorted, partition)))
        partition = tuple(map(tuple, partition))

        if partition in already_tested:
            continue
        already_tested.add(partition)

        col = partition_tuple.index(partition)
        num, denum = fraction(weingarten_matrix[0, col])
        weingarten = factor(num) / factor(denum)
        assert weingarten == ap.weingarten_orthogonal(representative, d)


@pytest.mark.parametrize(
    "dimension",
    [
        'a',
        0.1,
    ],
)
def test_gram_matrix_type_error(dimension):
    "Asserts typeError for wrong dimension type"
    partition_tuple = tuple(
        perfect for perfect in ap.pair_partitions(tuple(i for i in range(2)))
    )
    with pytest.raises(TypeError):
        ap.gram_matrix(partition_tuple, dimension)
