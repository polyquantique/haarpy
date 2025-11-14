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
Symmetric tests
"""

import pytest
from math import factorial, prod
from itertools import permutations
from random import seed, shuffle
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions
import haarpy as ap

seed(137)


@pytest.mark.parametrize(
    "degree, cycle, conjugacy",
    [
        (3, Permutation(0, 1, 2), (3,)),
        (3, Permutation(2)(0, 1), (2, 1)),
        (4, Permutation(0, 1, 2, 3), (4,)),
        (4, Permutation(0, 1, 2), (3, 1)),
        (4, Permutation(0, 1)(2, 3), (2, 2)),
        (5, Permutation(0, 1, 2), (3, 1, 1)),
        (5, Permutation(1, 2, 4), (3, 1, 1)),
        (5, Permutation(3)(0, 1, 2), (3, 1, 1)),
        (6, Permutation(2)(3, 4, 5)(0, 1), (3, 2, 1)),
        (6, Permutation(2, 3, 4, 0, 1, 5), (6,)),
        (7, Permutation(2, 3, 4, 0, 1, 6), (6, 1)),
        (1, Permutation(0), (1,)),
    ],
)
def test_get_conjugacy_class(degree, cycle, conjugacy):
    "Test the normal usage of get_conjugacy_class"
    assert ap.get_conjugacy_class(cycle, degree) == conjugacy


@pytest.mark.parametrize(
    "degree, cycle",
    [
        ("a", Permutation(0, 1, 2)),
        (0.1, Permutation(2)(0, 1)),
        ((1,), Permutation(0, 1, 2, 3)),
        ((5,), Permutation(0, 1, 2)),
    ],
)
def test_get_conjugacy_class_degree_type_error(degree, cycle):
    "Test the degree parameter TypeError"
    with pytest.raises(TypeError, match=".*degree must be of type int.*"):
        ap.get_conjugacy_class(cycle, degree)


@pytest.mark.parametrize("degree", range(-3, 1))
def test_get_conjugacy_class_degree_value_error(degree):
    "Test the degree parameter ValueError"
    with pytest.raises(
        ValueError,
        match=".*The degree you have provided is too low. It must be an integer greater than 0.*",
    ):
        ap.get_conjugacy_class(Permutation(0, 1, 2), degree)


@pytest.mark.parametrize(
    "degree, cycle",
    [
        (3, (1, 2, 3)),
        (4, "a"),
        (7, 2.0),
    ],
)
def test_get_conjugacy_class_cycle_type_error(degree, cycle):
    "Test the cycle parameter TypeError"
    with pytest.raises(
        TypeError,
        match=".*Permutation must be of type sympy.combinatorics.permutations.Permutation.*",
    ):
        ap.get_conjugacy_class(cycle, degree)


@pytest.mark.parametrize(
    "degree, cycle",
    [
        (3, Permutation(0, 1, 2, 3)),
        (3, Permutation(2, 3)(0, 1)),
        (4, Permutation(0, 1, 2, 3, 5)),
        (4, Permutation(4, 1)),
    ],
)
def test_get_conjugacy_class_cycle_value_error(degree, cycle):
    "Test the cycle parameter ValueError if permutation maximum value is greater than the degree"
    with pytest.raises(ValueError, match=".*Incompatible degree and permutation cycle.*"):
        ap.get_conjugacy_class(cycle, degree)


@pytest.mark.parametrize(
    "tableau, add_unit, partition, result",
    [
        (((), ()), 1, (3, 1), (((1,), ()),)),
        (((1, 2), (2, 3)), 4, (3, 2), (((1, 2, 4), (2, 3)),)),
        (((1, 1, 1, 3), (2,)), 4, (4, 2), (((1, 1, 1, 3), (2, 4,)),)),
        (((1,), ()), 1, (3, 1), (((1, 1), ()), ((1,), (1,)))),
        (((1, 2), ()), 2, (3, 2), (((1, 2, 2), ()), ((1, 2), (2,)))),
        (((), (), ()), 1, (3, 1, 1), (((1,), (), ()),)),
        (((1, 2, 3), (1, 2), (1,)), 3, (4, 2, 1), (((1, 2, 3, 3), (1, 2), (1,)),)),
    ],
)
def test_derivative_tableaux(tableau, add_unit, partition, result):
    "Test the normal usage of derivative_tableaux"
    assert tuple(ap.derivative_tableaux(tableau, add_unit, partition)) == result


@pytest.mark.parametrize(
    "partition, conjugacy_class, result",
    [
        ((3, 1), (2, 2), {((0, 1, 1), (0,)), ((0, 0, 1), (1,))}),
        (
            (3, 1, 1),
            (3, 2),
            {
                ((0, 0, 0), (1,), (1,)),
                ((0, 0, 1), (0,), (1,)),
                ((0, 1, 1), (0,), (0,)),
            },
        ),
        (
            (3, 2),
            (1, 2, 1, 1),
            {((0, 1, 1), (2, 3)), ((0, 1, 2), (1, 3)), ((0, 1, 3), (1, 2))},
        ),
        (
            (4, 2),
            (2, 2, 2),
            {
                ((0, 0, 1, 2), (1, 2)),
                ((0, 1, 1, 2), (0, 2)),
                ((0, 1, 2, 2), (0, 1)),
                ((0, 0, 1, 1), (2, 2)),
                ((0, 0, 2, 2), (1, 1)),
            },
        ),
        (
            (4, 3),
            (2, 2, 2),
            {
                ((0, 0, 1), (1, 2, 2)),
                ((0, 0, 1, 2), (1, 2)),
                ((0, 0, 2), (1, 1, 2)),
                ((0, 1, 1), (0, 2, 2)),
                ((0, 1, 2), (0, 1, 2)),
                ((0, 1, 1, 2), (0, 2)),
                ((0, 1, 2, 2), (0, 1)),
                ((0, 0, 1, 1), (2, 2)),
                ((0, 0, 2, 2), (1, 1)),
            },
        ),
        ((4, 2), (2, 2, 2, 2), set()),
    ],
)
def test_semi_standard_young_tableaux(partition, conjugacy_class, result):
    "Test the normal usage of semi_standard_young_tableaux"
    assert ap.semi_standard_young_tableaux(partition, conjugacy_class) == result


@pytest.mark.parametrize(
    "tableau, conjugacy_class",
    [
        (((1, 2, 2), (1,)), (2, 2)),
        (((1, 1, 1), (2,)), (3, 1)),
        (((1, 1, 2), (1,)), (3, 1)),
        (((1, 1, 1), (2,), (2,)), (3, 2)),
        (((1, 2, 2), (1,), (1,)), (3, 2)),
        (((1, 2, 2), (3, 4)), (1, 2, 1, 1)),
        (((1, 1, 2, 2), (3, 3)), (2, 2, 2)),
        (((1, 1, 3, 3), (2, 2)), (2, 2, 2)),
        (((1, 2, 3, 3), (1, 2)), (2, 2, 2)),
        (((1, 1, 1, 1, 1), (1, 2, 2)), (6, 2)),
    ],
)
def test_proper_border_strip_true(tableau, conjugacy_class):
    "Test bad_mapping for well mapped tableaux, returns False"
    assert ap.proper_border_strip(tableau, conjugacy_class)


@pytest.mark.parametrize(
    "tableau, conjugacy_class",
    [
        (((1, 1, 2), (2,)), (2, 2)),
        (((1, 1, 2), (1,), (2,)), (3, 2)),
        (((1, 2, 3), (2, 4)), (1, 2, 1, 1)),
        (((1, 2, 4), (2, 3)), (1, 2, 1, 1)),
        (((1, 1, 2, 3), (2, 3)), (2, 2, 2)),
        (((1, 2, 2, 3), (1, 3)), (2, 2, 2)),
        (((1, 1, 1, 1, 1), (1, 1, 2)), (7, 1)),
        (((1, 1, 1, 1, 2), (1, 1, 1)), (7, 1)),
    ],
)
def test_proper_border_strip_false(tableau, conjugacy_class):
    "Test bad_mapping for wrong mapped tableaux, returns True"
    assert not ap.proper_border_strip(tableau, conjugacy_class)


@pytest.mark.parametrize(
    "partition, conjugacy_class, character",
    [
        ((2, 1), (1, 1, 1), 2),
        ((3, 1), (1, 1, 1, 1), 3),
        ((3, 1), (3, 1), 0),
        ((3, 1), (2, 2), -1),
        ((3, 1), (2, 1, 1), 1),
        ((3, 1), (1, 2, 1), 1),
        ((3, 1), (1, 1, 2), 1),
        ((3, 2), (1, 1, 1, 1, 1), 5),
        ((3, 2), (1, 2, 1, 1), 1),
        ((3, 3), (1, 1, 1, 1, 1, 1), 5),
        ((3, 3), (2, 2, 2), -3),
        ((3, 3), (2, 2, 1, 1), 1),
        ((4, 1), (1, 1, 1, 1, 1), 4),
        ((4, 1), (2, 1, 1, 1), 2),
        ((4, 2), (2, 2, 2, 2), 0),
        ((4, 2), (1, 1, 1, 1, 1, 1), 9),
        ((4, 2), (2, 2, 2), 3),
        ((4, 2), (4, 1, 1), -1),
        ((4, 3), (1, 1, 1, 1, 1, 1, 1), 14),
        ((2, 2, 1), (2, 3), -1),
        ((2, 2, 2), (4, 2), -1),
        ((5, 3), (5, 1, 1, 1), -2),
        ((5, 3), (6, 1, 1), -1),
        ((5, 3), (7, 1), 0),
        ((4, 2, 1, 1), (7, 1), -1),
    ],
)
def test_murn_naka_rule(partition, conjugacy_class, character):
    "Test murn_naka_rule based on the outputs form weingarten mathematica package"
    assert ap.murn_naka_rule(partition, conjugacy_class) == character


@pytest.mark.parametrize(
    "partition, dimension",
    [
        ((1,), 1),
        ((5,), 1),
        ((3, 2), 5),
        ((11, 3), 273),
        ((8, 2, 2), 616),
        ((11, 3, 2), 13860),
        ((7, 3, 2, 2), 28028),
        ((5, 4, 2, 1, 1, 1), 63063),
        ((4, 2, 1, 1, 1, 5), -7007),
    ],
)
def test_irrep_dimension(partition, dimension):
    "Test irrep_dimension based on the outputs form weingarten mathematica package"
    assert ap.irrep_dimension(partition) == dimension


@pytest.mark.parametrize(
    "partition",
    [
        ((1,)),
        ((5,)),
        ((3, 2)),
        ((11, 3)),
        ((8, 2, 2)),
        ((5, 3, 2)),
        ((4, 3, 2, 2)),
        ((4, 3, 2, 1, 1)),
    ],
)
def test_irrep_dimension_murn_naka_rule(partition):
    "Reconcil irrep_dimension and murn_naka_rule for the identity conjugacy class"
    conjugacy_identity = sum(partition) * (1,)
    assert ap.irrep_dimension(partition) == ap.murn_naka_rule(partition, conjugacy_identity)


@pytest.mark.parametrize("degree", range(1,8))
def test_sorting_permutation_single_len(degree):
    "Test that the size of the permutation is the size of the sorted sequence"
    sample_size = 10
    shuffled_sequence = list(range(degree))
    for _ in range(sample_size):
        shuffle(shuffled_sequence)
        permutation = ap.sorting_permutation(tuple(shuffled_sequence))
        assert permutation.size == len(shuffled_sequence)


@pytest.mark.parametrize("degree", range(1,8))
def test_sorting_permutation_single_shuffle(degree):
    "Test that the sorting permutation properly sorts the shuffled sequence"
    sample_size = 10
    sorted_sequence = list(range(degree))
    for _ in range(sample_size):
        shuffled_sequence = sorted_sequence.copy()
        shuffle(shuffled_sequence)
        permutation = ap.sorting_permutation(tuple(shuffled_sequence))
        assert permutation(shuffled_sequence) == sorted_sequence


@pytest.mark.parametrize("degree", range(1,8))
def test_sorting_permutation_double_shuffle(degree):
    "Test that the sorting permutation properly sorts the shuffled sequence"
    sample_size = 10
    sorted_sequence = list(range(degree))
    for _ in range(sample_size):
        first_sequence = sorted_sequence.copy()
        second_sequence = sorted_sequence.copy()
        shuffle(first_sequence)
        shuffle(second_sequence)
        permutation = ap.sorting_permutation(
            tuple(first_sequence),
            tuple(second_sequence),
        )
        assert permutation(first_sequence) == second_sequence


@pytest.mark.parametrize("degree", range(3,8))
def test_sorting_permutation_type_error(degree):
    "Type error for more than 2 inputs"
    sequence = tuple(range(degree))
    with pytest.raises(TypeError):
        ap.sorting_permutation(
            *[tuple(sequence) for _ in range(degree)]
        )


@pytest.mark.parametrize(
    "seq1, seq2",
    [
        ((1,2,3),(0,1,2)),
        ((0,0,0,0), (0,0,0)),
    ]
)
def test_sorting_permutation_value_error(seq1, seq2):
    "Value error for different inputs"
    with pytest.raises(
        ValueError,
        match = "Incompatible sequences"
    ):
        ap.sorting_permutation(seq1, seq2)


@pytest.mark.parametrize('degree', range(1,8))
def test_young_subgroup_order(degree):
    "Test the order of the Young subgroup"
    for partition in partitions(degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        young = ap.YoungSubgroup(partition)
        assert (
            young.order()
            == prod(SymmetricGroup(part).order() for part in partition)
        )


@pytest.mark.parametrize('degree', range(2,8))
def test_young_subgroup_stabilizer(degree):
    "Test the stabilizer property of the Young subgroup"
    for partition in partitions(degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        if partition == degree*(1,):
            continue
        sequence = [i for i, j in enumerate(partition) for _ in range(j)]
        young = ap.YoungSubgroup(partition)
        assert young.random()(sequence) == sequence


@pytest.mark.parametrize(
    "partition",
    [
        {1,2,3},
        'abc',
        {1:1, 2:2},
        (2,2,0),
        (1,-1,2),
        (2,2,'a'),
    ],
)
def test_young_subgroup_type_error(partition):
    "Young subgroup TypeError"
    with pytest.raises(TypeError):
        ap.YoungSubgroup(partition)


@pytest.mark.parametrize(
        "seq1, seq2",
        [
            ((0,1),(0,2)),
            ((0,0,0), (0,0)),
            ((0,1,2,3,4), (0,1,2,3,3)),
        ]
)
def test_stabilizer_coset_empty(seq1, seq2):
    "test cases for which there are no stabilizing permutation"
    assert not len(tuple(ap.stabilizer_coset(seq1, seq2)))


@pytest.mark.parametrize("degree", range(2,6))
def test_stabilizer_coset_size(degree):
    "Test the size of the stabilizer coset"
    for partition in partitions(degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        seq1 = [i for i,j in enumerate(partition) for _ in range(j)]
        seq2 = seq1.copy()
        shuffle(seq1)
        shuffle(seq2)
        order = sum(1 for _ in ap.stabilizer_coset(tuple(seq1), tuple(seq2)))
        assert (
            order
            == prod(SymmetricGroup(part).order() for part in partition)
        )
    

@pytest.mark.parametrize("degree", range(2,8))
def test_stabilizer_coset_permutation(degree):
    "Test the set of permutations"
    for partition in partitions(degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        seq1 = [i for i,j in enumerate(partition) for _ in range(j)]
        seq2 = seq1.copy()
        shuffle(seq1)
        shuffle(seq2)
        assert all(
            perm(seq1) == seq2
            for perm in ap.stabilizer_coset(tuple(seq1), tuple(seq2))
        )


@pytest.mark.parametrize("degree", range(2,6))
def test_stabilizer_coset_brute_force(degree):
    "Test the set of permutations against brute force"
    for partition in partitions(degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        seq1 = [i for i,j in enumerate(partition) for _ in range(j)]
        seq2 = seq1.copy()
        shuffle(seq1)
        shuffle(seq2)
        stabilizer = tuple(ap.stabilizer_coset(tuple(seq1), tuple(seq2)))
        assert all(perm in stabilizer
            for perm in SymmetricGroup(degree).generate()
            if perm(seq1) == seq2
        )       


@pytest.mark.parametrize("degree", range(2,8))
def test_stabilizer_coset_single(degree):
    "Test the set of permutations for single input"
    for partition in partitions(degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        seq1 = [i for i,j in enumerate(partition) for _ in range(j)]
        shuffle(seq1)
        assert all(
            perm(seq1) == seq1
            for perm in ap.stabilizer_coset(tuple(seq1))
        )


@pytest.mark.parametrize("degree", range(3,8))
def test_stabilizer_coset_type_error(degree):
    "Type error for more than 2 inputs"
    sequence = tuple(range(degree))
    with pytest.raises(TypeError):
        ap.stabilizer_coset(
            *[tuple(sequence) for _ in range(degree)]
        )


@pytest.mark.parametrize("degree", range(1, 8))
def test_hyperoctahedral_order(degree):
    "Hyperoctahedral order test"
    assert ap.HyperoctahedralGroup(degree).order() == 2**degree * factorial(degree)


@pytest.mark.parametrize(
    "degree",
    [
        ("a",),
        ("str",),
        (0.1,),
        ((0, 1),),
    ],
)
def test_hyperoctahedral_type_error(degree):
    "Hyperoctahedral TypeError for wrong degree type"
    with pytest.raises(TypeError):
        ap.HyperoctahedralGroup(degree)


@pytest.mark.parametrize("degree", range(2, 12, 2))
def test_hyperoctahedral_transversal_size(degree):
    "Test the size of the hyperoctahedral transversal set"
    size = sum(1 for _ in ap.hyperoctahedral_transversal(degree))
    assert size == factorial(degree)/2**(degree//2)/factorial(degree//2)


@pytest.mark.parametrize("degree", range(2, 12, 2))
def test_hyperoctahedral_transversal_brute_force(degree):
    "Compare permutations of the transversal set with brute force method"
    brute_force_permutations = set()
    for permutation in permutations(range(degree)):
        if not all(permutation[2*i] < permutation[2*i+1] for i in range(degree//2)):
            continue
        if not all(permutation[2*i] < permutation[2*i+2] for i in range(degree//2-1)):
            continue
        brute_force_permutations.add(Permutation(permutation))

    transversal = set(ap.hyperoctahedral_transversal(degree))
    assert transversal == brute_force_permutations


@pytest.mark.parametrize("degree", range(3, 12, 2))
def test_hyperoctahedral_transversal_value_error(degree):
    "Test ValueError for odd degree"
    with pytest.raises(ValueError, match=".*degree should be a factor of 2*"):
        ap.hyperoctahedral_transversal(degree)


@pytest.mark.parametrize("half_degree", range(1,10))
def test_coset_type_partition(half_degree):
    "Test that all coset-types of S_2k are partitions of k"
    sample_size = 100 if half_degree > 2 else 4
    permutation_sample = (
        Permutation.random(2*half_degree) for _ in range(sample_size)
    )
    for permutation in permutation_sample:
        assert (
            sum(ap.coset_type(permutation)) == half_degree
        )


@pytest.mark.parametrize(
        "permutation",
        [
            [0,1,2],
            12,
            'abc',
            (1,1,1),
        ]
)
def test_coset_type_type_error(permutation):
    "Test TypeError if input is not a Permutation"
    with pytest.raises(TypeError):
        ap.coset_type(permutation)


@pytest.mark.parametrize("degree", range(1,11,2))
def test_coset_type_value_error(degree):
    "Test ValueError for odd size permutations"
    with pytest.raises(
        ValueError,
        match = "Coset-type are only defined for even sized permutations"
    ):
        ap.coset_type(Permutation.random(degree))


@pytest.mark.parametrize("half_degree", range(1,8))
def test_coset_type_coset_representative(half_degree):
    """ Taking all partitions of integer k, finding its coset-type
    representative and then finding the coset-type of this permutation
    should yield the initial partition
    """
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        assert (
            ap.coset_type(ap.coset_type_representative(partition))
            == partition
        )


@pytest.mark.parametrize(
        "partition",
        [
            ([1,2,3]),
            ("test"),
            (13),
        ]
)
def test_coset_type_representative_type_error(partition):
    "Test TypeError for invalid permutation and partition"
    with pytest.raises(TypeError):
        ap.coset_type_representative(partition)


@pytest.mark.parametrize("half_degree", range(2,7))
def test_coset_type_representative_in_transversal(half_degree):
    """assert that all coset-type representative permutations of integer partition are in M_2k as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    transversal = tuple(ap.hyperoctahedral_transversal(2*half_degree))
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        assert ap.coset_type_representative(partition) in transversal


@pytest.mark.parametrize("half_degree", range(2,7))
def test_coset_type_representative_signature(half_degree):
    """assert that all coset-type permutations of integer partition have signature of 1 as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        assert ap.coset_type_representative(partition).signature() == 1


@pytest.mark.parametrize("half_degree", range(2,10))
def test_coset_type_representative_identity(half_degree):
    """ asert that the coset-type permutation of the identity partition is the identity permutation
    as seen in `Matsumoto. Weingarten calculus for matrix ensembles associated with compact 
    symmetric spaces: <https://arxiv.org/abs/1301.5401>`_
    """
    assert ap.coset_type_representative(half_degree * (1,)) == Permutation(2*half_degree - 1)
