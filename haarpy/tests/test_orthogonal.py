# Copyright 2024 Polyquantique

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
Orthogonal tests
"""

import pytest
import haarpy as ap
from math import factorial
from itertools import permutations
from sympy import Symbol
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions
from fractions import Fraction

d = Symbol('d')


@pytest.mark.parametrize("degree", range(1, 8))
def test_hyperoctahedral_order(degree):
    "Hyperoctahedral order test"
    assert ap.hyperoctahedral(degree).order() == 2**degree * factorial(degree)


@pytest.mark.parametrize(
    "degree",
    [
        ("a",),
        ("str",),
        (0.1,),
        ((0, 1),),
    ],
)
def test_hyperoctahedral_TypeError(degree):
    "Hyperoctahedral TypeError for wrong degree type"
    with pytest.raises(TypeError):
        ap.hyperoctahedral(degree)


@pytest.mark.parametrize("degree", range(4, 12, 2))
def test_hyperoctahedral_transversal_size(degree):
    "Test the size of the hyperoctahedral transversal set"
    size = sum(1 for _ in ap.hyperoctahedral_transversal(degree))
    assert size == factorial(degree)/2**(degree//2)/factorial(degree//2)


@pytest.mark.parametrize("degree", range(4, 12, 2))
def test_hyperoctahedral_transversal_brute_force(degree):
    "Compare permutations of the transversal set with brute force method"
    brute_force_permutations = set()
    for permutation in permutations(range(degree)):
        if not all(permutation[2*i] < permutation[2*i+1] for i in range(degree//2)):
            continue
        if not all(permutation[2*i] < permutation[2*i+2] for i in range(degree//2-1)):
            continue
        brute_force_permutations.add(Permutation(permutation))

    transversal = {permutation for permutation in ap.hyperoctahedral_transversal(degree)}
    assert transversal == brute_force_permutations


@pytest.mark.parametrize("degree", range(3, 12, 2))
def test_hyperoctahedral_transversal_ValueError(degree):
    "Test ValueError for odd degree"
    with pytest.raises(ValueError, match=".*degree should be a factor of 2*"):
        ap.hyperoctahedral_transversal(degree)


@pytest.mark.parametrize(
    "permutation, partition1, partition2",
    [
        (Permutation(3), (1,1), (2,)),
        (Permutation(3)(0,1), (1,1), (2,)),
        (Permutation(0,1,2,3), (1,1), (2,)),
        (Permutation(3)(0,1,2), (1,1), (2,)),
        (Permutation(0,1,2,3,4,5), (3,), (2,1)),
        (Permutation(5)(0,4), (3,), (1,1,1)),
        (Permutation(5)(0,4), (3,), (2,1)),
        (Permutation(5)(0,1,2,3,4), (3,), (1,1,1)),
        (Permutation(5)(0,1,2,3,4), (1,1,1), (2,1)),
        (Permutation(5)(0,1,2,3,4), (3,), (2,1)),
        (Permutation(5), (3,), (1,1,1)),
        (Permutation(5), (3,), (2,1)),
        (Permutation(5), (2,1), (1,1,1)),
        (Permutation(7), (2,1,1), (2,2)),
        (Permutation(7)(0,1), (2,1,1), (2,2)),
    ],
)
def test_zonal_spherical_orthogonality_zero(permutation, partition1, partition2):
    """Orthogonality relation for the zonal spherical function as
    seen in Matsumoto's 'Weingarten calculus for matrix ensembles
    associated with compact symmetric spaces'. All test should be 0
    since partitions are distinct
    """
    degree = permutation.size
    convolution = sum(
        ap.zonal_spherical_function(tau, partition1)
        * ap.zonal_spherical_function(~tau * permutation, partition2)
        for tau in ap.hyperoctahedral_transversal(degree)
    )

    assert not convolution


@pytest.mark.parametrize(
    "permutation, partition",
    [
    (Permutation(3,), (2,)),
    (Permutation(3,), (1,1)),
    (Permutation(5,)(0,1), (2,1)),
    (Permutation(0,1,2,3,4,5), (3,)),
    (Permutation(5,)(0,3,4), (3,)),
    (Permutation(0,1,2,3,4,5), (1,1,1)),
    (Permutation(0,1,2,3,4,5), (2,1)),
    (Permutation(0,3,5), (2,1)),
    (Permutation(0,3,4,5), (2,1)),
    (Permutation(0,2,3,4,5), (2,1)),
    ],
)
def test_zonal_spherical_orthogonality_none_zero(permutation, partition):
    """Orthogonality relation for the zonal spherical function as
    seen in Matsumoto's 'Weingarten calculus for matrix ensembles
    associated with compact symmetric spaces'
    """
    degree = permutation.size
    half_degree = degree // 2
    convolution = sum(
        ap.zonal_spherical_function(tau, partition)
        * ap.zonal_spherical_function(~tau * permutation, partition)
        for tau in ap.hyperoctahedral_transversal(degree)
    )
    double_partition = tuple(2*i for i in partition)
    orthogonality = (
        Fraction(factorial(degree), 2**half_degree*factorial(half_degree))
        * ap.zonal_spherical_function(permutation, partition)
        / ap.irrep_dimension(double_partition)
    )
    assert convolution == orthogonality


@pytest.mark.parametrize("half_degree", range(2, 4))
def test_zonal_spherical_class_function(half_degree):
    "Validates that all permutations of the same class yield the same output"
    degree = 2*half_degree
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        for conjugacy_class in SymmetricGroup(degree).conjugacy_classes():
            representative = ap.zonal_spherical_function(conjugacy_class.pop(), partition)
            for permutation in conjugacy_class:
                assert ap.zonal_spherical_function(permutation, partition) == representative


@pytest.mark.parametrize("half_degree", range(2, 4))
def test_zonal_spherical_polymorphism(half_degree):
    "Validates that the argument can be either a partition or a permutation"
    degree = 2*half_degree
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        for conjugacy_classes in SymmetricGroup(degree).conjugacy_classes():
            fiducial = conjugacy_classes.pop()
            zonal_class = ap.zonal_spherical_function(
                ap.get_conjugacy_class(fiducial, degree), partition
            )
            zonal_permutation = ap.zonal_spherical_function(fiducial, partition)
            assert zonal_class == zonal_permutation


@pytest.mark.parametrize(
    "cycle_type, partition",
    [
        (Permutation(2)(0, 1), (1,)),
        (Permutation(4)(0, 1, 2), (1, 1)),
        (Permutation(0, 1, 2, 3, 4, 5, 6), (4,)),
        ((1, 1, 1), (2,)),
        ((2, 1, 1, 1), (2,)),
        ((4, 1), (3,)),
        ((3, 3, 1), (3,)),
    ],
)
def test_zonal_spherical_degree_error(cycle_type, partition):
    "Test ValueError for odd degree"
    with pytest.raises(ValueError, match=".*degree should be a factor of 2*"):
        ap.zonal_spherical_function(cycle_type, partition)


@pytest.mark.parametrize(
    "cycle_type, partition",
    [
        (Permutation(3)(0, 1), (1,)),
        (Permutation(5)(0, 1, 2), (1, 1)),
        (Permutation(0, 1, 2, 3, 4, 5), (4,)),
        ((1, 1, 1, 1), (3,)),
        ((2, 2, 1, 1), (2,)),
        ((4, 2), (4,)),
        ((3, 3, 2), (2, 2, 2)),
    ],
)
def test_zonal_spherical_partition_error(cycle_type, partition):
    "Test ValueError for invalid cycle-type and partition"
    with pytest.raises(ValueError, match=".*Invalid partition and cyle-type*"):
        ap.zonal_spherical_function(cycle_type, partition)


@pytest.mark.parametrize(
    "permutation, num, denum",
    [
        (Permutation(1), 1, d),
        (Permutation(3), d+1, d*(d-1)*(d+2)),
        (Permutation(0,1,2,3), -1, d*(d-1)*(d+2)),
        (Permutation(5), d**2+3*d-2, d*(d-1)*(d-2)*(d+2)*(d+4)),
        (Permutation(2,3,4,5), -1, d*(d-1)*(d-2)*(d+4)),
        (Permutation(0,1,2,3,4,5), 2, d*(d-1)*(d-2)*(d+2)*(d+4)),
        (Permutation(0,1,2,3,4,5,6,7), -5*d-6, d*(d-1)*(d-2)*(d-3)*(d+1)*(d+2)*(d+4)*(d+6)),
        (Permutation(0,1,2,3,4,7)(5,6), 2, (d-1)*(d-2)*(d-3)*(d+1)*(d+2)*(d+6)),
        (Permutation(0,1,2,3)(4,5,6,7), d**2+5*d+18, d*(d-1)*(d-2)*(d-3)*(d+1)*(d+2)*(d+4)*(d+6)),
        (Permutation(4,5,6,7), -d**3-6*d**2-3*d+6, d*(d-1)*(d-2)*(d-3)*(d+1)*(d+2)*(d+4)*(d+6)),
        (Permutation(7), (d+3)*(d**2+6*d+1), d*(d-1)*(d-3)*(d+1)*(d+2)*(d+4)*(d+6)),
    ]
)
def test_weingarten_orthogonal(permutation, num, denum):
    """Validates orthogonal Weingarten function against results shown
    in Dr. Collin's 'Integration with Respect to the Haar Measure on 
    Unitary, Orthogonal and Symplectic Group'.
    """
    assert ap.weingarten_orthogonal(permutation, d) == num/denum


@pytest.mark.parametrize("degree", range(3))
def test_weingarten_orthognal_degree_error(degree):
    """Value error assertion for symmetric group of odd degree"""
    for conjugacy_class in SymmetricGroup(2*degree+1).conjugacy_classes():
        with pytest.raises(ValueError, match=".*The degree of the symmetric group S_2k should be even*"):
            ap.weingarten_orthogonal(conjugacy_class.pop(), d)