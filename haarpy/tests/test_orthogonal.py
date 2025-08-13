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
Orthogonal tests
"""

from math import factorial
from itertools import permutations
from fractions import Fraction
import pytest
from sympy import Symbol
from sympy.combinatorics import Permutation, SymmetricGroup
import haarpy as ap

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
def test_hyperoctahedral_type_error(degree):
    "Hyperoctahedral TypeError for wrong degree type"
    with pytest.raises(TypeError):
        ap.hyperoctahedral(degree)


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
def test_zonal_spherical_orthogonality_transversal_zero(permutation, partition1, partition2):
    """Orthogonality relation for the zonal spherical function.
    All test should be 0 since partitions are distinct.
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces.
    <https://arxiv.org/abs/1301.5401>`_
    """
    degree = permutation.size
    convolution = sum(
        ap.zonal_spherical_function(tau, partition1)
        * ap.zonal_spherical_function(permutation * ~tau, partition2)
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
def test_zonal_spherical_orthogonality_transversal_none_zero(permutation, partition):
    """Orthogonality relation for the zonal spherical function
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    degree = permutation.size
    half_degree = degree // 2
    convolution = sum(
        ap.zonal_spherical_function(tau, partition)
        * ap.zonal_spherical_function(permutation * ~tau, partition)
        for tau in ap.hyperoctahedral_transversal(degree)
    )
    double_partition = tuple(2*i for i in partition)
    orthogonality = (
        Fraction(factorial(degree), 2**half_degree*factorial(half_degree))
        * ap.zonal_spherical_function(permutation, partition)
        / ap.irrep_dimension(double_partition)
    )
    assert convolution == orthogonality


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
    ],
)
def test_zonal_spherical_orthogonality_symmetric_zero(permutation, partition1, partition2):
    """Orthogonality relation for the zonal spherical function.
    All test should be 0 since partitions are distinct.
    `Matsumoto. General moments of matrix elements from circular orthogonal ensembles:
    <https://arxiv.org/abs/1109.2409>`_
    """
    degree = permutation.size
    convolution = sum(
        ap.zonal_spherical_function(tau, partition1)
        * ap.zonal_spherical_function(~tau * permutation, partition2)
        for tau in SymmetricGroup(degree).generate()
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
def test_zonal_spherical_orthogonality_symmetric_none_zero(permutation, partition):
    """Orthogonality relation for the zonal spherical function.
    All test should be 0 since partitions are distinct.
    `Matsumoto. General moments of matrix elements from circular orthogonal ensembles:
    <https://arxiv.org/abs/1109.2409>`_
    """
    degree = permutation.size
    convolution = sum(
        ap.zonal_spherical_function(tau, partition)
        * ap.zonal_spherical_function(~tau * permutation, partition)
        for tau in SymmetricGroup(degree).generate()
    )

    double_partition = tuple(2*i for i in partition)
    orthogonality = (
        Fraction(factorial(degree), ap.irrep_dimension(double_partition))
        * ap.zonal_spherical_function(permutation, partition)
    )

    assert convolution == orthogonality


@pytest.mark.parametrize(
    "cycle_type, partition",
    [
        (Permutation(2)(0, 1), (1,)),
        (Permutation(4)(0, 1, 2), (1, 1)),
        (Permutation(0, 1, 2, 3, 4, 5, 6), (4,)),
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
    ],
)
def test_zonal_spherical_partition_error(cycle_type, partition):
    "Test ValueError for invalid cycle-type and partition"
    with pytest.raises(ValueError, match=".*Invalid partition and cyle-type*"):
        ap.zonal_spherical_function(cycle_type, partition)


@pytest.mark.parametrize(
    "cycle_type, partition",
    [
        (Permutation(3)(0, 1), 'a'),
        (Permutation(5)(0, 1, 2), [1,1]),
        ((0, 1), (1,)),
        ('a', (1, 1)),
    ],
)
def test_zonal_spherical_type_error(cycle_type, partition):
    "Test ValueError for invalid cycle-type and partition"
    with pytest.raises(TypeError):
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
    """Symbolic validation of orthogonal Weingarten function against results shown in
    `Collins et al. Integration with Respect to the Haar Measure on Unitary, Orthogonal
    and Symplectic Group: <https://link.springer.com/article/10.1007/s00220-006-1554-3>`_
    """
    assert ap.weingarten_orthogonal(permutation, d) == num/denum


@pytest.mark.parametrize(
    "permutation, dimension, num, denum",
    [
        (Permutation(1), 7, 1, 7),
        (Permutation(3), 7, 8, 378),
        (Permutation(0, 1, 2, 3), 7, -1, 378),
        (Permutation(5), 7, 68, 20790),
        (Permutation(2, 3, 4, 5), 7, -1, 2310),
        (Permutation(0, 1, 2, 3, 4, 5), 7, 2, 20790),
        (Permutation(0, 1, 2, 3, 4, 5, 6, 7), 7, -41, 8648640),
        (Permutation(0, 1, 2, 3, 4, 7)(5, 6), 7, 2, 112320),
        (Permutation(0, 1, 2, 3)(4, 5, 6, 7), 7, 102, 8648640),
        (Permutation(4, 5, 6, 7), 7, -652, 8648640),
        (Permutation(7), 7, 920, 1729728),
    ]
)
def test_weingarten_orthogonal_numeric(permutation, dimension, num, denum):
    """Numeric validation of orthogonal Weingarten function against results shown in
    `Collins et al. Integration with Respect to the Haar Measure on Unitary, Orthogonal
    and Symplectic Group: <https://link.springer.com/article/10.1007/s00220-006-1554-3>`_
    """
    assert ap.weingarten_orthogonal(permutation, dimension) == Fraction(num,denum)


@pytest.mark.parametrize("degree", range(3))
def test_weingarten_orthognal_degree_error(degree):
    """Value error assertion for symmetric group of odd degree"""
    for conjugacy_class in SymmetricGroup(2*degree+1).conjugacy_classes():
        with pytest.raises(
            ValueError, match=".*The degree of the symmetric group S_2k should be even*"
        ):
            ap.weingarten_orthogonal(conjugacy_class.pop(), d)
