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
Symplectic tests
"""

from math import factorial
from fractions import Fraction
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
import pytest
from sympy import Symbol
import haarpy as ap

d = Symbol('d')


@pytest.mark.parametrize(
        "partition",
        [
            ([1,2,3]),
            ("test"),
            (13),
        ]
)
def test_coset_type_type_error(partition):
    "Test TypeError for invalid permutation and partition"
    with pytest.raises(TypeError):
        ap.coset_type(partition)


@pytest.mark.parametrize("half_degree", range(2,7))
def test_coset_type_in_transversal(half_degree):
    """assert that all coset-type permutations of integer partition are in M_2k as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    transversal = tuple(ap.hyperoctahedral_transversal(2*half_degree))
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        assert ap.coset_type(partition) in transversal


@pytest.mark.parametrize("half_degree", range(2,7))
def test_coset_type_signature(half_degree):
    """assert that all coset-type permutations of integer partition have signature of 1 as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    for partition in partitions(half_degree):
        partition = tuple(key for key, value in partition.items() for _ in range(value))
        assert ap.coset_type(partition).signature() == 1


@pytest.mark.parametrize("half_degree", range(2,10))
def test_coset_type_identity(half_degree):
    """ asert that the coset-type permutation of the identity partition is the identity permutation as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    assert ap.coset_type(half_degree * (1,)) == Permutation(2*half_degree - 1)


@pytest.mark.parametrize(
        "partition",
        [
            ((1,1)),
            ((2,)),
            ((1,1,1)),
            ((2,1)),
            ((3,)),
            ((1,1,1,1)),
            ((2,2)),
            ((3,1,1)),
        ]
)
def test_twisted_spherical_image(partition):
    """Validates that the twisted spherical function is the image of the zonal spherical function as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    half_degree = sum(partition)
    conjugate_partition = tuple(
        sum(1 for i in partition if i > j) for j in range(partition[0])
    )
    for coset_type in partitions(half_degree):
        coset_type = tuple(key for key, value in coset_type.items() for _ in range(value))
        coset_type_permutation = ap.coset_type(coset_type)
        assert ap.twisted_spherical_function(
            coset_type_permutation,
            partition,
        ) == ap.zonal_spherical_function(
            coset_type_permutation,
            conjugate_partition,
        )


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
def test_twisted_spherical_orthogonality_transversal_zero(permutation, partition1, partition2):
    """Orthogonality relation for the twisted spherical function.
    All test should be 0 since partitions are distinct.
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces.
    <https://arxiv.org/abs/1301.5401>`_
    """
    degree = permutation.size
    convolution = sum(
        ap.twisted_spherical_function(tau, partition1)
        * ap.twisted_spherical_function(permutation * ~tau, partition2)
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
def test_twisted_spherical_orthogonality_transversal_none_zero(permutation, partition):
    """Orthogonality relation for the twisted spherical function
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    degree = permutation.size
    half_degree = degree // 2
    convolution = sum(
        ap.twisted_spherical_function(tau, partition)
        * ap.twisted_spherical_function(permutation * ~tau, partition)
        for tau in ap.hyperoctahedral_transversal(degree)
    )
    duplicate_partition = tuple(part for part in partition for _ in range(2))
    orthogonality = (
        Fraction(factorial(degree), 2**half_degree*factorial(half_degree))
        * ap.twisted_spherical_function(permutation, partition)
        / ap.irrep_dimension(duplicate_partition)
    )
    assert convolution == orthogonality


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
def test_weingarten_symplectic(permutation, num, denum):
    """Symbolic validation of symplectic Weingarten function against results shown in
    `Collins et al. Integration with Respect to the Haar Measure on Unitary, Orthogonal
    and Symplectic Group: <https://link.springer.com/article/10.1007/s00220-006-1554-3>`_
    """
    assert ap.weingarten_symplectic(permutation, -d) == num/denum


@pytest.mark.parametrize(
    "permutation",
    [
        (Permutation(1)),
        (Permutation(3)),
        (Permutation(0,1,2,3)),
        (Permutation(5)),
        (Permutation(2,3,4,5)),
        (Permutation(0,1,2,3,4,5)),
        (Permutation(0,1,2,3,4,5,6,7)),
        (Permutation(0,1,2,3,4,7)(5,6)),
        (Permutation(0,1,2,3)(4,5,6,7)),
        (Permutation(4,5,6,7)),
        (Permutation(7)),
    ]
)
def test_weingarten_symplectic_orthogonal_relation(permutation):
    """Symbolic validation of the relation between the symplectic and 
    orthogonal Weingarten functions as seen in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    assert ap.weingarten_symplectic(permutation, d) == (
        (-1) ** permutation.size * permutation.signature() * ap.weingarten_orthogonal(permutation, -2*d)
    )
    
