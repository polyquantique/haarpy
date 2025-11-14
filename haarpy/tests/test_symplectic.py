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
from sympy import Symbol, simplify
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions
import pytest
import haarpy as ap

d = Symbol('d')


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
    """Validates that the twisted spherical function is the image of the zonal spherical function
    as seen in `Matsumoto. Weingarten calculus for matrix ensembles associated with compact
    symmetric spaces: <https://arxiv.org/abs/1301.5401>`_
    """
    half_degree = sum(partition)
    conjugate_partition = tuple(
        sum(1 for i in partition if i > j) for j in range(partition[0])
    )
    for coset_type in partitions(half_degree):
        coset_type = tuple(key for key, value in coset_type.items() for _ in range(value))
        coset_type_permutation = ap.coset_type_representative(coset_type)
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


@pytest.mark.parametrize("half_degree", range(1,3))
def test_weingarten_symplectic_hyperoctahedral_symbolic(half_degree):
    """Symbolic validation of symplectic Weingarten function against results shown in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    if half_degree == 1:
        for permutation in SymmetricGroup(2*half_degree).generate():
            assert ap.weingarten_symplectic(permutation, d) == (
                permutation.signature()/(2*d)
            )
    else:
        for permutation in SymmetricGroup(2*half_degree).generate():
            hyperoctahedral = ap.HyperoctahedralGroup(half_degree)
            coefficient = permutation.signature()/(4*d*(d-1)*(2*d+1))
            assert ap.weingarten_symplectic(permutation, d) == (
                simplify((2*d-1)*coefficient) if permutation in hyperoctahedral
                else coefficient
            )


@pytest.mark.parametrize("half_degree", range(1,3))
def test_weingarten_symplectic_hyperoctahedral_numeric(half_degree):
    """Symbolic validation of symplectic Weingarten function against results shown in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    if half_degree == 1:
        for permutation in SymmetricGroup(2*half_degree).generate():
            assert ap.weingarten_symplectic(permutation, 7) == (
                Fraction(permutation.signature(),(2*7))
            )
    else:
        for permutation in SymmetricGroup(2*half_degree).generate():
            hyperoctahedral = ap.HyperoctahedralGroup(half_degree)
            coefficient = Fraction(permutation.signature(),(4*7*(7-1)*(2*7+1)))
            assert ap.weingarten_symplectic(permutation, 7) == (
                (2*7-1)*coefficient if permutation in hyperoctahedral
                else coefficient
            )


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
    assert ap.weingarten_symplectic(permutation, d) == simplify(
        (-1) ** (permutation.size//2)
        * permutation.signature()
        * ap.weingarten_orthogonal(permutation, -2*d)
    )


@pytest.mark.parametrize(
    "permutation, partition",
    [
        (Permutation(3,), [2,]),
        ((3,1), (1,1)),
        (Permutation(5,)(0,1), 'a'),
        ('a', (3,)),
        (Permutation(5,)(0,3,4), 7),
        (7, (1,1,1)),
    ],
)
def test_twisted_spherical_function_type_error(permutation, partition):
    with pytest.raises(TypeError):
        ap.twisted_spherical_function(permutation, partition)


@pytest.mark.parametrize(
    "permutation, partition",
    [
        (Permutation(3,), (2,2)),
        (Permutation(3,), (1,1,1)),
        (Permutation(4,), (2,1)),
        (Permutation(4,), (1,1,1)),
    ],
)
def test_twisted_spherical_function_degree_value_error(permutation, partition):
    with pytest.raises(ValueError):
        ap.twisted_spherical_function(permutation, partition)


@pytest.mark.parametrize(
    "permutation",
    [
        (Permutation(2)),
        (Permutation(4)),
        (Permutation(0,1,2,3,4)),
        (Permutation(6)),
    ]
)
def test_weingarten_symplectic_degree_value_error(permutation):
    with pytest.raises(ValueError, match = "The degree of the symmetric group S_2k should be even"):
        ap.weingarten_symplectic(permutation, d) 
