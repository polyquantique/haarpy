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
import pytest
from sympy import Symbol
import haarpy as ap

d = Symbol('d')


@pytest.mark.parametrize(
    "permutation, partition",
    [
        (Permutation(3), (1,1)),
        (Permutation(3)(0,1), (1,1)),
        (Permutation(0,1,2,3), (1,1)),
        (Permutation(3)(0,1,2), (2,)),
        (Permutation(0,1,2,3,4,5), (3,)),
        (Permutation(5)(0,4), (3,)),
        (Permutation(5)(0,1,4), (3,)),
        (Permutation(5)(0,1,2,3,4), (3,)),
        (Permutation(5)(0,1,2,3,4), (1,1,1)),
        (Permutation(5), (1,1,1)),
        (Permutation(5), (2,1)),
        (Permutation(5), (2,1)),
        (Permutation(7), (2,2)),
        (Permutation(7)(0,1), (2,1,1)),
    ],
)
def test_twisted_spherical_image(permutation, partition):
    """Validates that the twisted spherical function is the image of the zonal spherical function as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    conjugate_partition = tuple(
        sum(1 for i in partition if i > j) for j in range(partition[0])
    )
    assert ap.twisted_spherical_function(
        permutation,
        partition,
    ) == ap.zonal_spherical_function(
        permutation,
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
    """Symbolic validation of orthogonal Weingarten function against results shown in
    `Collins et al. Integration with Respect to the Haar Measure on Unitary, Orthogonal
    and Symplectic Group: <https://link.springer.com/article/10.1007/s00220-006-1554-3>`_
    """
    assert ap.weingarten_symplectic(permutation, -d) == num/denum