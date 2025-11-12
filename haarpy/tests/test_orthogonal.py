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

from math import factorial, prod
from itertools import product
from fractions import Fraction
from random import seed, randint
import pytest
from sympy import Symbol, factorial2
from sympy.combinatorics import Permutation, SymmetricGroup
import haarpy as ap

seed(137)
d = Symbol('d')


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
    "permutation, partition",
    [
        (Permutation(3)(0, 1), 'a'),
        (Permutation(5)(0, 1, 2), [1,1]),
        ((0, 1), (1,)),
        ('a', (1, 1)),
    ],
)
def test_zonal_spherical_type_error(permutation, partition):
    "Test TypeError for invalid permutation and partition"
    with pytest.raises(TypeError):
        ap.zonal_spherical_function(permutation, partition)


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
def test_weingarten_orthogonal_literature(permutation, num, denum):
    """Symbolic validation of orthogonal Weingarten function against results shown in
    `Collins et al. Integration with Respect to the Haar Measure on Unitary, Orthogonal
    and Symplectic Group: <https://link.springer.com/article/10.1007/s00220-006-1554-3>`_
    """
    assert ap.weingarten_orthogonal(permutation, d) == num/denum


@pytest.mark.parametrize("degree", range(2,10,2))
def test_weingarten_orthogonal_coset_type(degree):
    "Test that the orthogonal function works for both a permutation or its coset-type"
    sample_size = 10 if degree > 2 else 5
    permutation_sample = (
        Permutation.random(degree) for _ in range(sample_size)
    )
    for permutation in permutation_sample:
        assert (
            ap.weingarten_orthogonal(permutation, d)
            == ap.weingarten_orthogonal(ap.coset_type(permutation), d)
        )


@pytest.mark.parametrize(
    "coset_type, dimension",
    [
        ((3, 2), 1.0),
        ((3, 1, 1), 'a'),
        ((2, 2, 1), (1, 0)),
        ((3, 3), (8,)),
    ],
)
def test_weingarten_orthogonal_class_dimension_type_error(coset_type, dimension):
    with pytest.raises(
        TypeError,
        match=".*orthogonal_dimension must be an instance of int or sympy.Symbol*",
    ):
        ap.weingarten_orthogonal(coset_type, dimension)


@pytest.mark.parametrize(
    "coset",
    [
        (1,2,"a"),
        (3, (1,2), 4),
        "abc",
    ]
)
def test_weingarten_orthogonal_coset_type_error(coset):
    "Test the type error for wrong permutation input"
    with pytest.raises(TypeError):
        ap.weingarten_orthogonal(coset, d)


@pytest.mark.parametrize("degree", range(1,10,2))
def test_weingarten_orthogonal_degree_value_error(degree):
    "Test value error for odd size permutations"
    with pytest.raises(
        ValueError,
        match="The degree of the symmetric group S_2k should be even",
    ):
        ap.weingarten_orthogonal(Permutation.random(degree), d)


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
    "Value error assertion for symmetric group of odd degree"
    for conjugacy_class in SymmetricGroup(2*degree+1).conjugacy_classes():
        with pytest.raises(
            ValueError, match=".*The degree of the symmetric group S_2k should be even*"
        ):
            ap.weingarten_orthogonal(conjugacy_class.pop(), d)


@pytest.mark.parametrize(
    "power_tuple",
    [
        (1,1),
        (2,2),
        (2,1),
        (2,3),
        (2,4),
        (3,3),
        (1,1,4),
        (1,4),
        (2,2,2),
        (2,2,4),
    ]
)
def test_haar_integral_orthogonal_column_symbolic(power_tuple):
    "Test based on the column integral of an orthogonal Haar-random matrix (symbolic)"
    seq_i = sum(power_tuple)*(1,)
    seq_j = tuple(i for i in range(len(power_tuple)) for _ in range(power_tuple[i]))
    if any(power % 2 for power in power_tuple):
        assert not ap.haar_integral_orthogonal((seq_i, seq_j), d)
    else:
        assert ap.haar_integral_orthogonal((seq_i, seq_j), d) == (
            prod(factorial2(power-1) for power in power_tuple)
            / prod((d+i) for i in range(0,sum(power_tuple), 2))
        )


@pytest.mark.parametrize(
    "power_tuple",
    [
        (1,1),
        (2,2),
        (2,1),
        (2,3),
        (2,4),
        (3,3),
        (1,1,4),
        (1,4),
    ]
)
def test_haar_integral_orthogonal_column_numeric(power_tuple):
    "Test based on the column integral of an orthogonal Haar-random matrix (numeric)"
    dimension = randint(5,15)
    seq_i = sum(power_tuple)*(1,)
    seq_j = tuple(i for i in range(len(power_tuple)) for _ in range(power_tuple[i]))
    if any(power % 2 for power in power_tuple):
        assert not ap.haar_integral_orthogonal((seq_i, seq_j), dimension)
    else:
        assert ap.haar_integral_orthogonal((seq_i, seq_j), dimension) == (
            prod(factorial2(power-1) for power in power_tuple)
            / prod((dimension+i) for i in range(0,sum(power_tuple), 2))
        )


@pytest.mark.parametrize(
        "dimension, half_power",
        product(range(2,5), range(1,4))
)
def test_haar_integral_orthogonal_trace(dimension, half_power):
    "Test based on the integral of the power of the trace"
    for half_power in range(1,min(dimension+1, 4)):
        integral = sum(
            ap.haar_integral_orthogonal(
                (seq_i, seq_i),
                dimension
            )
            for seq_i in product(range(dimension), repeat=2*half_power)
        )
        assert integral == factorial2(2*half_power - 1)


@pytest.mark.parametrize(
    "sequences",
    [
        ((1,),),
        ((1,2,3),),
        ((1,2),(3,4),(4,5)),
        "str",
        ((1,2),(3,4,5)),
        ((1,2,3),(3,4,5,6)),
    ]
)
def test_haar_integral_orthogonal_value_error(sequences):
    "Test haar integral value error"
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral_orthogonal(sequences, d)
