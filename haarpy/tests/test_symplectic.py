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
from random import randint
from fractions import Fraction
from sympy import Symbol, simplify
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions
import pytest
import haarpy as ap

d = Symbol('d')

monte_carlo_symplectic_dict = {
    ((0, 0, 0, 0), (0, 0, 0, 0), 2) : (0.09948889355794269+6.851639354662377e-21j),
    ((0, 1, 0, 1), (0, 0, 0, 0), 2) : (0.05025720143792921+3.8747831289542884e-21j),
    ((0, 1, 0, 1), (0, 1, 1, 0), 2) : (-0.02505580071352494-9.625388767948332e-07j),
    ((0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0), 3) : (0.018000029928256993+2.5207244033791384e-21j),
    ((0, 1, 2, 0, 1, 2), (0, 0, 0, 0, 0, 0), 3) : (0.0029514164273132704-1.8345674171674893e-21j),
    ((0, 0, 1, 0, 0, 1), (0, 2, 2, 0, 2, 2), 3) : (0.0037630369365813177+9.729122199612046e-23j),
    ((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0), 4) : (0.0029909334194272684-2.5931001156985104e-21j),
    ((0, 1, 2, 3, 0, 1, 2, 3), (0, 0, 0, 0, 0, 0, 0, 0), 4) : (0.0001260733931029849-4.471619720427258e-23j),
    ((0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 1, 1, 0, 0, 1, 1), 4) : (0.000502809515889242-4.1519024921401192e-22j),
}


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
    "Test type error for for wrong input formats"
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
    "Test value error for for wrong input formats"
    with pytest.raises(ValueError):
        ap.twisted_spherical_function(permutation, partition)


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
    "permutation",
    [
        (Permutation(2)),
        (Permutation(4)),
        (Permutation(0,1,2,3,4)),
        (Permutation(6)),
    ]
)
def test_weingarten_symplectic_degree_value_error(permutation):
    "Test value error for odd symmetric group degree"
    with pytest.raises(ValueError, match = "The degree of the symmetric group S_2k should be even"):
        ap.weingarten_symplectic(permutation, d)


@pytest.mark.parametrize(
    'seq_i, seq_j, half_dimension',
    [
        ((0,0,0,0),(0,0,0,0), 2),
        ((0,1,0,1),(0,0,0,0), 2),
        ((0,0,0,0,0,0),(0,0,0,0,0,0), 3),
        ((0,1,2,0,1,2),(0,0,0,0,0,0), 3),
        ((0,0,1,0,0,1),(0,2,2,0,2,2), 3),
        ((0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0), 4),
        ((0,1,2,3,0,1,2,3),(0,0,0,0,0,0,0,0), 4),
        ((0,0,0,0,0,0,0,0),(0,0,1,1,0,0,1,1), 4),
    ]
)
def test_haar_integral_symplectic_monte_carlo_symbolic(seq_i, seq_j, half_dimension):
    "Test haar integral symplectic moments against Monte Carlo simulation symbolic"
    epsilon_real = 2e-2
    epsilon_imag = 1e-6

    half_length = len(seq_i) // 2
    seq_i_symbolic = seq_i[:half_length] + tuple(i+d for i in seq_i[half_length:])
    seq_j_symbolic = seq_j[:half_length] + tuple(j+d for j in seq_j[half_length:])

    integral = ap.haar_integral_symplectic((seq_i_symbolic, seq_j_symbolic), d)
    integral = float(integral.subs(d, half_dimension))

    mc_integral = monte_carlo_symplectic_dict[(seq_i, seq_j, half_dimension)]

    assert (
        abs((integral - mc_integral.real) / integral) < epsilon_real
        and abs(mc_integral.imag) < epsilon_imag
        and integral != 0
    )


@pytest.mark.parametrize(
    'seq_i, seq_j, half_dimension',
    [
        ((0,0,0,0),(0,0,0,0), 2),
        ((0,1,0,1),(0,0,0,0), 2),
        ((0,0,0,0,0,0),(0,0,0,0,0,0), 3),
        ((0,1,2,0,1,2),(0,0,0,0,0,0), 3),
        ((0,0,1,0,0,1),(0,2,2,0,2,2), 3),
        ((0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0), 4),
        ((0,1,2,3,0,1,2,3),(0,0,0,0,0,0,0,0), 4),
        ((0,0,0,0,0,0,0,0),(0,0,1,1,0,0,1,1), 4),
    ]
)
def test_haar_integral_symplectic_monte_carlo_numeric(seq_i, seq_j, half_dimension):
    "Test haar integral symplectic moments against Monte Carlo simulation symbolic"
    epsilon_real = 2e-2
    epsilon_imag = 1e-6

    half_length = len(seq_i) // 2
    seq_i_numeric = seq_i[:half_length] + tuple(i+half_dimension for i in seq_i[half_length:])
    seq_j_numeric = seq_j[:half_length] + tuple(j+half_dimension for j in seq_j[half_length:])

    integral = float(ap.haar_integral_symplectic((seq_i_numeric, seq_j_numeric), half_dimension))

    mc_integral = monte_carlo_symplectic_dict[(seq_i, seq_j, half_dimension)]

    assert (
        abs((integral - mc_integral.real) / integral) < epsilon_real
        and abs(mc_integral.imag) < epsilon_imag
        and integral != 0
    )


@pytest.mark.parametrize(
    "sequences",
    [
        ((1,2,3,4),(1,2,3,4),(1,2,3,4)),
        ((1,2,3,4),),
        ((1,2,3,4), (1,2,3,4,5,6)),
        ((1,2,3,4,5,6), (1,2,3,4,5,6,7)),
    ]
)
def test_haar_integral_symplectic_value_error_wrong_tuple(sequences):
    "Value error for wrong sequence format"
    with pytest.raises(
        ValueError,
        match="Wrong sequence format"
    ):
        ap.haar_integral_symplectic(sequences, d)


@pytest.mark.parametrize(
    "sequences",
    [
        (('a','b','c','d'), (1,2,3,4)),
        ((1,1,d+1,d+1), (1,1,1,1)),
    ]
)
def test_haar_integral_symplectic_type_error_integer_dimension(sequences):
    "Type error for integer dimension with not integer sequences"
    dimension = randint(1,99)
    with pytest.raises(TypeError):
        ap.haar_integral_symplectic(sequences, dimension)


@pytest.mark.parametrize(
    "sequences, dimension",
    [
        (((1,3),(1,3)), 1),
        (((1,2,3,5),(1,2,3,4)), 2),
        (((1,2,3,41),(1,2,3,41)), 20),
    ]
)
def test_haar_integral_symplectic_value_error_outside_dimension_range(sequences, dimension):
    "Value error for sequences values outside dimension range"
    with pytest.raises(
        ValueError,
        match="The matrix indices are outside the dimension range",
    ):
        ap.haar_integral_symplectic(sequences, dimension)


@pytest.mark.parametrize(
    "sequences",
    [
        ((1,2,3,4),(1,2,3,'a')),
        ((1,2,3,4), (1,2,3,{1,2})),
        ((1,2,3,4),(1,2,3,4*d)),
        ((1,2,3,2*d+1), (1,2,3,4)),
        ((1,2,3,d+1), (1,2,3,4.0)),
        ((1,2,3,4), (1,2,3,d**2)),
        ((1,2,3,4), (1,2,3,1+d**2+d)),
        ((1,2,3,4), (1,2,3, d + Symbol('s'))),
    ]
)
def test_haar_integral_symplectic_type_error_wrong_format(sequences):
    "Type error for symbolic dimension with wrong sequence format"
    with pytest.raises(TypeError):
        ap.haar_integral_symplectic(sequences, d)


@pytest.mark.parametrize(
    "dimension",
    [
        'a',
        [1,2],
        {1,2},
        3.0,
    ]
)
def test_haar_integral_symplectic_wrong_dimension_format(dimension):
    "Type error if the symplectic dimension is not an int nor a symbol"
    with pytest.raises(TypeError):
        ap.haar_integral_symplectic(((1,2,3,4),(1,2,3,4)), dimension)


@pytest.mark.parametrize(
    "sequences, dimension",
    [
        (((1,1,1),(1,1,1)), d),
        (((1,1,1,1,1),(1,1,1,1,1)), d),
        (((1,1+d,1+d),(1,1+d,1+d)), d),
        (((1,1,1,1),(1,1,1,1)), d),
        (((1,1,d+1,d+2),(1,1,d+1,d+1)), d),
        (((1,0,0,d),(1,d+1,0,d)), d),
        (((0,0,0), (0,0,0)), 2),
        (((0,0,0,0), (0,0,0,0)), 2),
        (((1,2,3,3), (1,2,3,3)), 4),
        (((1,1,5,5,5), (1,1,5,5,5)), 4),
    ]
)
def test_haar_integral_symplectic_zero_cases(sequences, dimension):
    "Test cases that yield zero"
    assert not ap.haar_integral_symplectic(sequences, dimension)


@pytest.mark.parametrize("half_degree", range(1,5))
def test_haar_integral_symplectic_weingarten_reconciliation(half_degree):
    "Test single permutation moments match the symplectic weingarten function"
    seq_dim_base = tuple(i+d for i in range(half_degree))
    sequence = tuple(i+1 for pair in zip(range(half_degree), seq_dim_base) for i in pair)

    for perm in ap.hyperoctahedral_transversal(2*half_degree):
        inv_perm = ~perm
        perm_sequence = tuple(inv_perm(sequence))

        assert (
            ap.haar_integral_symplectic((sequence, perm_sequence), d)
            == ap.weingarten_symplectic(perm, d)
        )
