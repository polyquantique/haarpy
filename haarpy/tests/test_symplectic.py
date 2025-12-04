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

from math import factorial, prod
from random import seed, randint
from fractions import Fraction
from sympy import Symbol, simplify
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions
import numpy as np
import pytest
import haarpy as ap

seed(137)
d = Symbol('d')


def random_unit_quaternion():
    """Generate a random unit quaternion (q0,q1,q2,q3) with q0^2+q1^2+q2^2+q3^2 = 1."""
    vec = np.random.normal(size=4)
    vec /= np.linalg.norm(vec)
    return vec  # [q0, q1, q2, q3]


def generate_random_usp(d):
    """Generate a Haar-random USp(2d) matrix of size (2d x 2d)."""
    # Step 1: Random quaternionic d x d matrix with i.i.d. N(0,1) components
    A = np.random.normal(size=(d, d))
    B = np.random.normal(size=(d, d))
    C = np.random.normal(size=(d, d))
    D = np.random.normal(size=(d, d))
    # Prepare list to store orthonormal quaternion columns (each as 4 real component arrays)
    basis = []
    # Step 2: Quaternionic Gram-Schmidt
    for j in range(d):
        # j-th column as quaternion vector components
        v0, v1, v2, v3 = A[:, j].copy(), B[:, j].copy(), C[:, j].copy(), D[:, j].copy()
        # Make orthogonal to previous basis vectors
        for (u0, u1, u2, u3) in basis:
            # Compute quaternion inner product: q = <u, v> = sum_i conj(u_i)*v_i (a quaternion)
            p0, p1, p2, p3 = u0, -u1, -u2, -u3      # components of conj(u)
            q0 = p0 @ v0 - p1 @ v1 - p2 @ v2 - p3 @ v3  # (conj(u)·v)_real
            q1 = p0 @ v1 + p1 @ v0 + p2 @ v3 - p3 @ v2  # (conj(u)·v)_i
            q2 = p0 @ v2 - p1 @ v3 + p2 @ v0 + p3 @ v1  # (conj(u)·v)_j
            q3 = p0 @ v3 + p1 @ v2 - p2 @ v1 + p3 @ v0  # (conj(u)·v)_k
            # Subtract projection: v := v - u * q
            t0 = u0 * q0 - u1 * q1 - u2 * q2 - u3 * q3
            t1 = u0 * q1 + u1 * q0 + u2 * q3 - u3 * q2
            t2 = u0 * q2 - u1 * q3 + u2 * q0 + u3 * q1
            t3 = u0 * q3 + u1 * q2 - u2 * q1 + u3 * q0
            v0 -= t0;  v1 -= t1;  v2 -= t2;  v3 -= t3
        # Normalize v (so that <v,v> = 1)
        norm = np.sqrt(v0@v0 + v1@v1 + v2@v2 + v3@v3)
        v0 /= norm;  v1 /= norm;  v2 /= norm;  v3 /= norm
        # Step 3: Multiply by a random unit quaternion (random phase)
        q0, q1, q2, q3 = random_unit_quaternion()
        t0 = v0*q0 - v1*q1 - v2*q2 - v3*q3
        t1 = v0*q1 + v1*q0 + v2*q3 - v3*q2
        t2 = v0*q2 - v1*q3 + v2*q0 + v3*q1
        t3 = v0*q3 + v1*q2 - v2*q1 + v3*q0
        v0, v1, v2, v3 = t0, t1, t2, t3
        basis.append((v0, v1, v2, v3))
    # Step 4: Assemble the complex 2d x 2d matrix from quaternion basis
    d2 = 2 * d
    U = np.zeros((d2, d2), dtype=np.complex128)
    # Fill block entries for each quaternion column
    for j, (u0, u1, u2, u3) in enumerate(basis):
        U[:d, j]      = u0 + 1j*u1         # top-left block
        U[:d, j+d]    = u2 + 1j*u3         # top-right block
        U[d:, j]      = -u2 + 1j*u3        # bottom-left block
        U[d:, j+d]    = u0 - 1j*u1         # bottom-right block
    return U


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
        ((0,1,0,1),(0,1,1,0), 2),
        ((0,0,0,0,0,0),(0,0,0,0,0,0), 3),
        ((0,1,2,0,1,2),(0,0,0,0,0,0), 3),
        ((0,0,1,0,0,1),(0,2,2,0,2,2), 3),
        ((0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0), 4),
        ((0,0,0,0,0,0,0,0),(0,0,1,1,0,0,1,1), 4),
    ]
)
def test_haar_integral_symplectic_monte_carlo_symbolic(seq_i, seq_j, half_dimension):
    "Test haar integral symplectic moments against Monte Carlo simulation symbolic"
    sample_size = int(1e4)
    epsilon_real = 1e4
    epsilon_imag = 1e6

    half_length = len(seq_i) // 2
    seq_i_symbolic = seq_i[:half_length] + tuple(i+d for i in seq_i[half_length:])
    seq_j_symbolic = seq_j[:half_length] + tuple(j+d for j in seq_j[half_length:])
    seq_i_numeric = seq_i[:half_length] + tuple(i+half_dimension for i in seq_i[half_length:])
    seq_j_numeric = seq_j[:half_length] + tuple(j+half_dimension for j in seq_j[half_length:])

    integral = ap.haar_integral_symplectic((seq_i_symbolic, seq_j_symbolic), d)
    integral = float(integral.subs(d, half_dimension))

    monte_carlo_matrix = (generate_random_usp(half_dimension) for _ in range(sample_size))
    monte_carlo_integral = sum(
        prod(
            symplectic_matrix[i-1, j-1]
            for i, j in zip(seq_i_numeric, seq_j_numeric)
        )
        for symplectic_matrix in monte_carlo_matrix
    ) / sample_size

    assert (
        abs((integral-monte_carlo_integral.real)/integral) < epsilon_real
        and abs(monte_carlo_integral.imag) < epsilon_imag
        and integral != 0
    )


@pytest.mark.parametrize(
    'seq_i, seq_j, half_dimension',
    [
        ((0,0,0,0),(0,0,0,0), 2),
        ((0,1,0,1),(0,0,0,0), 2),
        ((0,1,0,1),(0,1,1,0), 2),
        ((0,0,0,0,0,0),(0,0,0,0,0,0), 3),
        ((0,1,2,0,1,2),(0,0,0,0,0,0), 3),
        ((0,0,1,0,0,1),(0,2,2,0,2,2), 3),
        ((0,0,0,0,0,0,0,0),(0,0,0,0,0,0,0,0), 4),
        ((0,0,0,0,0,0,0,0),(0,0,1,1,0,0,1,1), 4),
    ]
)
def test_haar_integral_symplectic_monte_carlo_numeric(seq_i, seq_j, half_dimension):
    "Test haar integral symplectic moments against Monte Carlo simulation symbolic"
    sample_size = int(1e4)
    epsilon_real = 1e4
    epsilon_imag = 1e6

    half_length = len(seq_i) // 2
    seq_i_numeric = seq_i[:half_length] + tuple(i+half_dimension for i in seq_i[half_length:])
    seq_j_numeric = seq_j[:half_length] + tuple(j+half_dimension for j in seq_j[half_length:])

    integral = float(ap.haar_integral_symplectic((seq_i_numeric, seq_j_numeric), half_dimension))

    monte_carlo_matrix = (generate_random_usp(half_dimension) for _ in range(sample_size))
    monte_carlo_integral = sum(
        prod(
            symplectic_matrix[i-1, j-1]
            for i, j in zip(seq_i_numeric, seq_j_numeric)
        )
        for symplectic_matrix in monte_carlo_matrix
    ) / sample_size

    assert (
        abs((integral-monte_carlo_integral.real)/integral) < epsilon_real
        and abs(monte_carlo_integral.imag) < epsilon_imag
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
