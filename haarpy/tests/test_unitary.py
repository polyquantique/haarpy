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
haarpy tests
"""

import pytest
import haarpy as ap
from fractions import Fraction
from sympy.combinatorics import Permutation
from sympy import Symbol, simplify, fraction, factor
from sympy.combinatorics.named_groups import SymmetricGroup


@pytest.mark.parametrize(
    "degree,cycle,conjugacy",
    [
        (3, Permutation(0, 1, 2), (3,)),
        (
            3,
            Permutation(
                2,
            )(0, 1),
            (2, 1),
        ),
        (4, Permutation(0, 1, 2, 3), (4,)),
        (4, Permutation(0, 1, 2), (3, 1)),
        (4, Permutation(0, 1)(2, 3), (2, 2)),
        (5, Permutation(0, 1, 2), (3, 1, 1)),
        (5, Permutation(1, 2, 4), (3, 1, 1)),
        (
            5,
            Permutation(
                3,
            )(0, 1, 2),
            (3, 1, 1),
        ),
        (
            6,
            Permutation(
                2,
            )(
                3, 4, 5
            )(0, 1),
            (3, 2, 1),
        ),
        (6, Permutation(2, 3, 4, 0, 1, 5), (6,)),
        (7, Permutation(2, 3, 4, 0, 1, 6), (6, 1)),
        (
            1,
            Permutation(
                0,
            ),
            (1,),
        ),
    ],
)
def test_get_class(degree, cycle, conjugacy):
    """Test the normal usage of get_class"""
    assert ap.get_class(cycle, degree) == conjugacy


@pytest.mark.parametrize(
    "degree,cycle",
    [
        ("a", Permutation(0, 1, 2)),
        (0.1, Permutation(2)(0, 1)),
        ((1,), Permutation(0, 1, 2, 3)),
        ([5], Permutation(0, 1, 2)),
    ],
)
def test_get_class_degree_type_error(degree, cycle):
    """Test the degree parameter TypeError"""
    with pytest.raises(TypeError, match=".*degree must be of type int.*"):
        ap.get_class(cycle, degree)


@pytest.mark.parametrize("degree", range(-3, 1))
def test_get_class_degree_value_error(degree):
    """Test the degree parameter ValueError"""
    with pytest.raises(
        ValueError,
        match=".*The degree you have provided is too low. It must be an integer greater than 0.*",
    ):
        ap.get_class(Permutation(0, 1, 2), degree)


@pytest.mark.parametrize(
    "degree,cycle",
    [
        (3, [1, 2, 3]),
        (4, "a"),
        (7, 2.0),
    ],
)
def test_get_class_cycle_type_error(degree, cycle):
    """Test the cycle parameter TypeError"""
    with pytest.raises(
        TypeError,
        match=".*cycle must be of type sympy.combinatorics.permutations.Permutation.*",
    ):
        ap.get_class(cycle, degree)


@pytest.mark.parametrize(
    "degree,cycle",
    [
        (3, Permutation(0, 1, 2, 3)),
        (3, Permutation(2, 3)(0, 1)),
        (4, Permutation(0, 1, 2, 3, 5)),
        (4, Permutation(4, 1)),
    ],
)
def test_get_class_cycle_value_error(degree, cycle):
    """Test the cycle parameter ValueError if permutation maximum value is greater than the degree"""
    with pytest.raises(
        ValueError, match=".*Incompatible degree and permutation cycle.*"
    ):
        ap.get_class(cycle, degree)


@pytest.mark.parametrize(
    "tableau,add_unit,partition,result",
    [
        ([[0, 0, 0], [0, 0, 0]], 1, (3, 1), [[[1, 0, 0], [0, 0, 0]]]),
        (
            [[1, 0, 0], [0, 0, 0]],
            1,
            (3, 1),
            [[[1, 1, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]]],
        ),
        (
            [[1, 2, 0], [0, 0, 0]],
            2,
            (3, 2),
            [[[1, 2, 2], [0, 0, 0]], [[1, 2, 0], [2, 0, 0]]],
        ),
        ([[1, 2, 0], [2, 3, 0]], 4, [3, 2], [[[1, 2, 4], [2, 3, 0]]]),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            1,
            (3, 1, 1),
            [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
        ),
        ([[1, 1, 1, 3], [2, 0, 0, 0]], 4, (4, 2), [[[1, 1, 1, 3], [2, 4, 0, 0]]]),
        (
            [[1, 2, 3, 0], [1, 2, 0, 0], [1, 0, 0, 0]],
            3,
            (4, 2, 1),
            [[[1, 2, 3, 3], [1, 2, 0, 0], [1, 0, 0, 0]]],
        ),
    ],
)
def test_derivative_tableau(tableau, add_unit, partition, result):
    """Test the normal usage of derivative_tableau"""
    assert ap.derivative_tableau(tableau, add_unit, partition) == result


@pytest.mark.parametrize(
    "partition, conjugacy_class, result",
    [
        ((3, 1), (2, 2), [[[1, 2, 2], [1, 0, 0]], [[1, 1, 2], [2, 0, 0]]]),
        (
            (3, 1, 1),
            (3, 2),
            [
                [[1, 1, 1], [2, 0, 0], [2, 0, 0]],
                [[1, 1, 2], [1, 0, 0], [2, 0, 0]],
                [[1, 2, 2], [1, 0, 0], [1, 0, 0]],
            ],
        ),
        (
            (3, 2),
            (1, 2, 1, 1),
            [[[1, 2, 2], [3, 4, 0]], [[1, 2, 3], [2, 4, 0]], [[1, 2, 4], [2, 3, 0]]],
        ),
        (
            (4, 2),
            (2, 2, 2),
            [
                [[1, 1, 2, 3], [2, 3, 0, 0]],
                [[1, 2, 2, 3], [1, 3, 0, 0]],
                [[1, 2, 3, 3], [1, 2, 0, 0]],
                [[1, 1, 2, 2], [3, 3, 0, 0]],
                [[1, 1, 3, 3], [2, 2, 0, 0]],
            ],
        ),
        (
            (4, 3),
            (2, 2, 2),
            [
                [[1, 1, 2, 0], [2, 3, 3, 0]],
                [[1, 1, 2, 3], [2, 3, 0, 0]],
                [[1, 1, 3, 0], [2, 2, 3, 0]],
                [[1, 2, 2, 0], [1, 3, 3, 0]],
                [[1, 2, 3, 0], [1, 2, 3, 0]],
                [[1, 2, 2, 3], [1, 3, 0, 0]],
                [[1, 2, 3, 3], [1, 2, 0, 0]],
                [[1, 1, 2, 2], [3, 3, 0, 0]],
                [[1, 1, 3, 3], [2, 2, 0, 0]],
            ],
        ),
        ((4, 2), (2, 2, 2, 2), []),
    ],
)
def test_ssyt(partition, conjugacy_class, result):
    """Test the normal usage of ssyt"""
    assert ap.ssyt(partition, conjugacy_class) == result


@pytest.mark.parametrize(
    "tableau, conjugacy_class",
    [
        ([[1, 2, 2], [1, 0, 0]], (2, 2)),
        ([[1, 1, 1], [2, 0, 0]], (3, 1)),
        ([[1, 1, 2], [1, 0, 0]], (3, 1)),
        ([[1, 1, 1], [2, 0, 0], [2, 0, 0]], (3, 2)),
        ([[1, 2, 2], [1, 0, 0], [1, 0, 0]], (3, 2)),
        ([[1, 2, 2], [3, 4, 0]], (1, 2, 1, 1)),
        ([[1, 1, 2, 2], [3, 3, 0, 0]], (2, 2, 2)),
        ([[1, 1, 3, 3], [2, 2, 0, 0]], (2, 2, 2)),
        ([[1, 2, 3, 3], [1, 2, 0, 0]], (2, 2, 2)),
    ],
)
def test_bad_mapping_false(tableau, conjugacy_class):
    "Test bad_mapping for well mapped tableaux, returns False"
    assert not ap.bad_mapping(tableau, conjugacy_class)


@pytest.mark.parametrize(
    "tableau, conjugacy_class",
    [
        ([[1, 1, 2], [2, 0, 0]], (2, 2)),
        ([[1, 1, 2], [1, 0, 0], [2, 0, 0]], (3, 2)),
        ([[1, 2, 3], [2, 4, 0]], (1, 2, 1, 1)),
        ([[1, 2, 4], [2, 3, 0]], (1, 2, 1, 1)),
        ([[1, 1, 2, 3], [2, 3, 0, 0]], (2, 2, 2)),
        ([[1, 2, 2, 3], [1, 3, 0, 0]], (2, 2, 2)),
    ],
)
def test_bad_mapping_true(tableau, conjugacy_class):
    "Test bad_mapping for wrong mapped tableaux, returns True"
    assert ap.bad_mapping(tableau, conjugacy_class)


@pytest.mark.parametrize(
    "partition, conjugacy_class, caracter",
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
        ((5,3), (7,1), 0),
        ((4, 2, 1, 1), (7,1), -1),
    ],
)
def test_murn_naka(partition, conjugacy_class, caracter):
    "Test murn_naka based on the outputs form weingarten mathematica package"
    assert ap.murn_naka(partition, conjugacy_class) == caracter


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
def test_sn_dimension(partition, dimension):
    "Test sn_dimension based on the outputs form weingarten mathematica package"
    assert ap.sn_dimension(partition) == dimension


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
def test_sn_dimension_murn_naka(partition):
    "Reconcil sn_dimension and murn_naka for a class mu of ones"
    mu = sum(partition) * [1]
    assert ap.sn_dimension(partition) == ap.murn_naka(partition, mu)


@pytest.mark.parametrize(
    "partition, dimension",
    [
        ((1,), 17),
        ((5,), 20349),
        ((3, 2), 65892),
        ((11, 3), 7979191740),
        ((8, 2, 2), 2489487616),
        ((7, 3, 2, 2), 98023574880),
        ((5, 4, 2, 1, 1, 1), 86129014608),
        ((6, 5, 4), 168562278720),
        ((11, 3, 2), 405097426800),
    ],
)
def test_ud_dimension(partition, dimension):
    "Test ud_dimension based on the outputs form weingarten mathematica package"
    assert ap.ud_dimension(partition, 17) == dimension


@pytest.mark.parametrize(
    "partition",
    [
        ((3, 2)),
        ((11, 3)),
        ((8, 2, 2)),
        ((7, 3, 2, 2)),
        ((5, 4, 2, 1, 1, 1)),
        ((6, 5, 4)),
    ],
)
def test_ud_dimension_wrong_dimension(partition):
    "ud_dimension returns 0 if the dimension is lower than the number of parts in the partition"
    assert not ap.ud_dimension(partition, len(partition) - 1)


@pytest.mark.parametrize(
    "conjugacy, dimension, num, denum",
    [
        ((1,), 7, 1, 7),
        ((2,), 7, -1, 336),
        ((1, 1), 7, 1, 48),
        ((2, 1), 7, -1, 2160),
        ((1, 1, 1), 7, 47, 15120),
        ((3, 1), 7, 19, 846720),
        ((2, 2), 7, 11, 846720),
        ((3, 2), 7, -61, 69854400),
        ((3, 1, 1), 7, 1, 249480),
        ((2, 2, 1), 7, 47, 19958400),
        ((3, 3), 7, 311, 3353011200),
        ((2, 1, 1, 1, 1), 7, -421, 191600640),
        ((1, 1, 1, 1, 1, 1), 7, 82477, 6706022400),
        ((4, 3), 7, -17, 792529920),
        ((7,1), 8, 151, 317011968000),
    ],
)
def test_weingarten_class(conjugacy, dimension, num, denum):
    "Test weingarten_class based on the outputs form weingarten mathematica package"
    assert ap.weingarten_class(conjugacy, dimension) == Fraction(num, denum)


@pytest.mark.parametrize(
    "cycle, degree, num, denum",
    [
        (Permutation(0, 1), 2, -1, 336),
        (Permutation(0), 2, 1, 48),
        (Permutation(0, 1, 2), 3, 1, 7560),
        (Permutation(1, 2), 3, -1, 2160),
        (Permutation(2), 3, 47, 15120),
        (Permutation(0, 1, 2), 4, 19, 846720),
        (Permutation(0, 2)(1, 3), 4, 11, 846720),
        (Permutation(2), 4, 403, 846720),
        (Permutation(4, 1), 5, -1739, 139708800),
        (Permutation(2), 5, 1499, 19958400),
        (Permutation(3, 4, 5), 6, 5167, 6706022400),
    ],
)
def test_weingarten_element(cycle, degree, num, denum):
    "Test weingarten_element based on the outputs form weingarten mathematica package"
    assert ap.weingarten_element(cycle, degree, 7) == Fraction(num, denum)


@pytest.mark.parametrize(
    "cycle, degree",
    [
        (Permutation(0, 1), 2),
        (Permutation(0), 2),
        (Permutation(0, 1, 2), 3),
        (Permutation(1, 2), 3),
        (Permutation(2), 3),
        (Permutation(0, 1, 2), 4),
        (Permutation(0, 2)(1, 3), 4),
        (Permutation(2), 4),
        (Permutation(4, 1), 5),
        (Permutation(2), 5),
        (Permutation(3, 4, 5), 6),
    ],
)
def test_weingarten_reconciliation(cycle, degree):
    "Numeric reconciliation of weingarten_class and weingarten_element"
    assert ap.weingarten_element(cycle, degree, 9) == ap.weingarten_class(
        ap.get_class(cycle, degree), 9
    )


@pytest.mark.parametrize(
    "cycle, degree",
    [
        (Permutation(0, 1), 2),
        (Permutation(0), 2),
        (Permutation(0, 1, 2), 3),
        (Permutation(1, 2), 3),
        (Permutation(2), 3),
        (Permutation(0, 1, 2), 4),
        (Permutation(0, 2)(1, 3), 4),
        (Permutation(2), 4),
        (Permutation(4, 1), 5),
        (Permutation(2), 5),
        (Permutation(3, 4, 5), 6),
    ],
)
def test_weingarten_reconciliation_symbolic(cycle, degree):
    "Symbolic reconciliation of weingarten_class and weingarten_element"
    d = Symbol("d")
    assert ap.weingarten_element(cycle, degree, d) == ap.weingarten_class(
        list(ap.get_class(cycle, degree)), d
    )


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
def test_gram_orthogonality(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    d = Symbol("d")
    group = lambda n: SymmetricGroup(n).generate_schreier_sims()
    character = lambda g, d, n: d ** (g.cycles)
    weingarten = lambda g, d, n: ap.weingarten_element(g, n, d)
    orthogonality = sum(character(g, d, n) * weingarten(g, d, n) for g in group(n))
    assert simplify(orthogonality) == 1


@pytest.mark.parametrize(
    "target, shuffled, weingarten",
    [
        (
            (1, 2),
            (1, 2),
            {
                (1,): 1,
            },
        ),
        ("ijkj", "ijkj", {(1, 1): 1, (2,): 1}),
        ((1, 2, 3, 2), (1, 2, 3, 2), {(1, 1): 1, (2,): 1}),
        ("ijkl", "ijkl", {(1, 1): 1}),
        ("ijkl", "ilkj", {(2,): 1}),
        ("ijklmn", "ijklmn", {(1, 1, 1): 1}),
        ("iljmkn", "imjnkl", {(3,): 1}),
        ("iljlkm", "iljmkl", {(3,): 1, (2, 1): 1}),
        ((1, 4, 2, 4, 3, 4), (1, 4, 2, 4, 3, 4), {(1, 1, 1): 1, (2, 1): 3, (3,): 2}),
    ],
)
def test_haar_integral_hand(target, shuffled, weingarten):
    "Test integral of Haar distribution unitaries against hand-calculated integrals"
    dimension = Symbol("d")
    integral = sum(
        frequency * ap.weingarten_class(conjugacy, dimension)
        for conjugacy, frequency in weingarten.items()
    )
    numerator, denominator = fraction(simplify(integral))
    integral = factor(numerator) / factor(denominator)
    assert ap.haar_integral(target, shuffled, dimension) == integral
