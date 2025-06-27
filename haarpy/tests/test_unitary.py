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
    """Test the normal usage of get_conjugacy_class"""
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
    """Test the degree parameter TypeError"""
    with pytest.raises(TypeError, match=".*degree must be of type int.*"):
        ap.get_conjugacy_class(cycle, degree)


@pytest.mark.parametrize("degree", range(-3, 1))
def test_get_conjugacy_class_degree_value_error(degree):
    """Test the degree parameter ValueError"""
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
    """Test the cycle parameter TypeError"""
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
    """Test the cycle parameter ValueError if permutation maximum value is greater than the degree"""
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
    """Test the normal usage of derivative_tableaux"""
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
    """Test the normal usage of semi_standard_young_tableaux"""
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
def test_representation_dimension(partition, dimension):
    "Test representation_dimension based on the outputs form weingarten mathematica package"
    assert ap.representation_dimension(partition, 17) == dimension


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
def test_representation_dimension_wrong_dimension(partition):
    "representation_dimension returns 0 if the dimension is lower than the number of parts in the partition"
    assert not ap.representation_dimension(partition, len(partition) - 1)


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
        ((7, 1), 8, 151, 317011968000),
    ],
)
def test_weingarten_class(conjugacy, dimension, num, denum):
    "Test weingarten_class based on the outputs form weingarten mathematica package"
    assert ap.weingarten_class(conjugacy, dimension) == Fraction(num, denum)


@pytest.mark.parametrize(
    "cycle, degree, dimension, num, denum",
    [
        (Permutation(0, 1), 2, 7, -1, 336),
        (Permutation(0), 2, 7, 1, 48),
        (Permutation(0, 1, 2), 3, 7, 1, 7560),
        (Permutation(1, 2), 3, 7, -1, 2160),
        (Permutation(2), 3, 7, 47, 15120),
        (Permutation(0, 1, 2), 4, 7, 19, 846720),
        (Permutation(0, 2)(1, 3), 4, 7, 11, 846720),
        (Permutation(2), 4, 7, 403, 846720),
        (Permutation(4, 1), 5, 7, -1739, 139708800),
        (Permutation(2), 5, 7, 1499, 19958400),
        (Permutation(3, 4, 5), 6, 7, 5167, 6706022400),
    ],
)
def test_weingarten_element(cycle, degree, dimension, num, denum):
    "Test weingarten_element based on the outputs form weingarten mathematica package"
    assert ap.weingarten_element(cycle, degree, dimension) == Fraction(num, denum)


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
        ap.get_conjugacy_class(cycle, degree), 9
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
        ap.get_conjugacy_class(cycle, degree), d
    )


@pytest.mark.parametrize(
    "partition, dimension",
    [
        ((3, 2), 1.0),
        ((3, 1, 1), 'a'),
        ((2, 2, 1), (1, 0)),
        ((3, 3), (8,)),
    ],
)
def test_weingarten_class_dimension_typeError(partition, dimension):
    with pytest.raises(
        TypeError,
        match=".*unitary_dimension must be an instance of int or sympy.Symbol*",
    ):
        ap.weingarten_class(partition, dimension)


@pytest.mark.parametrize(
    "cycle, degree, dimension",
    [
        (Permutation(0, 1, 2), 4, 1.0),
        (Permutation(0, 2)(1, 3), 4, 'a'),
        (Permutation(2), 4, (0, 1)),
        (Permutation(4, 1), 5, (8,)),
    ],
)
def test_weingarten_element_dimension_typeError(cycle, degree, dimension):
    with pytest.raises(
        TypeError,
        match=".*unitary_dimension must be an instance of int or sympy.Symbol*",
    ):
        ap.weingarten_element(cycle, degree, dimension)


@pytest.mark.parametrize("n", range(2,5))
def test_gram_orthogonality_elements(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    d = Symbol("d")
    orthogonality = sum(
        d ** (g.cycles) * ap.weingarten_element(g, n, d)
        for g in SymmetricGroup(n).generate_schreier_sims()
        )
    assert simplify(orthogonality) == 1


@pytest.mark.parametrize("n", range(2,10))
def test_gram_orthogonality_classes(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    d = Symbol("d")
    weight = lambda g : d ** (g.cycles) * ap.weingarten_class(ap.get_conjugacy_class(g, n), d)
    orthogonality = sum(
        len(c) * weight(c.pop())
        for c in SymmetricGroup(n).conjugacy_classes()
        )
    assert simplify(orthogonality) == 1


@pytest.mark.parametrize(
    "sequences, weingarten_map",
    [
        (((1,), (2,), (1,), (2,)), {(1,): 1}),
        (("ik", "jj", "ik", "jj"), {(1, 1): 1, (2,): 1}),
        (((1, 3), (2, 2), (1, 3), (2, 2)), {(1, 1): 1, (2,): 1}),
        (("ik", "jl", "ik", "jl"), {(1, 1): 1}),
        (("ik", "jl", "ik", "lj"), {(2,): 1}),
        (("ikm", "jln", "ikm", "jln"), {(1, 1, 1): 1}),
        (("ijk", "lmn", "ijk", "mnl"), {(3,): 1}),
        (("ijk", "llm", "ijk", "lml"), {(3,): 1, (2, 1): 1}),
        (((1, 2, 3), (4, 4, 4), (1, 2, 3), (4, 4, 4)), {(1, 1, 1): 1, (2, 1): 3, (3,): 2}),
        (((1, 1, 3), (1, 4, 4), (1, 2, 3), (1, 4, 4)), {(3,): 0}),
        (((1, 2, 3), (4, 4, 4), (1, 2, 4), (4, 4, 4)), {(3,): 0}),
        (((1, 2, 3), (4, 4, 4), (1, 2), (4, 4)), {(3,): 0}),
    ],
)
def test_haar_integral_hand(sequences, weingarten_map):
    "Test integral of Haar distribution unitaries against hand-calculated integrals"
    dimension = Symbol("d")
    integral = sum(
        frequency * ap.weingarten_class(conjugacy, dimension)
        for conjugacy, frequency in weingarten_map.items()
    )
    numerator, denominator = fraction(simplify(integral))
    integral = factor(numerator) / factor(denominator)
    assert ap.haar_integral(sequences, dimension) == integral


@pytest.mark.parametrize(
    "sequence",
    [
        ((1,), (1,), (1,)),
        ((1, 1, 1), (1, 1), (1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 1), (1, 1), (1, 1, 1)),
    ],
)
def test_haar_integral_wrong_format(sequence):
    """Test wrong tuple format ValueError"""
    dimension = Symbol("d")
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral(sequence, dimension)
