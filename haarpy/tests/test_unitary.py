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
Unitary tests
"""

import pytest
from fractions import Fraction
from random import seed, randint
from sympy.combinatorics import Permutation
from sympy import Symbol, simplify, fraction, factor
from sympy.combinatorics.named_groups import SymmetricGroup
import haarpy as ap
from haarpy.unitary import _column_integral_unitary

seed(137)
d = Symbol("d")


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
def test_weingarten_unitary_class(conjugacy, dimension, num, denum):
    "Test weingarten_unitary based on the outputs form weingarten mathematica package"
    assert ap.weingarten_unitary(conjugacy, dimension) == Fraction(num, denum)


@pytest.mark.parametrize(
    "cycle, dimension, num, denum",
    [
        (Permutation(0, 1), 7, -1, 336),
        (Permutation(1), 7, 1, 48),
        (Permutation(0, 1, 2), 7, 1, 7560),
        (Permutation(1, 2), 7, -1, 2160),
        (Permutation(2), 7, 47, 15120),
        (Permutation(3)(0, 1, 2), 7, 19, 846720),
        (Permutation(0, 2)(1, 3), 7, 11, 846720),
        (Permutation(3), 7, 403, 846720),
        (Permutation(4, 1), 7, -1739, 139708800),
        (Permutation(4), 7, 1499, 19958400),
        (Permutation(3, 4, 5), 7, 5167, 6706022400),
    ],
)
def test_weingarten_unitary_element(cycle, dimension, num, denum):
    "Test weingarten_unitary based on the outputs form weingarten mathematica package"
    assert ap.weingarten_unitary(cycle, dimension) == Fraction(num, denum)


@pytest.mark.parametrize(
    "cycle",
    [
        Permutation(0, 1),
        Permutation(0),
        Permutation(0, 1, 2),
        Permutation(1, 2),
        Permutation(2),
        Permutation(0, 1, 2),
        Permutation(0, 2)(1, 3),
        Permutation(2),
        Permutation(4, 1),
        Permutation(2),
        Permutation(3, 4, 5),
    ],
)
def test_weingarten_reconciliation_numeric(cycle):
    "Numeric reconciliation of permutation and conjugacy class input"
    assert ap.weingarten_unitary(cycle, 9) == ap.weingarten_unitary(
        ap.get_conjugacy_class(cycle), 9
    )


@pytest.mark.parametrize(
    "cycle",
    [
        Permutation(0, 1),
        Permutation(0),
        Permutation(0, 1, 2),
        Permutation(1, 2),
        Permutation(2),
        Permutation(0, 1, 2),
        Permutation(0, 2)(1, 3),
        Permutation(2),
        Permutation(4, 1),
        Permutation(2),
        Permutation(3, 4, 5),
    ],
)
def test_weingarten_reconciliation_symbolic(cycle):
    "Symbolic reconciliation of permutation and conjugacy class input"
    assert ap.weingarten_unitary(cycle, d) == ap.weingarten_unitary(
        ap.get_conjugacy_class(cycle), d
    )


@pytest.mark.parametrize(
    "partition, dimension",
    [
        ((3, 2), 1.0),
        ((3, 1, 1), "a"),
        ((2, 2, 1), (1, 0)),
        ((3, 3), (8,)),
    ],
)
def test_weingarten_unitary_class_dimension_type_error(partition, dimension):
    "Test type error for for wrong unitary dimension input"
    with pytest.raises(
        TypeError,
        match=".*unitary_dimension must be an instance of int or sympy.Expr*",
    ):
        ap.weingarten_unitary(partition, dimension)


@pytest.mark.parametrize(
    "cycle, dimension",
    [
        (Permutation(0, 1, 2), 1.0),
        (Permutation(0, 2)(1, 3), "a"),
        (Permutation(2), (0, 1)),
        (Permutation(4, 1), (8,)),
    ],
)
def test_weingarten_unitary_element_dimension_type_error(cycle, dimension):
    "Test type error for for wrong unitary dimension input"
    with pytest.raises(
        TypeError,
        match=".*unitary_dimension must be an instance of int or sympy.Expr*",
    ):
        ap.weingarten_unitary(cycle, dimension)


@pytest.mark.parametrize(
    "cycle",
    [
        (1, 2, "a"),
        (3, (1, 2), 4),
        "abc",
    ],
)
def test_weingarten_unitary_cycle_type_error(cycle):
    "Test the type error for wrong permutation input"
    with pytest.raises(TypeError):
        ap.weingarten_unitary(cycle, d)


@pytest.mark.parametrize("n", range(2, 5))
def test_gram_orthogonality_elements(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    orthogonality = sum(
        d ** (g.cycles) * ap.weingarten_unitary(g, d)
        for g in SymmetricGroup(n).generate_schreier_sims()
    )
    assert simplify(orthogonality) == 1


@pytest.mark.parametrize("n", range(2, 10))
def test_gram_orthogonality_classes(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    weight = lambda g: d ** (g.cycles) * ap.weingarten_unitary(ap.get_conjugacy_class(g), d)
    orthogonality = sum(len(c) * weight(c.pop()) for c in SymmetricGroup(n).conjugacy_classes())
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
    integral = sum(
        frequency * ap.weingarten_unitary(conjugacy, d)
        for conjugacy, frequency in weingarten_map.items()
    )
    numerator, denominator = fraction(simplify(integral))
    integral = factor(numerator) / factor(denominator)
    assert ap.haar_integral_unitary(sequences[:2], sequences[2:], d) == integral


@pytest.mark.parametrize(
    "sequence",
    [
        ((1,), (1,), (1,)),
        ((1, 1, 1), (1, 1), (1, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 1, 1), (1, 1), (1, 1, 1)),
    ],
)
def test_haar_integral_wrong_format(sequence):
    "Test wrong tuple format ValueError"
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral_unitary(sequence[:2], sequence[2:], d)


@pytest.mark.parametrize(
    "monomial, monomial_conj, structure, algo, result",
    [
        (((2, 0, 0, 2), (0, 1, 1, 0), (2, 0, 0, 0)), ((3, 0, 0, 2), (0, 1, 1, 0), (2, 0, 0, 0)), "matrix", "Collins", 0),
        (((1, 0), (0, 1)), ((1, 0), (0, 2)), "matrix", "Collins", 0),
        (((1, 1, 1), (1, 1, 1), (1, 1, 1)), ((1, 2, 0), (1, 1, 1), (1, 1, 1)), "matrix", "Collins", 0),
        (((),), ((),), "matrix", "Collins", 1),
        (((), ()), ((), ()), "matrix", "Collins", 1),
        (((0,),), ((0,),), "matrix", "Collins", 1),
        (((0, 0), (0, 0)), ((0, 0), (0, 0)), "matrix", "Collins", 1),
        (((1, 1, 1),), ((1, 1, 2),), "matrix", "Collins", 0),
        (((1, 1, 2, 2), (1, 2, 2, 2)), ((1, 1, 1, 2), (1, 2, 2, 2)), "sequences", "Collins", 0),
        (((1, 2, 2, 2), (2, 2, 2, 2)), ((1, 2, 2, 2), (2, 2, 2, 1)), "sequences", "Collins", 0),
        (((1, 1, 1, 1, 1), (2, 2, 2, 2, 2)), ((2, 2, 2, 2, 2), (1, 1, 1, 1, 1)), "sequences", "Collins", 0),
        (((), ()), ((), ()), "sequences", "Collins", 1),
        (((1, 0), (0, 1)), ((1, 0), (0, 2)), "matrix", "Gorin", 0),
        (((),), ((),), "matrix", "Gorin", 1),
        (((), ()), ((), ()), "matrix", "Gorin", 1),
        (((), (), ()), ((), (), ()), "matrix", "Gorin", 1),
        (((0,),), ((0,),), "matrix", "Gorin", 1),
        (((0, 0), (0, 0)), ((0, 0), (0, 0)), "matrix", "Gorin", 1),
        (((1, 1, 2, 2), (1, 2, 2, 2)), ((1, 1, 1, 2), (1, 2, 2, 2)), "sequences", "Gorin", 0),
        (((1, 2, 2, 2), (2, 2, 2, 2)), ((1, 2, 2, 2), (2, 2, 2, 1)), "sequences", "Gorin", 0),
        (((1, 1, 1, 1, 1), (2, 2, 2, 2, 2)), ((2, 2, 2, 2, 2), (1, 1, 1, 1, 1)), "sequences", "Gorin", 0),
        (((), ()), ((), ()), "sequences", "Gorin", 1),
        (((1, 1, 1),), ((1, 1, 2),), "matrix", "Gorin", 0),
    ],
)
def test_haar_integral_unitary_trivial(monomial, monomial_conj, structure, algo, result):
    "Test the trivial integrals"
    assert ap.haar_integral_unitary(monomial, monomial_conj, d, algo, structure) == result


@pytest.mark.parametrize(
    "monomial, monomial_conj, structure",
    [
        (((), ()), ((), ()), "sequences"),
        (((0, 0, 1, 1), (1, 0, 0, 1)), ((0, 1, 0, 1), (0, 1, 0, 1)), "sequences"),
        (((0, 0, 1, 1, 2, 2), (1, 2, 2, 0, 1, 1)), ((2, 2, 0, 0, 1, 1), (1, 2, 2, 1, 0, 1)), "sequences"),
        (((0, 0, 1, 1, 2), (4, 4, 4, 4, 5)), ((0, 0, 1, 1, 2), (4, 4, 4, 4, 5)), "sequences"),
        (((2, 0), (0, 2)), ((2, 0), (0, 2)), "matrix"),
        (((2, 0), (0, 2)), ((0, 2), (2, 0)), "matrix"),
        (((2, 0), (0, 2)), ((1, 1), (1, 1)), "matrix"),
        (((1, 1), (1, 1)), ((1, 1), (1, 1)), "matrix"),
        (((2, 0, 0), (0, 2, 2), (2, 0, 0)), ((1, 1, 0), (1, 1, 2), (2, 0, 0)), "matrix"),
        (((1, 1, 1), (1, 1, 1), (1, 1, 1)), ((3, 0, 0), (0, 3, 0), (0, 0, 3)), "matrix"),
        (((1, 1, 1),), ((1, 1, 1),), "matrix"),
    ],
)
def test_haar_integral_unitary_gorin_collins_reconcile(monomial, monomial_conj, structure):
    "Test that Gorin and Collins algorithms reconcile"
    dimension_num = randint(8, 15)
    gorin_num = ap.haar_integral_unitary(monomial, monomial_conj, dimension_num, "gorin", structure)
    gorin_symb = ap.haar_integral_unitary(monomial, monomial_conj, d, "gorin", structure)
    collins_num = ap.haar_integral_unitary(monomial, monomial_conj, dimension_num, "collins", structure)
    collins_symb = ap.haar_integral_unitary(monomial, monomial_conj, d, "collins", structure)

    assert collins_num
    assert gorin_symb == collins_symb
    assert gorin_num == collins_num


@pytest.mark.parametrize(
    "value, parameter",
    [
        ("a", "unitary_dimension"),
        ([1, 1], "unitary_dimension"),
        (1, "algorithm"),
        ([1, 1], "algorithm"),
        (1, "structure"),
        ([1, 1], "structure"),
    ],
)
def test_haar_integral_unitary_type_error(value, parameter):
    "Test Haar integral algorithm type error"
    with pytest.raises(TypeError):
        if parameter == "unitary_dimension":
            ap.haar_integral_unitary(((), ()), ((), ()), unitary_dimension=value)
        if parameter == "algorithm":
            ap.haar_integral_unitary(((), ()), ((), ()), d, algorithm=value)
        if parameter == "structure":
            ap.haar_integral_unitary(((), ()), ((), ()), d, structure=value)


@pytest.mark.parametrize(
    "string_value, parameter",
    [
        ("ggorin", "algorithm"),
        ("ccollins", "algorithm"),
        ("ssequences", "structure"),
        ("mmatrix", "structure"),
    ],
)
def test_haar_integral_unitary_string_value_error(string_value, parameter):
    "Test Haar integral algorithm and structure string value error"
    error_msg = (
        "The 'algorithm' must be either 'Collins' or 'Gorin'.\n"
        "The 'structure' must be either 'matrix' or 'sequences'"
    )
    with pytest.raises(ValueError, match=error_msg):
        if parameter == "algorithm":
            ap.haar_integral_unitary(((), ()), ((), ()), d, algorithm=string_value)
        if parameter == "structure":
            ap.haar_integral_unitary(((), ()), ((), ()), d, structure=string_value)


@pytest.mark.parametrize(
    "mono, mono_conj",
    [
        (((1,),), ((1,),(1,))),
        (((1,),(1,)), ((1,),)),
        (((1,),(1,2)), ((1,),(1,))),
        (((1,),(1,)), ((1,),(1,2))),
    ],
)
def test_haar_integral_unitary_tuple_format_value_error(mono, mono_conj):
    "Test Haar integral tuple format value error"
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral_unitary(mono, mono_conj, d)


@pytest.mark.parametrize(
    "power_matrix, power_matrix_conj",
    [
        ("a", ((1, 1), (1, 1))),
        (((1, 1), (1, 1)), range(4)),
        (("a", "b"), ((1, 1), (1, 1))),
        (((1, 1), (1, "a")), ((1, 1), (1, 1))),
        (((1, 1), (1, -1)), ((1, 1), (1, 1))),
    ],
)
def test_haar_integral_unitary_power_matrix_value_error(power_matrix, power_matrix_conj):
    "Test Haar integral power matrix value error"
    with pytest.raises(ValueError, match="Wrong power matrix format"):
        ap.haar_integral_unitary(power_matrix, power_matrix_conj, d, structure="matrix")


@pytest.mark.parametrize(
    "column_m, column_n",
    [
        ((1, 1, 1), (2, 1, 1)),
        ((1, 3, 1), (1, 3, 0)),
        ((5, 1, 1, 0, 2), (5, 1, 1, 1, 1)),
    ],
)
def test_zero_column_integral_unitary(column_m, column_n):
    "Vanishing column integral"
    assert not _column_integral_unitary(column_m, column_n, d)


@pytest.mark.parametrize(
    "column_m, column_n",
    [
        ((1, 1, 1), (1, 1, 0, 1)),
        ((1, 3, 1), (1, 2, 1, 1)),
        ((5, 1, 1, 0, 2), (5, 1, 1, 2)),
    ],
)
def test_column_integral_unitary_value_error(column_m, column_n):
    "Test value error if both columns are of different lengths"
    with pytest.raises(ValueError):
        _column_integral_unitary(column_m, column_n, d)
