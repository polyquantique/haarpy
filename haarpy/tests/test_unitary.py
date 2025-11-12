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
import haarpy as ap
from fractions import Fraction
from sympy.combinatorics import Permutation
from sympy import Symbol, simplify, fraction, factor
from sympy.combinatorics.named_groups import SymmetricGroup


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
        ap.get_conjugacy_class(cycle, cycle.size), 9
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
    d = Symbol("d")
    assert ap.weingarten_unitary(cycle, d) == ap.weingarten_unitary(
        ap.get_conjugacy_class(cycle, cycle.size), d
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
        (Permutation(0, 2)(1, 3), 'a'),
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
        (1,2,"a"),
        (3, (1,2), 4),
        "abc",
    ]
)
def test_weingarten_unitary_cycle_type_error(cycle):
    "Test the type error for wrong permutation input"
    with pytest.raises(TypeError):
        ap.weingarten_unitary(cycle, Symbol('d'))


@pytest.mark.parametrize("n", range(2,5))
def test_gram_orthogonality_elements(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    d = Symbol("d")
    orthogonality = sum(
        d ** (g.cycles) * ap.weingarten_unitary(g, d)
        for g in SymmetricGroup(n).generate_schreier_sims()
        )
    assert simplify(orthogonality) == 1


@pytest.mark.parametrize("n", range(2,10))
def test_gram_orthogonality_classes(n):
    "Test the orthogonality relation between Weingarten matrix and Graham matrix"
    d = Symbol("d")
    weight = lambda g : d ** (g.cycles) * ap.weingarten_unitary(ap.get_conjugacy_class(g, n), d)
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
        frequency * ap.weingarten_unitary(conjugacy, dimension)
        for conjugacy, frequency in weingarten_map.items()
    )
    numerator, denominator = fraction(simplify(integral))
    integral = factor(numerator) / factor(denominator)
    assert ap.haar_integral_unitary(sequences, dimension) == integral


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
    dimension = Symbol("d")
    with pytest.raises(ValueError, match="Wrong tuple format"):
        ap.haar_integral_unitary(sequence, dimension)
