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
Circular ensembles tests
"""

from fractions import Fraction
from sympy import Symbol, simplify
from sympy.combinatorics import SymmetricGroup
import pytest
import haarpy as ap

d = Symbol('d')


@pytest.mark.parametrize("half_degree", range(1,3))
def test_weingarten_circular_orthogonal_hyperoctahedral_symbolic(half_degree):
    """Symbolic validation of COE Weingarten function against results shown in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    if half_degree == 1:
        for permutation in SymmetricGroup(2*half_degree).generate():
            assert ap.weingarten_circular_orthogonal(permutation, d) == 1/(d+1)
    else:
        for permutation in SymmetricGroup(2*half_degree).generate():
            hyperoctahedral = ap.hyperoctahedral(half_degree)
            coefficient = 1/(d*(d+1)*(d+3))
            assert ap.weingarten_circular_orthogonal(permutation, d) == (
                simplify((d+2)*coefficient) if permutation in hyperoctahedral
                else -coefficient
            )


@pytest.mark.parametrize("half_degree", range(1,3))
def test_weingarten_circular_orthogonal_hyperoctahedral_numeric(half_degree):
    """Symbolic validation of COE Weingarten function against results shown in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    if half_degree == 1:
        for permutation in SymmetricGroup(2*half_degree).generate():
            assert ap.weingarten_circular_orthogonal(permutation, 7) == 1/(7+1)
    else:
        for permutation in SymmetricGroup(2*half_degree).generate():
            hyperoctahedral = ap.hyperoctahedral(half_degree)
            coefficient = Fraction(1,(7*(7+1)*(7+3)))
            assert ap.weingarten_circular_orthogonal(permutation, 7) == (
                simplify((7+2)*coefficient) if permutation in hyperoctahedral
                else -coefficient
            )


@pytest.mark.parametrize("half_degree", range(1,3))
def test_weingarten_circular_symplectic_hyperoctahedral_symbolic(half_degree):
    """Symbolic validation of CSE Weingarten function against results shown in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    if half_degree == 1:
        for permutation in SymmetricGroup(2*half_degree).generate():
            assert ap.weingarten_circular_symplectic(permutation, d) == (
                permutation.signature()/(2*d-1)
            )
    else:
        for permutation in SymmetricGroup(2*half_degree).generate():
            hyperoctahedral = ap.hyperoctahedral(half_degree)
            coefficient = permutation.signature()/(d*(2*d-1)*(2*d-3))
            assert ap.weingarten_circular_symplectic(permutation, d) == (
                simplify((d-1)*coefficient) if permutation in hyperoctahedral
                else coefficient/2
            )


@pytest.mark.parametrize("half_degree", range(1,3))
def test_weingarten_circular_symplectic_hyperoctahedral_numeric(half_degree):
    """Symbolic validation of CSE Weingarten function against results shown in
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    if half_degree == 1:
        for permutation in SymmetricGroup(2*half_degree).generate():
            assert ap.weingarten_circular_symplectic(permutation, 7) == Fraction(
                permutation.signature(),
                (2*7-1),
            )
    else:
        for permutation in SymmetricGroup(2*half_degree).generate():
            hyperoctahedral = ap.hyperoctahedral(half_degree)
            coefficient = Fraction(permutation.signature(), (7*(2*7-1)*(2*7-3)))
            assert ap.weingarten_circular_symplectic(permutation, 7) == (
                simplify((7-1)*coefficient) if permutation in hyperoctahedral
                else coefficient*Fraction(1,2)
            )
