# Copyright 2026 Polyquantique

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
Quantum tests
"""

import pytest
from random import randint, seed
import haarpy as ap
from sympy import Symbol

seed(137)
d = Symbol('d')


@pytest.mark.parametrize(
    "sequences",
    [
        ((0,1), (0,0)),
        ((0,1,0), (0,1,1)),
        ((0,1,2), (0,1,1)),
        ((1,0,0), (0,1,2)),
        ((0,1,0), (0,1,2)),
        ((0,1,2), (0,1,0)),
        ((0,1,0,1), (0,1,0,0)),
        ((1,0,0,2), (0,1,2,1)),
        ((0,0,0,0), (0,1,2,3)),
        ((0,0,0,1), (0,1,2,3)),
        ((0,0,1,1), (0,1,2,3)),
        ((1,2,0,1), (0,1,2,3)),
        ((1,2,1,2), (1,2,0,1)),
        ((1,2,0,1), (1,2,1,2)),
    ],
)
def test_free_symmetric_matrix_trivially_zero(sequences):
    """Integrals are trivially $0$ if two succesive matrix elements
    (including first and last) are on the same row or column
    """
    assert not ap.haar_integral_free_symmetric(sequences, d)
    assert not ap.haar_integral_free_symmetric(sequences, randint(5,20))


# TEST BOTH NUMERIC AND SYMBOL
# TEST ZERO CASES
# TEST NON ZERO CASES
# TEST THE RELATION BETWEEN BOTH FUNCTIONS