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
Circular ensembles Python interface
"""

from fractions import Fraction
from functools import lru_cache
from sympy import Symbol
from sympy.combinatorics import Permutation
from haarpy import weingarten_orthogonal, weingarten_symplectic


@lru_cache
def weingarten_circular_orthogonal(
    permutation: Permutation, coe_dimension: Symbol
) -> Symbol:
    """Returns the circular orthogonal ensembles Weingarten functions

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        coe_dimension (int): The dimension of the COE

    Returns:
        Symbol : The Weingarten function
    """
    return weingarten_orthogonal(permutation, coe_dimension + 1)


@lru_cache
def weingarten_circular_symplectic(
    permutation: Permutation, cse_dimension: Symbol
) -> Symbol:
    """Returns the circular symplectic ensembles Weingarten functions

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        cse_dimension (int): The dimension of the CSE

    Returns:
        Symbol : The Weingarten function
    """
    symplectic_dimension = (
        (2 * cse_dimension - 1) / 2
        if isinstance(cse_dimension, Symbol)
        else Fraction(2 * cse_dimension - 1, 2)
    )
    return weingarten_symplectic(permutation, symplectic_dimension)
