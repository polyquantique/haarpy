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
Symplectic group Python interface
"""

from functools import lru_cache
from sympy import Symbol
from sympy.combinatorics import Permutation


@lru_cache
def twisted_spherical_function(permutation: Permutation, partition: tuple[int]) -> float:
    """Returns the twisted spherical function of the Gelfand pair (S_2k, H_k)
    as seen in Macdonald's "Symmetric Functions and Hall Polynomials" chapter VII

    Args:
        perm (Permutation): A permutation of the symmetric group S_2k
        partition (tuple[int]): A partition of k

    Returns:
        (float): The twisted spherical function of the given permutation

    Raise:
        TypeError: If degree partition is not a tuple
        TypeError: If permutation argument is not a permutation.
    """


@lru_cache
def weingarten_symplectic(
    permutation: Permutation, symplectic_dimension: Symbol
) -> Symbol:
    """Returns the symplectic Weingarten function

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k
        symplectic_dimension (int): The dimension of the symplectic group

    Returns:
        Symbol : The Weingarten function

    Raise:
        ValueError : If the degree 2k of the symmetric group S_2k is not a factor of 2
    """