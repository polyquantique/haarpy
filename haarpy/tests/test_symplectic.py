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

from sympy.combinatorics import Permutation
import pytest
import haarpy as ap


@pytest.mark.parametrize(
    "permutation, partition",
    [
        (Permutation(3), (1,1)),
        (Permutation(3)(0,1), (1,1)),
        (Permutation(0,1,2,3), (1,1)),
        (Permutation(3)(0,1,2), (2,)),
        (Permutation(0,1,2,3,4,5), (3,)),
        (Permutation(5)(0,4), (3,)),
        (Permutation(5)(0,1,4), (3,)),
        (Permutation(5)(0,1,2,3,4), (3,)),
        (Permutation(5)(0,1,2,3,4), (1,1,1)),
        (Permutation(5), (1,1,1)),
        (Permutation(5), (2,1)),
        (Permutation(5), (2,1)),
        (Permutation(7), (2,2)),
        (Permutation(7)(0,1), (2,1,1)),
    ],
)
def test_twisted_spherical_image(permutation, partition):
    """Validates that the twisted spherical function is the image of the zonal spherical function as seen in 
    `Matsumoto. Weingarten calculus for matrix ensembles associated with compact symmetric spaces: 
    <https://arxiv.org/abs/1301.5401>`_
    """
    conjugate_partition = tuple(
        sum(1 for i in partition if i > j) for j in range(partition[0])
    )
    assert ap.twisted_spherical_function(
        permutation,
        partition,
    ) == ap.zonal_spherical_function(
        permutation,
        conjugate_partition,
    )