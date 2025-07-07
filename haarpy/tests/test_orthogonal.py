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
Orthogonal tests
"""

import pytest
import haarpy as ap
from math import factorial
from sympy.combinatorics import Permutation


@pytest.mark.parametrize("degree", range(1, 8))
def test_hyperoctahedral_order(degree):
    "Hyperoctahedral order test"
    assert ap.hyperoctahedral(degree).order() == 2**degree * factorial(degree)


@pytest.mark.parametrize(
    "degree",
    [
        ("a",),
        ("str", ),
        (0.1, ),
        ((0, 1),),
    ],
)
def test_hyperoctahedral_TypeError(degree):
    "Hyperoctahedral TypeError for wrong degree type"
    with pytest.raises(TypeError):
        ap.hyperoctahedral(degree)


@pytest.mark.parametrize("n", range(2,10))
def test_zonal_spherical_orthogonality(permutation, partition1, partition2):
    """Orthogonality relation for the zonal spherical function as
    seen in Matsumoto's 'Weingarten calculus for matrix ensembles
    associated with compact symmetric spaces'
    """
    assert False


@pytest.mark.parametrize(
    "cycle_type, partition",
    [
        (Permutation(2)(0,1), (1,)),
        (Permutation(4)(0,1,2), (1,1)),
        (Permutation(0,1,2,3,4,5,6), (4,)),
        ((1,1,1), (2,)),
        ((2,1,1,1), (2,)),
        ((4,1), (3,)),
        ((3,3,1), (3,)),
    ]
)
def test_zonal_spherical_degree_error(cycle_type, partition):
    "Test ValueError for odd degree"
    with pytest.raises(
        ValueError,
        match=".*degree should be a factor of 2*"
    ):
        ap.zonal_spherical_function(cycle_type, partition)

    
@pytest.mark.parametrize(
    "cycle_type, partition",
    [
        (Permutation(3)(0,1), (1,)),
        (Permutation(5)(0,1,2), (1,1)),
        (Permutation(0,1,2,3,4,5), (4,)),
        ((1,1,1,1), (3,)),
        ((2,2,1,1), (2,)),
        ((4,2), (4,)),
        ((3,3,2), (2,2,2)),
    ]
)
def test_zonal_spherical_partition_error(cycle_type, partition):
    "Test ValueError for invalid cycle-type and partition"
    with pytest.raises(
        ValueError,
        match=".*Invalid partition and cyle-type*"
    ):
        ap.zonal_spherical_function(cycle_type, partition)