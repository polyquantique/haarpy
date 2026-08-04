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
Utils tests
"""

import pytest
import haarpy._utils as ut


@pytest.mark.parametrize(
    "row_index_tuple, col_index_tuple",
    [
        (((1,1),), ((1,1), (2,2))),
        (((1,1),(1,1,1)), ((1,1), (2,2))),
    ]
)
def test_sequence_to_matrix_value_error(row_index_tuple, col_index_tuple):
    "Test mismatch sequence formats"
    with pytest.raises(ValueError):
        ut._sequence_to_matrix(row_index_tuple, col_index_tuple)


@pytest.mark.parametrize(
    "row_sums, power_matrix",
    [
        ((2,2), ((1,1,1), (1,1,0))),
        ((3,3,3), ((1,1,1), (1,1,2), (1,1,1))),
    ]
)
def test_vanishing_vector_multinomial(row_sums, power_matrix):
    "Test mismatch in multinomial coefficients"
    assert not ut._vector_multinomial(row_sums, power_matrix)
