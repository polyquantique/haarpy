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
r"""
Haarpy
==========

.. currentmodule:: Haarpy

This is the top level module of Haarpy, containing functions for
the symbolic calculation of Weingarten functions and related averages
of unitary matrices sampled uniformly at random from the Haar measure.

Functions
---------

.. autosummary::
    weingarten_unitary
    weingarten_orthogonal
    weingarten_symplectic
    weingarten_circular_orthogonal
    weingarten_circular_symplectic
    weingarten_permutation
    weingarten_centered_permutation
    haar_integral_unitary
    haar_integral_orthogonal
    haar_integral_circular_orthogonal
    haar_integral_permutation
    haar_integral_centered_permutation
    get_conjugacy_class
    derivative_tableaux
    semi_standard_young_tableaux
    proper_border_strip
    murn_naka_rule
    irrep_dimension
    representation_dimension
    sorting_permutation
    YoungSubgroup
    stabilizer_coset
    HyperoctahedralGroup
    hyperoctahedral_transversal
    zonal_spherical_function
    coset_type
    coset_type_representative
    twisted_spherical_function
    mobius_function
    set_partitions
    perfect_matchings
    partial_order
    meet_operation
    join_operation
    is_crossing_partition

Code details
------------
"""
from .partition import (
    set_partitions,
    perfect_matchings,
    partial_order,
    meet_operation,
    join_operation,
    is_crossing_partition,
)

from .symmetric import (
    get_conjugacy_class,
    derivative_tableaux,
    semi_standard_young_tableaux,
    proper_border_strip,
    murn_naka_rule,
    irrep_dimension,
    sorting_permutation,
    YoungSubgroup,
    stabilizer_coset,
    HyperoctahedralGroup,
    hyperoctahedral_transversal,
    coset_type,
    coset_type_representative,
)

from .unitary import (
    representation_dimension,
    weingarten_unitary,
    haar_integral_unitary,
)

from .orthogonal import (
    zonal_spherical_function,
    weingarten_orthogonal,
    haar_integral_orthogonal,
)

from .symplectic import (
    twisted_spherical_function,
    weingarten_symplectic,
)

from .circular_ensembles import (
    weingarten_circular_orthogonal,
    weingarten_circular_symplectic,
    haar_integral_circular_orthogonal,
)

from .permutation import (
    mobius_function,
    weingarten_permutation,
    weingarten_centered_permutation,
    haar_integral_permutation,
    haar_integral_centered_permutation,
)

from ._version import __version__

__all__ = [
    "weingarten_unitary",
    "weingarten_orthogonal",
    "weingarten_symplectic",
    "weingarten_circular_orthogonal",
    "weingarten_circular_symplectic",
    "weingarten_permutation",
    "weingarten_centered_permutation",
    "haar_integral_unitary",
    "haar_integral_orthogonal",
    "haar_integral_circular_orthogonal",
    "haar_integral_permutation",
    "haar_integral_centered_permutation",
    "get_conjugacy_class",
    "derivative_tableaux",
    "semi_standard_young_tableaux",
    "proper_border_strip",
    "murn_naka_rule",
    "irrep_dimension",
    "representation_dimension",
    "sorting_permutation",
    "YoungSubgroup",
    "stabilizer_coset",
    "HyperoctahedralGroup",
    "hyperoctahedral_transversal",
    "zonal_spherical_function",
    "coset_type",
    "coset_type_representative",
    "twisted_spherical_function",
    "mobius_function",
    "set_partitions",
    "perfect_matchings",
    "partial_order",
    "meet_operation",
    "join_operation",
    "is_crossing_partition",
]


def version():
    r"""
    Get version number of haarpy

    Returns:
      str: The package version number
    """
    return __version__


def about():
    """Haarpy information.

    Prints the installed version numbers for Haarpy and its dependencies,
    and some system info. Please include this information in bug reports.

    **Example:**

    .. code-block:: pycon

        >>> haarpy.about()
        Haarpy: a Python library for the symbolic calculation of Weingarten functions.

        Python version:            3.12.3
        Platform info:             Linux-6.8.0-31-generic-x86_64-with-glibc2.39
        Installation path:         /home/username/haarpy
        Haarpy version:            0.0.6
        SymPy version:             1.12
    """
    # pylint: disable=import-outside-toplevel
    import os
    import platform

    import sys
    import sympy

    # a QuTiP-style infobox
    print(
        "\nHaarpy: a Python library for the symbolic calculation of Weingarten functions."
    )
    # print("Copyright 2018-2021 Polyquantique\n")

    print("Python version:            {}.{}.{}".format(*sys.version_info[0:3]))
    print(f"Platform info:             {platform.platform()}")
    print(f"Installation path:         {os.path.dirname(__file__)}")
    print(f"Haarpy version:            {__version__}")
    print(f"SymPy version:             {sympy.__version__}")
