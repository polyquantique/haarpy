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
    get_conjugacy_class
    derivative_tableaux
    semi_standard_young_tableaux
    proper_border_strip
    murn_naka_rule
    irrep_dimension
    representation_dimension
    weingarten_class
    weingarten_element
    haar_integral

Code details
------------
"""
from .unitary import (
    get_conjugacy_class,
    derivative_tableaux,
    semi_standard_young_tableaux,
    proper_border_strip,
    murn_naka_rule,
    irrep_dimension,
    representation_dimension,
    weingarten_class,
    weingarten_element,
    haar_integral,
)

from ._version import __version__

__all__ = [
    "get_conjugacy_class",
    "derivative_tableaux",
    "semi_standard_young_tableaux",
    "proper_border_strip",
    "murn_naka_rule",
    "irrep_dimension",
    "representation_dimension",
    "weingarten_class",
    "weingarten_element",
    "haar_integral",
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
        Haarpy version:            0.0.5
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
