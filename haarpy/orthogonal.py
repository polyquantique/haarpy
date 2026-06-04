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
Orthogonal group Python interface

References
----------
    [1] Collins, B., & Śniady, P. (2006). Integration with respect to the Haar measure on unitary,
    orthogonal and symplectic group. Communications in Mathematical Physics, 264(3), 773-795.

    [2] Matsumoto, S. (2013). Weingarten calculus for matrix ensembles associated with compact
    symmetric spaces. arXiv preprint arXiv:1301.5401.

    [3] Macdonald, I. G. (1998). Symmetric functions and Hall polynomials. Oxford university press.
"""

from math import prod
from fractions import Fraction
from itertools import product
from functools import lru_cache
from collections import Counter
from sympy import Symbol, Expr, factorial
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from haarpy import (
    murn_naka_rule,
    get_conjugacy_class,
    irrep_dimension,
    HyperoctahedralGroup,
    hyperoctahedral_transversal,
    coset_type,
    coset_type_representative,
)
from ._utils import _simplify


@lru_cache
def zonal_spherical_function(permutation: Permutation, partition: tuple[int, ...]) -> Fraction:
    """Returns the zonal spherical function of the Gelfand pair :math:`(S_{2p}, H_p)`

    Parameters
    ----------
    permutation : Permutation
        A permutation of the symmetric group :math:`S_{2p}`

    partition : tuple[int, ...]
        An integer partition of :math:`p`

    Returns
    -------
    Fraction
        The zonal spherical function of the given permutation

    Raises
    ------
    TypeError
        If ``partition`` is not a tuple
    TypeError
        If ``permutation`` is not a permutation

    Examples
    --------
    >>> from sympy.combinatorics import Permutation
    >>> from haarpy import zonal_spherical_function
    >>> zonal_spherical_function(Permutation(5)(0,1,2), (2,1))
    Fraction(1, 6)

    See Also
    --------
    :func:`haarpy.symmetric.HyperoctahedralGroup`
        Returns the hyperoctahedral group :math:`H_p`
    :func:`haarpy.symmetric.murn_naka_rule`
        Implementation of the Murnaghan-Nakayama rule for the characters irreducible
        representations of the symmetric group
    """
    if not isinstance(partition, tuple):
        raise TypeError

    if not isinstance(permutation, Permutation):
        raise TypeError

    degree = permutation.size

    if degree % 2:
        raise ValueError("degree should be a factor of 2")
    if degree != 2 * sum(partition):
        raise ValueError("Invalid partition and cyle-type")

    double_partition = tuple(2 * part for part in partition)
    hyperocta = HyperoctahedralGroup(degree // 2)
    numerator = sum(
        murn_naka_rule(double_partition, get_conjugacy_class(permutation * zeta))
        for zeta in hyperocta.generate()
    )
    return Fraction(numerator, hyperocta.order())


@lru_cache
def weingarten_orthogonal(
    permutation: Permutation | tuple[int, ...], orthogonal_dimension: Symbol
) -> Expr:
    """Returns the orthogonal Weingarten function

    Parameters
    ----------
    permutation : Permutation | tuple[int, ...]
        A permutation of the symmetric group or its coset-type

    orthogonal_dimension : Symbol
        The dimension of the orthogonal group

    Returns
    -------
    Expr
        The Weingarten function

    Raises
    ------
    TypeError
        If unitary_dimension has the wrong type
    TypeError
        If ``permutation`` has the wrong type
    ValueError
        If the degree :math:`2p` of the symmetric group :math:`S_{2p}` is not even

    Notes
    -----
    Since the orthogonal Weingarten function is invariant over coset-types, the argument
    may be given either as a permutation or as its coset-type

    Examples
    --------
    >>> from sympy import Symbol
    >>> from sympy.combinatorics import Permutation
    >>> from haarpy import weingarten_orthogonal
    >>> d = Symbol("d")
    >>> weingarten_orthogonal(Permutation(5)(0,1,2), 6)
    Fraction(-1, 1200)
    >>> weingarten_orthogonal(Permutation(5)(0,1,2), d)
    -1/(d*(d - 2)*(d - 1)*(d + 4))
    >>> weingarten_orthogonal((2,1), d)
    -1/(d*(d - 2)*(d - 1)*(d + 4))

    See Also
    --------
    :func:`haarpy.orthogonal.zonal_spherical_function`
        Returns the zonal spherical function of the Gelfand pair :math:`(S_{2p}, H_p)`
    :func:`haarpy.symmetric.coset_type`
        Returns the coset-type of a given permutation of the symmetric group
    :func:`haarpy.symmetric.coset_type_representative`
        Returns a representative permutation of the symmetric group :math:`S_{2p}`
        for a given input coset-type
    """
    if not isinstance(orthogonal_dimension, (Expr, int)):
        raise TypeError("orthogonal_dimension must be an instance of int or sympy.Symbol")

    if isinstance(permutation, (tuple, list)) and all(
        isinstance(value, int) for value in permutation
    ):
        permutation = coset_type_representative(permutation)
    elif not isinstance(permutation, Permutation):
        raise TypeError

    degree = permutation.size
    if degree % 2:
        raise ValueError("The degree of the symmetric group S_2k should be even")

    half_degree = degree // 2

    partition_tuple = tuple(
        sum((value * (key,) for key, value in part.items()), ()) for part in partitions(half_degree)
    )
    double_partition_tuple = tuple(
        tuple(2 * part for part in partition) for partition in partition_tuple
    )
    irrep_dimension_gen = (irrep_dimension(partition) for partition in double_partition_tuple)
    zonal_spherical_gen = (
        zonal_spherical_function(permutation, partition) for partition in partition_tuple
    )
    coefficient_gen = (
        prod(
            orthogonal_dimension + 2 * j - i
            for i in range(len(partition))
            for j in range(partition[i])
        )
        for partition in partition_tuple
    )

    if isinstance(orthogonal_dimension, int):
        weingarten = sum(
            Fraction(
                irrep_dim * zonal_spherical,
                coefficient,
            )
            for irrep_dim, zonal_spherical, coefficient in zip(
                irrep_dimension_gen, zonal_spherical_gen, coefficient_gen
            )
            if coefficient
        ) * Fraction(
            2**half_degree * factorial(half_degree),
            factorial(degree),
        )
    else:
        weingarten_gen = (
            irrep_dim * zonal_spherical / coefficient
            for irrep_dim, zonal_spherical, coefficient in zip(
                irrep_dimension_gen, zonal_spherical_gen, coefficient_gen
            )
            if coefficient
        )
        weingarten = _simplify(
            weingarten_gen, Fraction(2**half_degree * factorial(half_degree), factorial(degree))
        )

    return weingarten


@lru_cache
def haar_integral_orthogonal(
    sequences: tuple[tuple[int, ...], ...], orthogonal_dimension: Symbol
) -> Expr:
    """Returns the integral over orthogonal group polynomial sampled at random from the Haar measure

    Parameters
    ----------
    sequences : tuple[tuple[int, ...], ...]
        Sequences of matrix elements
    orthogonal_dimension : Symbol
        The dimension of the orthogonal group

    Returns
    -------
    Expr
        The integral under the Haar measure

    Raises
    ------
    ValueError
        If the argument ``sequences`` does not contain 2 tuples
    ValueError
        If the sequences are of different lengths

    Examples
    --------
    >>> from sympy import Symbol
    >>> from haarpy import haar_integral_orthogonal
    >>> d = Symbol("d")
    >>> row_indices, column_indices = (0, 0, 1, 1, 2, 2), (0, 2, 2, 1, 1, 0)
    >>> haar_integral_orthogonal((row_indices, column_indices), 4)
    Fraction(1, 576)
    >>> haar_integral_orthogonal((row_indices, column_indices), d)
    2/(d*(d - 2)*(d - 1)*(d + 2)*(d + 4))

    See Also
    --------
    :func:`haarpy.symmetric.coset_type`
        Returns the coset-type of a given permutation of the symmetric group
    :func:`haarpy.symmetric.hyperoctahedral_transversal`
        Yields the permutations of :math:`M_{2p}`, the complete set of coset
        representatives of the quotient group :math:`S_{2p}/H_p`
    :func:`haarpy.orthogonal.weingarten_orthogonal`
        Computes the orthogonal Weingarten function
    """
    if len(sequences) != 2:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j = sequences
    degree = len(seq_i)

    if degree != len(seq_j):
        raise ValueError("Wrong tuple format")

    if degree % 2:
        return 0

    permutation_i = (
        perm
        for perm in hyperoctahedral_transversal(degree)
        if perm(seq_i)[::2] == perm(seq_i)[1::2]
    )

    permutation_j = (
        perm
        for perm in hyperoctahedral_transversal(degree)
        if perm(seq_j)[::2] == perm(seq_j)[1::2]
    )

    coset_mapping = Counter(
        coset_type(cycle_j * ~cycle_i) for cycle_i, cycle_j in product(permutation_i, permutation_j)
    )

    integral_gen = (
        count * weingarten_orthogonal(coset, orthogonal_dimension)
        for coset, count in coset_mapping.items()
    )

    return sum(integral_gen) if isinstance(orthogonal_dimension, int) else _simplify(integral_gen)
