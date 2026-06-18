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
Unitary group Python interface

References
----------
    [1] Collins, B. (2003). Moments and cumulants of polynomial random variables on unitarygroups,
    the Itzykson-Zuber integral, and free probability. International Mathematics Research Notices,
    2003(17), 953-982.

    [2] Matsumoto, S. (2013). Weingarten calculus for matrix ensembles associated with compact
    symmetric spaces. arXiv preprint arXiv:1301.5401.
"""

from math import factorial, prod
from functools import lru_cache
from itertools import product
from collections import Counter
from fractions import Fraction
from sympy import Symbol, Expr
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from haarpy import (
    get_conjugacy_class,
    murn_naka_rule,
    irrep_dimension,
    stabilizer_coset,
)
from ._utils import _simplify


@lru_cache
def representation_dimension(partition: tuple[int, ...], unitary_dimension: Symbol) -> Expr:
    """Returns the dimension of the unitary group's representation labelled by the input partition

    Parameters
    ----------
    partition : tuple[int, ...]
        A partition labelling a representation of the unitary group :math:`U(d)`

    unitary_dimension : Symbol
        The dimension :math:`d` of the unitary group

    Returns
    -------
    Expr
        The dimension of the unitary group's representation labelled by the input partition

    Examples
    --------
    >>> from sympy import Symbol
    >>> from haarpy import representation_dimension
    >>> d = Symbol("d")
    >>> representation_dimension((2,1,1), 4)
    15
    >>> representation_dimension((2,1,1), d)
    d*(d/2 - 1/2)*(d - 2)*(d + 1)/4
    """
    conjugate_partition = tuple(
        sum(1 for part in partition if i < part) for i in range(partition[0])
    )
    if isinstance(unitary_dimension, int):
        dimension = prod(
            Fraction(
                unitary_dimension + j - i,
                part + conjugate_partition[j] - i - j - 1,
            )
            for i, part in enumerate(partition)
            for j in range(part)
        )

        return dimension.numerator

    dimension = prod(
        (unitary_dimension + j - i) / (part + conjugate_partition[j] - i - j - 1)
        for i, part in enumerate(partition)
        for j in range(part)
    )

    return dimension


@lru_cache
def weingarten_unitary(cycle: Permutation | tuple[int, ...], unitary_dimension: Symbol) -> Expr:
    """Returns the unitary Weingarten function

    Parameters
    ----------
    cycle : Permutation | tuple[int, ...]
        A permutation from the symmetric group or a partition reprensenting its cycle-type

    unitary_dimension : Symbol
        The dimension :math:`d` of the unitary matrix :math:`U(d)`

    Returns
    -------
    Expr
        The Weingarten function

    Raises
    ------
    TypeError
        If unitary_dimension has the wrong type
    TypeError
        If cycle has the wrong type

    Notes
    -----
    Since the unitary Weingarten function is a class function on the symmetric group, the argument
    may be given either as a permutation or as its cycle-type

    Examples
    --------
    >>> from sympy import Symbol
    >>> from haarpy import weingarten_unitary
    >>> d = Symbol("d")
    >>> weingarten_unitary(Permutation(2)(0, 1), 4)
    Fraction(-1, 180)
    >>> weingarten_unitary(Permutation(2)(0, 1), d)
    -1/((d - 2)*(d - 1)*(d + 1)*(d + 2))
    >>> weingarten_unitary((2, 1), d)
    -1/((d - 2)*(d - 1)*(d + 1)*(d + 2))

    See Also
    --------
    :func:`haarpy.symmetric.murn_naka_rule`
        Implementation of the Murnaghan-Nakayama rule for the characters irreducible
        representations of the symmetric group :math:`S_p`
    :func:`haarpy.unitary.representation_dimension`
        Computes the dimension of the unitary group's representation labelled by a
        given partition
    """
    if not isinstance(unitary_dimension, (Expr, int)):
        raise TypeError("unitary_dimension must be an instance of int or sympy.Expr")

    if isinstance(cycle, Permutation):
        degree = cycle.size
        conjugacy_class = get_conjugacy_class(cycle)
    elif isinstance(cycle, (tuple, list)) and all(isinstance(value, int) for value in cycle):
        degree = sum(cycle)
        conjugacy_class = tuple(cycle)
    else:
        raise TypeError

    partition_tuple = tuple(
        tuple(summand for summand, mult in partition.items() for _ in range(mult))
        for partition in partitions(degree)
    )
    irrep_dimension_tuple = (irrep_dimension(part) for part in partition_tuple)

    if isinstance(unitary_dimension, int):
        weingarten = sum(
            Fraction(
                irrep_dimension**2 * murn_naka_rule(part, conjugacy_class),
                representation_dimension(part, unitary_dimension),
            )
            for part, irrep_dimension in zip(partition_tuple, irrep_dimension_tuple)
        ) * Fraction(1, factorial(degree) ** 2)
    else:
        weingarten_gen = (
            irrep_dimension**2
            * murn_naka_rule(partition, conjugacy_class)
            / representation_dimension(partition, unitary_dimension)
            for partition, irrep_dimension in zip(partition_tuple, irrep_dimension_tuple)
        )

        weingarten = _simplify(weingarten_gen, Fraction(1, factorial(degree) ** 2))

    return weingarten


@lru_cache
def haar_integral_unitary(
    sequences: tuple[tuple[int, ...], ...], unitary_dimension: Symbol
) -> Expr:
    """Returns the integral of a monomial over the unitary group

    Parameters
    ----------
    sequences : tuple[tuple[int, ...], ...]
        Sequences of matrix elements

    unitary_dimension : Symbol
        The dimension of the unitary group

    Returns
    -------
    Expr
        The integral under the Haar measure

    Raises
    ------
    ValueError
        if sequences do not contain 4 tuples
    ValueError
        if the input sequences are of different lengths

    Examples
    --------
    >>> from sympy import Symbol
    >>> from haarpy import haar_integral_unitary
    >>> d = Symbol("d")
    >>> seq_i, seq_j = (0, 1, 2), (0, 0, 1)
    >>> seq_i_prime, seq_j_prime = (0, 1, 2), (0, 1, 0)
    >>> haar_integral_unitary((seq_i, seq_j, seq_i_prime, seq_j_prime), 5)
    Fraction(-1, 840)
    >>> haar_integral_unitary((seq_i, seq_j, seq_i_prime, seq_j_prime), d)
    -1/(d*(d - 1)*(d + 1)*(d + 2))

    See Also
    --------
    :func:`haarpy.symmetric.stabilizer_coset`
        Returns all permutations sending a first sequence to a second sequence
    :func:`haarpy.unitary.weingarten_unitary`
        Returns the unitary Weingarten function
    """
    if len(sequences) != 4:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j, seq_i_prime, seq_j_prime = sequences

    if len(seq_i) != len(seq_j) or len(seq_i_prime) != len(seq_j_prime):
        raise ValueError("Wrong tuple format")

    #integral over the Haar measure
    if len(seq_i) == 0:
        return 1
    
    class_mapping = Counter(
        get_conjugacy_class(cycle_i * ~cycle_j)
        for cycle_i, cycle_j in product(
            stabilizer_coset(seq_i, seq_i_prime),
            stabilizer_coset(seq_j, seq_j_prime),
        )
    )

    integral_gen = (
        count * weingarten_unitary(conjugacy, unitary_dimension)
        for conjugacy, count in class_mapping.items()
    )

    return sum(integral_gen) if isinstance(unitary_dimension, int) else _simplify(integral_gen)
