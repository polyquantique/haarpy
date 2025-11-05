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
Unitary group Python interface
"""

from math import factorial, prod
from functools import lru_cache
from typing import Union
from itertools import product
from collections import Counter
from fractions import Fraction
from sympy import Symbol, Expr, fraction, simplify, factor
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from haarpy import (
    get_conjugacy_class,
    murn_naka_rule,
    irrep_dimension,
    stabilizer_coset,
)


@lru_cache
def representation_dimension(partition: tuple[int], unitary_dimension: Symbol) -> Expr:
    """Returns the dimension of the unitary group U(d) labelled by the input partition

    Args:
        partition (tuple[int]) : A partition labelling a representation of U(d)
        unitary_dimension (Symbol) : dimension d of the unitary matrix U

    Returns:
        Expr : The dimension of the representation of U(d) labeled by the partition
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
def weingarten_unitary(
    cycle: Union[Permutation, tuple[int]], unitary_dimension: Symbol
) -> Expr:
    """Returns the Weingarten function

    Args:
        cycle (Permutation, tuple[int]): Permutation from the symmetric group or its cycle-type
        unitary_dimension (Symbol): Dimension d of the unitary matrix U

    Returns:
        Expr: The Weingarten function

    Raise:
        TypeError: if unitary_dimension has the wrong type
        TypeError: if cycle has the wrong type
    """
    if not isinstance(unitary_dimension, (Expr, int)):
        raise TypeError("unitary_dimension must be an instance of int or sympy.Expr")

    if isinstance(cycle, Permutation):
        degree = cycle.size
        conjugacy_class = get_conjugacy_class(cycle, degree)
    elif isinstance(cycle, (tuple, list)) and all(
        isinstance(value, int) for value in cycle
    ):
        degree = sum(cycle)
        conjugacy_class = tuple(cycle)
    else:
        raise TypeError

    partition_tuple = tuple(
        sum((value * (key,) for key, value in part.items()), ())
        for part in partitions(degree)
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
        weingarten = (
            sum(
                irrep_dimension**2
                * murn_naka_rule(partition, conjugacy_class)
                / representation_dimension(partition, unitary_dimension)
                for partition, irrep_dimension in zip(
                    partition_tuple, irrep_dimension_tuple
                )
            )
            / factorial(degree) ** 2
        )
        numerator, denominator = fraction(simplify(weingarten))
        weingarten = factor(numerator) / factor(denominator)

    return weingarten


@lru_cache
def haar_integral_unitary(
    sequences: tuple[tuple[int]], unitary_dimension: Symbol
) -> Expr:
    """Returns integral over unitary group polynomial sampled at random from the Haar measure

    Args:
        sequences (tuple[tuple[int]]): Indices of matrix elements
        unitary_dimension (Symbol): Dimension of the unitary group

    Returns:
        Expr: Integral under the Haar measure

    Raise:
        ValueError: If sequences doesn't contain 4 tuples
        ValueError: If tuples i and j are of different length
    """
    if len(sequences) != 4:
        raise ValueError("Wrong tuple format")

    seq_i, seq_j, seq_i_prime, seq_j_prime = sequences

    if len(seq_i) != len(seq_j) or len(seq_i_prime) != len(seq_j_prime):
        raise ValueError("Wrong tuple format")

    degree = len(seq_i)

    class_mapping = Counter(
        get_conjugacy_class(cycle_i * ~cycle_j, degree)
        for cycle_i, cycle_j in product(
            stabilizer_coset(seq_i, seq_i_prime),
            stabilizer_coset(seq_j, seq_j_prime),
        )
    )

    integral = sum(
        count * weingarten_unitary(conjugacy, unitary_dimension)
        for conjugacy, count in class_mapping.items()
    )

    if isinstance(unitary_dimension, Expr):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral
