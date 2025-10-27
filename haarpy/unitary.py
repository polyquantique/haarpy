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
from sympy import Symbol, fraction, simplify, factor
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions
from haarpy import get_conjugacy_class, murn_naka_rule, irrep_dimension


@lru_cache
def representation_dimension(partition: tuple[int], unitary_dimension: Symbol) -> int:
    """Returns the dimension of the unitary group U(d) labelled by the input partition

    Args:
        partition (tuple[int]) : A partition labelling a representation of U(d)
        unitary_dimension (Symbol) : dimension d of the unitary matrix U

    Returns:
        Symbol : The dimension of the representation of U(d) labeled by the partition
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
def weingarten_unitary(cycle: Union[Permutation, tuple[int]], unitary_dimension: Symbol) -> Symbol:
    """Returns the Weingarten function

    Args:
        cycle (Permutation, tuple(int)) : Permutation cycle from the symmetric group or its cycle-type
        unitary_dimension (Symbol) : Dimension d of the unitary matrix U

    Returns:
        Symbol : The Weingarten function
    """
    if not isinstance(unitary_dimension, (Symbol, int)):
        raise TypeError("unitary_dimension must be an instance of int or sympy.Symbol")
    
    if isinstance(cycle, Permutation):
        degree = cycle.size
        conjugacy_class = get_conjugacy_class(cycle, degree)
    elif isinstance(cycle, (tuple,list)) and all(isinstance(value, int) for value in cycle):
        degree = sum(cycle)
        conjugacy_class = tuple(cycle)
    else:
        raise TypeError

    partition_tuple = tuple(
        sum((value * (key,) for key, value in part.items()), ()) for part in partitions(degree)
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
                for partition, irrep_dimension in zip(partition_tuple, irrep_dimension_tuple)
            )
            / factorial(degree) ** 2
        )
        numerator, denominator = fraction(simplify(weingarten))
        weingarten = factor(numerator) / factor(denominator)

    return weingarten


@lru_cache
def haar_integral_unitary(sequences: tuple[tuple[int]], group_dimension: int) -> Symbol:
    """Returns integral over unitary group polynomial sampled at random from the Haar measure

    Args:
        sequences (tuple(tuple(int))) : Indices of matrix elements
        group_dimension (int) : Dimension of the compact group

    Returns:
        Symbol : Integral under the Haar measure

    Raise:
        ValueError : If sequences doesn't contain 4 tuples
        ValueError : If tuples i and j are of different length
    """
    if len(sequences) != 4:
        raise ValueError("Wrong tuple format")

    str_i, str_j, str_i_prime, str_j_prime = sequences

    if len(str_i) != len(str_j) or len(str_i_prime) != len(str_j_prime):
        raise ValueError("Wrong tuple format")

    if sorted(str_i) != sorted(str_i_prime) or sorted(str_j) != sorted(str_j_prime):
        return 0

    degree = len(str_i)
    str_i, str_j = list(str_i), list(str_j)

    permutation_i = (
        perm
        for perm in SymmetricGroup(degree).generate_schreier_sims()
        if perm(str_i_prime) == str_i
    )

    permutation_j = (
        perm
        for perm in SymmetricGroup(degree).generate_schreier_sims()
        if perm(str_j_prime) == str_j
    )

    class_mapping = dict(
        Counter(
            get_conjugacy_class(cycle_i * ~cycle_j, degree)
            for cycle_i, cycle_j in product(permutation_i, permutation_j)
        )
    )
    integral = sum(
        count * weingarten_unitary(conjugacy, group_dimension)
        for conjugacy, count in class_mapping.items()
    )

    if isinstance(group_dimension, Symbol):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral
