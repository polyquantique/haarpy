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
haarpy Python interface
"""

from math import factorial
from copy import deepcopy
from itertools import product
from collections import Counter
from fractions import Fraction
import numpy as np
from sympy import Symbol, fraction, simplify, factor
from sympy.combinatorics import Permutation, SymmetricGroup
from sympy.utilities.iterables import partitions


def get_class(cycle: Permutation, degree: int) -> tuple:
    """Returns the conjugacy class of an element of the symmetric group Sp

    Args:
        degree (integer): Order of the symmetric group
        cycle (cycle): Permutation cycle from the symmetric group

    Returns:
        tuple[int]: the conjugacy class in partition form

    Raise:
        TypeError : order must be of type int
        ValueError : If order is not an integer greaten than 1
        TypeError : cycle must be of type sympy.combinatorics.permutations.Permutation
        ValueError : Incompatible degree and permutation cycle
    """
    if not isinstance(degree, int):
        raise TypeError("degree must be of type int")
    if degree < 1:
        raise ValueError(
            "The degree you have provided is too low. It must be an integer greater than 0."
        )
    if not isinstance(cycle, Permutation):
        raise TypeError(
            "cycle must be of type sympy.combinatorics.permutations.Permutation"
        )

    if cycle.support() == []:
        return degree * (1,)

    if max(cycle.support()) >= degree:
        raise ValueError("Incompatible degree and permutation cycle")
    return tuple(sorted([len(i) for i in cycle.cyclic_form], reverse=True)) + (1,) * (
        degree - len(cycle.support())
    )


def derivative_tableau(
    tableau: list[list[int]], add_unit: int, partition: tuple[int]
) -> list[list[list[int]]]:
    """Takes a single tableau and adds the selected number to it's contents
    in a way that keeps it semi-standard

    Args:
        tableau (list[list[int]]): The tableau
        add_unit (int): Selected number to be added
        partition ([list[int]) : partition characterizing an irrep of Sp

    Returns:
        list[list[list[int]]]: Modified tableau
    """
    tableau_derivatives = []
    for i, _ in enumerate(partition):
        for j in range(partition[i]):
            if tableau[i][j] == 0:
                cond = i == 0 and j == 0 and add_unit == 1
                cond |= (
                    j == 0 and tableau[i - 1][0] <= add_unit and tableau[i - 1][0] != 0
                )
                cond |= (
                    i == 0 and tableau[0][j - 1] <= add_unit and tableau[0][j - 1] != 0
                )
                cond |= (
                    i != 0
                    and j != 0
                    and tableau[i - 1][j] <= add_unit
                    and tableau[i][j - 1] <= add_unit
                    and tableau[i - 1][j] != 0
                    and tableau[i][j - 1] != 0
                )
                if cond:
                    copy = deepcopy(tableau)
                    copy[i][j] = add_unit
                    tableau_derivatives.append(copy)
    return tableau_derivatives


def ssyt(partition: tuple[int], conjugacy_class: tuple[int]) -> list[list[int]]:
    """all eligible semi-standard young tableaux based of the partition

    Args:
        partition ([list[int]) : partition characterizing an irrep of Sp
        conjugacy_class (list[int]) : A conjugacy class, in partition form, of Sp

    Returns:
        list[list[list[int]]]: all eligible semi-standard young tableaux
    """
    tableau = [[partition[0] * [0] for _ in range(len(partition))]]
    add_unit = [i + 1 for i, m in enumerate(conjugacy_class) for j in range(m)]

    for unit in add_unit:
        tableau = [derivative_tableau(see, unit, partition) for see in tableau]
        tableau = [i for j in tableau for i in j]  # flattens the list
        tableau = {tuple(tuple(i) for i in j) for j in tableau}  # removes duplicates
        tableau = [[list(i) for i in j] for j in tableau]

    return tableau


def bad_mapping(tableau: list[list[int]], conjugacy_class: tuple[int]) -> bool:
    """Flags tableaux that have a bad mapping

    Args:
        tableau (list[list[int]]) : A semi-standard Young tableau
        conjugacy_class (list[int]) : A conjugacy class, in partition form, of Sp

    Returns:
        bool : True if mapping is wrong
    """
    for i in range(len(conjugacy_class)):
        # indices of matching elements
        matching = [
            (j + 1, k + 1)
            for j, tab in enumerate(tableau)
            for k, val in enumerate(tab)
            if val == i + 1
        ]
        counting = len(matching)
        if counting == 1:
            matching.append((matching[0][0] + 1, matching[0][1]))

        shape = (counting, counting, 2, 2)
        matching = np.array(list(product(matching, repeat=2)))
        matching = matching.flatten()[: np.prod(shape)]
        matching = matching.reshape(shape).tolist()

        # Euclidean distance
        for j in range(counting):
            for k in range(counting):
                matching[j][k] = np.linalg.norm(
                    np.array(matching[j][k][1]) - np.array(matching[j][k][0])
                )

        d = [matching[j].count(1) for j in range(counting)]

        for val in d:
            if counting >= 2 and val == 0:
                return True

        if d.count(1) > 2 or (d.count(1) == 0 and counting != 1):
            return True

    return False


def murn_naka(partition: tuple[int], conjugacy_class: tuple[int]) -> int:
    """Implementation of the Murnaghan-Nakayama rule for the characters irreducible
    representations of the symmetric group Sp

    Args:
        partition (tuple[int]) : partition characterizing an irrep of Sp
        conjugacy_class (tuple[int]) : A conjugacy class, in partition form, of Sp

    Returns:
        int : Character of the elements in class mu of the irrep of the symmetric group
    """
    if sum(partition) != sum(conjugacy_class):
        return 0

    tableaux_list = ssyt(partition, conjugacy_class)
    tableaux_list = [
        tableau
        for tableau in tableaux_list
        if not bad_mapping(tableau, conjugacy_class)
    ]

    skip = False
    deleted = 0
    tableaux_list = np.array(tableaux_list)
    for i, tableau in enumerate(tableaux_list):
        for j in range(len(partition) - 1):
            if skip:
                skip = False
                break
            for k in range(partition[0] - 1):
                if (
                    tableau[j, k]
                    and tableau[j, k]
                    == tableau[j, k + 1]
                    == tableau[j + 1, k]
                    == tableau[j + 1, k + 1]
                ):
                    tableaux_list = np.delete(tableaux_list, i - deleted, 0)
                    deleted += 1
                    skip = True
                    break

    height = np.empty(shape=(len(tableaux_list), len(conjugacy_class)))
    height.fill(-1)
    weight = np.empty(shape=(len(tableaux_list),))
    for i, tableau in enumerate(tableaux_list):
        for j in range(len(conjugacy_class)):
            for k in range(len(partition)):
                height[i, j] += 1 if np.count_nonzero(tableau[k] == j + 1) else 0
        weight[i] = (-1) ** np.sum(height[i, : len(conjugacy_class)])

    return int(np.sum(weight[: len(tableaux_list)]))


def sn_dimension(partition: tuple[int]) -> int:
    """Returns the dimension of the irrep of the symmetric group Sp labelled by the input partition

    Args:
        partition (list[int]) : A partition labelling an irrep of Sp

    Returns:
        int : The dimension of the irrep
    """
    numerator = np.prod(
        [
            part_i - part_j + j + 1
            for i, part_i in enumerate(partition)
            for j, part_j in enumerate(partition[i + 1 :])
        ],
        dtype=np.int64,
    )
    len_partition = len(partition)
    denominator = np.prod(
        [factorial(part + len_partition - i - 1) for i, part in enumerate(partition)],
        dtype=np.int64,
    )
    dimension = Fraction(numerator, denominator) * factorial(sum(partition))
    if dimension.denominator != 1:
        raise ValueError("Fraction too large for dtype int32")

    return dimension.numerator


def ud_dimension(partition: tuple[int], unitary_dimension: Symbol) -> int:
    """Returns the dimension of the unitary group U(d) labelled by the input partition

    Args:
        partition (list[int]) : A partition labelling U(d)
        unitary_dimension (Symbol) : dimension d of the unitary matrix U

    Returns:
        Symbol : The dimension of U(d)
    """
    conjugate_partition = [
        sum(1 for _, part in enumerate(partition) if i < part)
        for i in range(partition[0])
    ]
    if isinstance(unitary_dimension, int):
        dimension = np.prod(
            [
                np.prod(
                    [
                        Fraction(
                            unitary_dimension + j - i,
                            part + conjugate_partition[j] - i - j - 1,
                        )
                        for j in range(part)
                    ]
                )
                for i, part in enumerate(partition)
            ]
        )
        if dimension.denominator != 1:
            raise ValueError("Fraction too large for dtype int32")
        return dimension.numerator

    numerator = np.prod(
        [
            np.prod([(unitary_dimension + k - i) for k in range(part)])
            for i, part in enumerate(partition)
        ]
    )

    denominator = np.prod(
        [
            np.prod([part + conjugate_partition[j] - i - j - 1 for j in range(part)])
            for i, part in enumerate(partition)
        ]
    )

    return numerator / denominator


def weingarten_class(conjugacy_class: tuple[int], unitary_dimension: Symbol) -> Symbol:
    """Returns the Weingarten function

    Args:
        conjugacy_class (tuple[int]) : A conjugacy class, in partition form, of Sp
        unitary_dimension (Symbol) : Dimension d of the unitary matrix U

    Returns:
        Symbol : The Weingarten function

    Raise:
        TypeError : unitary_dimension must be an instance of int or sympy.Symbol
    """
    if not isinstance(unitary_dimension, (Symbol, int)):
        raise TypeError("unitary_dimension must be an instance of int or sympy.Symbol")

    degree = sum(conjugacy_class)
    partition_list = [
        sum([value * (key,) for key, value in part.items()], ())
        for part in partitions(degree)
    ]
    irrep_dimension_list = [sn_dimension(part) for part in partition_list]

    if isinstance(unitary_dimension, int):
        weingarten = sum(
            Fraction(
                irrep_dimension**2 * murn_naka(part, conjugacy_class),
                ud_dimension(part, unitary_dimension),
            )
            for part, irrep_dimension in zip(partition_list, irrep_dimension_list)
        ) * Fraction(1, factorial(degree) ** 2)
    else:
        weingarten = (
            sum(
                irrep_dimension**2
                * murn_naka(part, conjugacy_class)
                / ud_dimension(part, unitary_dimension)
                for part, irrep_dimension in zip(partition_list, irrep_dimension_list)
            )
            / factorial(degree) ** 2
        )
        numerator, denominator = fraction(simplify(weingarten))
        weingarten = factor(numerator) / factor(denominator)

    return weingarten


def weingarten_element(
    cycle: Permutation, degree: int, unitary_dimension: Symbol
) -> Symbol:
    """Returns the Weingarten function

    Args:
        cycle (Permutation) : Permutation cycle from the symmetric group
        degree (int) : Degree p of the symmetric group Sp
        unitary_dimension (Symbol) : Dimension d of the unitary matrix U

    Returns:
        Symbol : The Weingarten function
    """
    conjugacy_class = list(get_class(cycle, degree))
    return weingarten_class(conjugacy_class, unitary_dimension)


def string_permutation(str_target: str, str_shuffled: str):
    """Returns a generator of the ordering permutations

    Args:
        str_target (str) : The target string
        str_shuffled (str) : The shuffled string

    Returns:
        Generator : Generator with ordering permutations
    """
    # different sizes or components
    if sorted(str_target) != sorted(str_shuffled):
        return ()

    # unique element
    string_len = len(str_target)
    string_set_len = len(set(str_target))
    if string_set_len == 1:
        return (cycle for cycle in SymmetricGroup(string_len).elements)

    # set case
    cycle_target = Permutation(sorted(range(len(str_target)), key=lambda i: str_target[i]))
    cycle_shuffled = Permutation(sorted(range(len(str_shuffled)), key=lambda i: str_shuffled[i]))
    if string_len == string_set_len:
        return (~cycle_target * cycle_shuffled for _ in (1,))

    # general case
    sorted_string = cycle_target(str_target)
    identical_groups = (
        (sorted_string.index(j), sorted_string.count(j)) for j in set(sorted_string)
    )
    seed = (Permutation(string_len - 1),)  # the seed is the identity
    for index, size in identical_groups:
        if size == 1:
            continue
        symm = (
            Permutation([[i + index for i in j] for j in p.cyclic_form])
            for p in SymmetricGroup(size).elements
        )
        seed = (i * j for i, j in product(seed, symm))

    return (~cycle_target * cycle * cycle_shuffled for cycle in seed)


def haar_integral(str_target: str, str_shuffled: str, group_dimension: int) -> Symbol:
    """Returns integral under Haar measure

    Args:
        str_target (str) : The target string
        str_shuffled (str) : The shuffled string
        group_dimension (int) : Dimension of the compact group

    Returns:
        Symbol : Integral under the Haar measure
    """
    str_i, str_j = str_target[::2], str_target[1::2]
    str_i_prime, str_j_prime = str_shuffled[::2], str_shuffled[1::2]
    if len(str_i) != len(str_j):
        raise ValueError("Requires two strings of even size")
    if sorted(str_i) != sorted(str_i_prime) or sorted(str_j) != sorted(str_j_prime):
        return 0
    
    degree = len(str_i)
    class_mapping = dict(Counter(
        get_class(cycle_i*~cycle_j, degree) for cycle_i, cycle_j in product(
            string_permutation(str_i,str_i_prime),
            string_permutation(str_j,str_j_prime)
        )
    ))
    integral = sum(frequency*weingarten_class(conjugacy, group_dimension) for conjugacy, frequency in class_mapping.items())

    if isinstance(group_dimension, Symbol):
        numerator, denominator = fraction(simplify(integral))
        integral = factor(numerator) / factor(denominator)

    return integral