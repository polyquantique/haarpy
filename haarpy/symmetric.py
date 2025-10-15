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
Symmetric group Python interface
"""

from math import factorial, prod
from functools import lru_cache
from typing import Generator
from fractions import Fraction
from sympy.combinatorics import Permutation


@lru_cache
def get_conjugacy_class(perm: Permutation, degree: int) -> tuple:
    """Returns the conjugacy class of an element of the symmetric group Sp

    Args:
        perm (Permutation): Permutation cycle from the symmetric group
        degree (integer): Order of the symmetric group

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
    if not isinstance(perm, Permutation):
        raise TypeError("Permutation must be of type sympy.combinatorics.permutations.Permutation")

    if perm.size > degree:
        raise ValueError("Incompatible degree and permutation cycle")

    perm = perm * Permutation(degree - 1)

    return tuple(
        sorted(
            (key for key, value in perm.cycle_structure.items() for _ in range(value)),
            reverse=True,
        )
    )


def derivative_tableaux(
    tableau: tuple[tuple[int]], increment: int, partition: tuple[int]
) -> Generator[tuple[tuple[int]], None, None]:
    """Takes a single tableau and adds the selected number to its contents
    in a way that keeps it semi-standard. All possible tableaux are yielded

    Args:
        tableau (tuple[tuple[int]]): An incomplet Young tableau
        increment (int): Selected number to be added
        partition (tuple[int]) : partition characterizing an irrep of Sp

    Yields:
        tuple[tuple[int]]: Modified tableaux
    """
    # empty tableau
    if not tableau[0]:
        yield ((increment,),) + tableau[1:]
        return

    # first row
    if len(tableau[0]) < partition[0] and tableau[0][-1] <= increment:
        yield (tableau[0] + (increment,),) + tableau[1:]

    # other rows
    for index, part in enumerate(partition[1:]):
        previous_row__length = len(tableau[index])
        current_row_length = len(tableau[index + 1])
        # first row input
        if not current_row_length:
            if tableau[index][0] <= increment:
                yield tuple(
                    row if i != index + 1 else row + (increment,) for i, row in enumerate(tableau)
                )
            return
        if (
            current_row_length < min(part, previous_row__length)
            and tableau[index][current_row_length] <= increment
        ):
            yield tuple(
                row if i != index + 1 else row + (increment,) for i, row in enumerate(tableau)
            )


@lru_cache
def semi_standard_young_tableaux(
    partition: tuple[int], conjugacy_class: tuple[int]
) -> set[tuple[tuple[int]]]:
    """all eligible semi-standard young tableaux based of the partition

    Args:
        partition (tuple[int]) : partition characterizing an irrep of Sp
        conjugacy_class (tuple[int]) : A conjugacy class, in partition form, of Sp

    Returns:
        set[tuple[tuple[int]]]: all eligible semi-standard young tableaux
    """
    tableaux = (tuple(() for _ in partition),)
    cell_values = (i for i, m in enumerate(conjugacy_class) for _ in range(m))
    for increment in cell_values:
        tableaux = {
            tableau
            for derivative in tableaux
            for tableau in derivative_tableaux(derivative, increment, partition)
        }

    return tableaux


@lru_cache
def proper_border_strip(tableau: tuple[tuple[int]], conjugacy_class: tuple[int]) -> bool:
    """Returns True if input Young tableau is a valid border-strip tableau

    Args:
        tableau (tuple[tuple[int]]) : A semi-standard Young tableau
        conjugacy_class (tuple[int]) : A conjugacy class, in partition form, of Sp

    Returns:
        bool : True if the tableau is a border-strip
    """
    if len(tableau) == 1:
        return True

    # skewness condition
    for cell_value in range(len(conjugacy_class)):
        matching = tuple(
            (k, {i for i, j in enumerate(row) if j == cell_value + 1})
            for k, row in enumerate(tableau)
            if cell_value + 1 in row
        )

        if len(matching) == 1:
            continue

        for current, following in zip(matching, matching[1:]):
            if following[0] != current[0] + 1 or not current[1] & following[1]:
                return False

    # 2x2 square condition
    for i in range(1, len(tableau)):
        for j in range(1, len(tableau[i])):
            if tableau[i][j] == tableau[i][j - 1] == tableau[i - 1][j] == tableau[i - 1][j - 1]:
                return False

    return True


@lru_cache
def murn_naka_rule(partition: tuple[int], conjugacy_class: tuple[int]) -> int:
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

    tableaux = semi_standard_young_tableaux(partition, conjugacy_class)
    tableaux = (tableau for tableau in tableaux if proper_border_strip(tableau, conjugacy_class))

    tableaux_set = ((set(row) for row in tableau) for tableau in tableaux)
    heights = (tuple(i for row in tableau for i in row) for tableau in tableaux_set)
    heights = (
        sum(height.count(unit) - 1 for unit in range(len(conjugacy_class))) for height in heights
    )

    character = sum((-1) ** height for height in heights)
    return character


@lru_cache
def irrep_dimension(partition: tuple[int]) -> int:
    """Returns the dimension of the irrep of the symmetric group Sp labelled by the input partition

    Args:
        partition (tuple[int]) : A partition labelling an irrep of Sp

    Returns:
        int : The dimension of the irrep
    """
    numerator = prod(
        part_i - part_j + j + 1
        for i, part_i in enumerate(partition)
        for j, part_j in enumerate(partition[i + 1 :])
    )
    denominator = prod(factorial(part + len(partition) - i - 1) for i, part in enumerate(partition))
    dimension = Fraction(numerator, denominator) * factorial(sum(partition))

    return dimension.numerator