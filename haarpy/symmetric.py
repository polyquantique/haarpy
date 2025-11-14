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
from sympy.combinatorics import (
    Permutation,
    PermutationGroup,
    SymmetricGroup,
    DirectProduct,
)
from haarpy import perfect_matchings, join_operation


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
        raise TypeError(
            "Permutation must be of type sympy.combinatorics.permutations.Permutation"
        )

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
                    row if i != index + 1 else row + (increment,)
                    for i, row in enumerate(tableau)
                )
            return
        if (
            current_row_length < min(part, previous_row__length)
            and tableau[index][current_row_length] <= increment
        ):
            yield tuple(
                row if i != index + 1 else row + (increment,)
                for i, row in enumerate(tableau)
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
def proper_border_strip(
    tableau: tuple[tuple[int]], conjugacy_class: tuple[int]
) -> bool:
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
            if (
                tableau[i][j]
                == tableau[i][j - 1]
                == tableau[i - 1][j]
                == tableau[i - 1][j - 1]
            ):
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
    tableaux = (
        tableau for tableau in tableaux if proper_border_strip(tableau, conjugacy_class)
    )

    tableaux_set = ((set(row) for row in tableau) for tableau in tableaux)
    heights = (tuple(i for row in tableau for i in row) for tableau in tableaux_set)
    heights = (
        sum(height.count(unit) - 1 for unit in range(len(conjugacy_class)))
        for height in heights
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
    denominator = prod(
        factorial(part + len(partition) - i - 1) for i, part in enumerate(partition)
    )
    dimension = Fraction(numerator, denominator) * factorial(sum(partition))

    return dimension.numerator


@lru_cache
def sorting_permutation(*sequence: tuple[int]) -> Permutation:
    """Returns the sorting permutation of a given sequence

    Args:
        sequence (tuple[int]): a sequence of unorderd elements

    Returns:
        Permutation: the sorting permutation

    Raise:
        ValueError: for incompatible sequence inputs
        TypeError: if more than two sequences are passed as arguments
    """
    if len(sequence) == 1:
        return Permutation(
            sorted(range(len(sequence[0])), key=lambda k: sequence[0][k])
        )
    if len(sequence) == 2:
        if sorted(sequence[0]) != sorted(sequence[1]):
            raise ValueError("Incompatible sequences")
        return ~sorting_permutation(sequence[1]) * sorting_permutation(sequence[0])

    raise TypeError


def YoungSubgroup(partition: tuple[int]) -> PermutationGroup:
    """Returns the Young subgroup of a given input partition
    See `<https://en.wikipedia.org/wiki/YoungSubgroup>`_

    Args:
        partition (tuple[int]): A partition

    Returns:
        PermutationGroup: the associated Young subgroup

    Raise:
        TypeError: if partition is not a tuple or a list
        TypeError: if partition is not made of positive integers
    """
    if not isinstance(partition, (tuple, list)):
        raise TypeError
    if not all(isinstance(part, int) and part > 0 for part in partition):
        raise TypeError
    return DirectProduct(*[SymmetricGroup(part) for part in partition])


def stabilizer_coset(*sequence: tuple) -> Generator[Permutation, None, None]:
    """Returns all permutations that, when acting on sequence[0], return sequence[1]

    Args:
        *sequence (tuple): the sequences acted upon

    Returns:
        Generator[Permutation]: permutations that, when acting on sequence[0], return sequence[1]

    Raise:
        TypeError: if the sequence argument contains more than two sequences
    """
    if len(sequence) == 1:
        sequence = tuple(sequence[0] for _ in range(2))
    elif len(sequence) == 2:
        if sorted(sequence[0]) != sorted(sequence[1]):
            return ()
    else:
        raise TypeError

    young_partition = tuple(sequence[0].count(i) for i in sorted(set(sequence[0])))

    return (
        ~sorting_permutation(sequence[1])
        * permutation
        * sorting_permutation(sequence[0])
        for permutation in YoungSubgroup(young_partition).generate()
    )


@lru_cache
def HyperoctahedralGroup(degree: int) -> PermutationGroup:
    """Return the hyperoctahedral group

    Args:
        degree (int): The degree k of the hyperoctahedral group H_k

    Returns:
        (PermutationGroup): The hyperoctahedral group

    Raise:
        TypeError: If degree is not of type int
    """
    if not isinstance(degree, int):
        raise TypeError
    transpositions = tuple(
        Permutation(2 * degree - 1)(2 * i, 2 * i + 1) for i in range(degree)
    )
    double_transpositions = tuple(
        Permutation(2 * degree - 1)(2 * i, 2 * j)(2 * i + 1, 2 * j + 1)
        for i in range(degree)
        for j in range(i + 1, degree)
    )
    return PermutationGroup(transpositions + double_transpositions)


def hyperoctahedral_transversal(degree: int) -> Generator[Permutation, None, None]:
    """Returns a generator with the permutations of M_2k, the complete set of coset
    representatives of S_2k/H_k as seen in Macdonald's "Symmetric Functions and Hall
    Polynomials" chapter VII

    Args:
        degree (int): Degree 2k of the set M_2k

    Returns:
        (Generator[Permutation]): The permutations of M_2k
    """
    if degree % 2:
        raise ValueError("degree should be a factor of 2")
    if degree == 2:
        return (Permutation(1),)
    flatten_pmp = (
        tuple(i for pair in pmp for i in pair)
        for pmp in perfect_matchings(tuple(range(degree)))
    )
    return (Permutation(pmp) for pmp in flatten_pmp)


@lru_cache
def coset_type(permutation: Permutation) -> tuple[int]:
    """Returns the coset-type of a given permutation of S_2k as seen
    `Matsumoto. Weingarten calculus for matrix ensembles associated with
    compact symmetric spaces <https://arxiv.org/pdf/1301.5401>`_

    Args:
        permutation (Permutation): A permutation of the symmetric group S_2k

    Returns:
        tuple[int]: The associated coset-type as a partition of k

    Raise:
        TypeError: If partition is not a Permutation
        ValueError: If the symmetric group is of odd degree
    """
    if not isinstance(permutation, Permutation):
        raise TypeError

    degree = permutation.size
    if degree % 2:
        raise ValueError("Coset-type are only defined for even sized permutations")

    array_form = permutation.array_form
    base_partition = tuple((2 * i, 2 * i + 1) for i in range(degree // 2))
    matching_partition = tuple(
        (array_form[2 * i], array_form[2 * i + 1]) for i in range(degree // 2)
    )
    return tuple(
        sorted(
            (
                len(block) // 2
                for block in join_operation(base_partition, matching_partition)
            ),
            reverse=True,
        )
    )


@lru_cache
def coset_type_representative(partition: tuple[int]) -> Permutation:
    """Returns a representative permutation of S_2k for a given
    input coset-type (partition of k) as seen
    `Matsumoto. Weingarten calculus for matrix ensembles associated with
    compact symmetric spaces <https://arxiv.org/pdf/1301.5401>`_

    Args:
        partition (tuple[int]): The coset-type (partition of k)

    Returns:
        (Permutation): The associated permutation of S_2k

    Raise:
        TypeError: If partition is not a tuple
    """
    if not isinstance(partition, tuple):
        raise TypeError

    half_degree = sum(partition)
    degree = 2 * half_degree
    permutation_list = degree * [None]
    for r, _ in enumerate(partition):
        partial_sum = sum(partition[:r])
        permutation_list[2 * partial_sum] = 2 * partial_sum
        permutation_list[2 * partial_sum + 1] = 2 * partial_sum + 2 * partition[r] - 1
        for p in range(3, 2 * partition[r] + 1):
            permutation_list[2 * partial_sum + p - 1] = 2 * partial_sum + p - 2

    return Permutation(permutation_list)
