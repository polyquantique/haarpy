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
Utility Python interface
"""

from math import factorial
from itertools import product
from collections.abc import Iterable
from fractions import Fraction
from sympy import Add, Mul, together, fraction, factor, factor_list, Expr, PolificationFailed


def _simplify(expr_iter: Iterable[Expr], constant: Fraction = Fraction(1, 1)) -> Expr:
    """Factorizes a sum of rational fraction into a
    single factorized, simplified fraction

    Parameters
    ----------
    expr : Expr
        The expression to be simplified

    constant : Fraction
        A constant fraction multiplying the simplification result

    Returns
    -------
    Expr
        The simplified fraction
    """
    equation = together(Add(*expr_iter))
    # automatically factorises the denominator
    num, denum = fraction(constant * equation)

    # Error occurs if numerator is a constant
    try:
        num_factor_list = factor_list(num)
    except PolificationFailed:
        return factor(num) / denum

    denum_factor_list = factor_list(denum)

    # gets rid of common factors
    num_factor_dict, denum_factor_dict = dict(num_factor_list[1]), dict(denum_factor_list[1])
    for fact in num_factor_dict:
        if fact in denum_factor_dict:
            common = min(num_factor_dict[fact], denum_factor_dict[fact])
            num_factor_dict[fact] -= common
            denum_factor_dict[fact] -= common

    num_simplified = Mul(*(fact**power for fact, power in num_factor_dict.items()))
    denum_simplified = Mul(*(fact**power for fact, power in denum_factor_dict.items()))
    constant_simplified = Fraction(num_factor_list[0], denum_factor_list[0])

    return Mul(constant_simplified.numerator, num_simplified, evaluate=False) / Mul(
        constant_simplified.denominator, denum_simplified, evaluate=False
    )


def _generate_matrices_with_row_sums(
    row_sums: tuple[int, ...],
    col_count: int,
) -> Iterable[tuple[tuple[int, ...], ...]]:
    """
    Generate all nonnegative integer matrices with prescribed row sums

    Parameters
    ----------
    row_sums : tuple[int, ...]
        Sum of each row

    col_count : int
        The number of columns

    Returns
    -------
    iterator of tuple[tuple[int, ...], ...]
        Yielded all nonnegative integer matrix with prescribed row sum
    """

    def generate_compositions(total: int, length: int) -> Iterable[tuple[int, ...]]:
        "Generate all length-tuples of nonnegative integers summing to total"
        if length == 0:
            if total == 0:
                yield tuple()
            return

        if length == 1:
            yield (total,)
            return

        for first in range(total + 1):
            for rest in generate_compositions(total - first, length - 1):
                yield (first,) + rest

    row_options = [list(generate_compositions(r, col_count)) for r in row_sums]
    for rows in product(*row_options):
        yield tuple(rows)


def _vector_multinomial(
    row_sums: tuple[int, ...], power_matrix: tuple[tuple[int, ...], ...]
) -> int:
    """Product of multinomial coefficients"""

    def multinomial(total: int, parts: Iterable[int]) -> int:
        """Multinomial coefficient total! / prod parts_i!"""
        parts = list(parts)
        if sum(parts) != total:
            return 0

        out = factorial(total)
        for p in parts:
            out //= factorial(p)
        return out

    out = 1
    for s, row in zip(row_sums, power_matrix):
        out *= multinomial(s, row)
    return out


def _matrix_to_sequence(
    power_matrix: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    "Converts power matrix to sequences of row and column indices"
    row_index = tuple(
        i for i, j in enumerate(sum(k for k in row) for row in power_matrix) for _ in range(j)
    )
    col_index = tuple(
        i
        for row in power_matrix
        for i, j in zip(tuple(range(len(power_matrix))), row)
        for _ in range(j)
    )

    return row_index, col_index


def _sequence_to_matrix(
    row_index_tuple: tuple[tuple[int, ...], ...],
    col_index_tuple: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    "Converts sequences of row and column indices to a power matrix"
    if len(row_index_tuple) != len(col_index_tuple):
        raise ValueError
    if not all(len(row) == len(col) for row, col in zip(row_index_tuple, col_index_tuple)):
        raise ValueError

    index_set = set(
        index for index_list in row_index_tuple + col_index_tuple for index in index_list
    )
    index_dict = {value: index for index, value in enumerate(index_set)}
    index_length = len(index_dict)
    matrix_list = [
        [[0] * index_length for _ in range(index_length)] for _ in range(len(row_index_tuple))
    ]
    for position, index_values in enumerate(zip(row_index_tuple, col_index_tuple)):
        for r, c in zip(*index_values):
            matrix_list[position][index_dict[r]][index_dict[c]] += 1

    return [tuple(tuple(row) for row in matrix) for matrix in matrix_list]
