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

    [3] Macdonald, I. G. (1998). Symmetric functions and Hall polynomials. Oxford university press.

    [4] Gorin, T., & López, G. V. (2008). Monomial integrals on the classical groups. Journal of
    mathematical physics, 49(1).
"""

from math import factorial, prod, comb
from functools import lru_cache
from itertools import product
from collections import Counter
from fractions import Fraction
from sympy import Symbol, Expr, Integer, rf, factor
from sympy.combinatorics import Permutation
from sympy.utilities.iterables import partitions
from haarpy import (
    get_conjugacy_class,
    murn_naka_rule,
    irrep_dimension,
    stabilizer_coset,
)
from ._utils import (
    _simplify,
    _generate_matrices_with_row_sums,
    _vector_multinomial,
    _matrix_to_sequence,
    _sequence_to_matrix,
    _is_power_matrix,
    _compressed_unitary_pair,
)


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
            if representation_dimension(part, unitary_dimension)
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
def _haar_integral_unitary_collins(
    sequences: tuple[tuple[int, ...], ...], unitary_dimension: Symbol
) -> Expr:
    """Returns the integral of a monomial over the unitary group using Weingarten calculus

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
    """
    seq_i, seq_j, seq_i_prime, seq_j_prime = sequences

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


@lru_cache
def _column_integral_unitary(
    col_vector_m: tuple[int, ...], col_vector_n: tuple[int, ...], group_dimension: Symbol
) -> Expr:
    """Integral over a single column of a unitary matrix

    Parameters
    ----------
    col_vector_m : tuple[int, ...]
        A vector of power of the unitary entries

    col_vector_m : tuple[int, ...]
        A vector of power of the conjugated unitary entries

    group_dimension : Symbol
        The dimension of the unitary group

    Raises
    ------
    ValueError
        If both input vectors have different lengths

    Returns
    -------
    Expr
        The integral under the Haar measure
    """
    if len(col_vector_m) != len(col_vector_n):
        raise ValueError

    if not all(m == n for m, n in zip(col_vector_m, col_vector_n)):
        return 0

    numerator = Integer(prod(factorial(m) for m in col_vector_m))
    denominator = rf(group_dimension, sum(col_vector_m))

    return _simplify(numerator / denominator)


@lru_cache
def _haar_integral_unitary_gorin(
    power_matrix_m: tuple[tuple[int, ...], ...],
    power_matrix_n: tuple[tuple[int, ...], ...],
    unitary_dimension: Symbol | int,
) -> Expr:
    """Returns the integral over unitary group polynomial sampled at random from the Haar measure
    using Gorin's algorithm

    Parameters
    ----------
    power_matrix_m : tuple[tuple[int, ...], ...]
        Power matrix of non-negative integers for unitary monomial

    power_matrix_n : tuple[tuple[int, ...], ...]
        Power matrix of non-negative integers for conjugate unitary monomial

    unitary_dimension : Symbol
        The dimension of the unitary group

    Returns
    -------
    Expr
        The integral under the Haar measure
    """
    col_sum_tuple_m = tuple(
        sum(power_matrix_m[i][j] for i in range(len(power_matrix_m)))
        for j in range(len(power_matrix_m[0]))
    )
    col_sum_tuple_n = tuple(
        sum(power_matrix_n[i][j] for i in range(len(power_matrix_n)))
        for j in range(len(power_matrix_n[0]))
    )
    if not all(
        col_sum == col_sum_conj for col_sum, col_sum_conj in zip(col_sum_tuple_m, col_sum_tuple_n)
    ):
        return 0

    row_count, col_count = len(power_matrix_m), len(power_matrix_m[0])

    if row_count == 1:
        return _column_integral_unitary(power_matrix_m[0], power_matrix_n[0], unitary_dimension)

    if col_count == 1:
        col_vector_m = tuple(power_matrix_m[i][0] for i in range(row_count))
        col_vector_n = tuple(power_matrix_n[i][0] for i in range(row_count))
        return _column_integral_unitary(col_vector_m, col_vector_n, unitary_dimension)

    last_col_m = tuple(power_matrix_m[i][col_count - 1] for i in range(row_count))
    last_col_n = tuple(power_matrix_n[i][col_count - 1] for i in range(row_count))

    last_col_sum = sum(last_col_m)

    power_previous_m = tuple(tuple(row[: col_count - 1]) for row in power_matrix_m)
    power_previous_n = tuple(tuple(row[: col_count - 1]) for row in power_matrix_n)

    integral = 0

    kappa_options = [range(0, min(m, n) + 1) for m, n in zip(last_col_m, last_col_n)]

    for kappa_vector in product(*kappa_options):
        kappa_vector = tuple(kappa_vector)
        kappa_sum = sum(kappa_vector)

        binomial_coeff = prod(comb(m, k) for m, k in zip(last_col_m, kappa_vector)) * prod(
            comb(n, k) for n, k in zip(last_col_n, kappa_vector)
        )

        kappa_integral = _column_integral_unitary(kappa_vector, kappa_vector, unitary_dimension)

        a, b = last_col_sum, kappa_sum
        z1, z2 = unitary_dimension, col_count - 1
        b_function = _simplify((-1) ** (a - b) * rf(z1, b) * rf(z1, a - b) / rf(z1 - z2, a))

        col_coefficient = binomial_coeff * kappa_integral * b_function

        prescribed_row_sum_k = tuple(m - k for m, k in zip(last_col_m, kappa_vector))
        prescribed_row_sum_l = tuple(n - k for n, k in zip(last_col_n, kappa_vector))

        reduced_integral = 0

        for power_matrix_k in _generate_matrices_with_row_sums(prescribed_row_sum_k, col_count - 1):
            kcs_vector = tuple(
                sum(power_matrix_k[i][j] for i in range(len(power_matrix_k)))
                for j in range(len(power_matrix_k[0]))
            )
            multinomial_k = _vector_multinomial(prescribed_row_sum_k, power_matrix_k)

            next_power_matrix_m = tuple(
                tuple(a + b for a, b in zip(row_m, row_k))
                for row_m, row_k in zip(power_previous_m, power_matrix_k)
            )

            for power_matrix_l in _generate_matrices_with_row_sums(
                prescribed_row_sum_l, col_count - 1
            ):
                lcs_vector = tuple(
                    sum(power_matrix_l[i][j] for i in range(len(power_matrix_l)))
                    for j in range(len(power_matrix_l[0]))
                )
                multinomial_l = _vector_multinomial(prescribed_row_sum_l, power_matrix_l)

                next_power_matrix_n = tuple(
                    tuple(a + b for a, b in zip(row_n, row_l))
                    for row_n, row_l in zip(power_previous_n, power_matrix_l)
                )

                cs_integral = _column_integral_unitary(kcs_vector, lcs_vector, unitary_dimension)

                recursive_integral = _haar_integral_unitary_gorin(
                    next_power_matrix_m, next_power_matrix_n, unitary_dimension
                )
                reduced_integral += multinomial_k * multinomial_l * cs_integral * recursive_integral

        integral += col_coefficient * reduced_integral

    return (
        factor(_simplify(integral)) if isinstance(unitary_dimension, Symbol) else Fraction(integral)
    )


@lru_cache
def haar_integral_unitary(
    monomial: tuple[tuple[int, ...], ...],
    monomial_conjugate: tuple[tuple[int, ...], ...],
    unitary_dimension: Symbol | int,
    algorithm: str = "collins",
    structure: str = "sequences",
) -> Expr:
    """Returns the integral over unitary group polynomial sampled at random from the Haar measure

    Parameters
    ----------
    monomial : tuple[tuple[int, ...], ...]
        Sequences of matrix elements or a power matrix of non-negative integers

    monomial_conjugate : tuple[tuple[int, ...], ...]
        Sequences of matrix elements or a power matrix of non-negative integers

    unitary_dimension : Symbol
        The dimension of the unitary group

    algorithm : str
        The algorithm to be used to compute the integral. Either ``Collins`` or ``Gorin``

    structure : str
        The type of ``monomial`` can be either ``sequences`` or ``matrix``

    Returns
    -------
    Expr
        The integral under the Haar measure

    Raises
    ------
    TypeError
        If ``algorithm``, ``structure`` or ``unitary_dimension`` have the wrong type
    ValueError
        If ``algorithm`` is neither ``Collins`` nor ``Gorin``
    ValueError
        If ``structure`` is neither ``matrix`` nor ``sequences``
    ValueError
        If the argument ``monomial`` does not contain 2 tuples with ``structure`` type ``sequences``
    ValueError
        If the argument ``monomial_conjugate`` does not contain 2 tuples with ``structure`` type ``sequences``
    ValueError
        If the sequences are of different lengths with ``structure`` type ``sequences``
    ValueError
        If ``monomial`` or  ``monomial_conjugate`` are not proper power matrix with ``structure`` type ``matrix``

    Returns
    -------
    Expr
        The integral under the Haar measure

    Examples
    --------
    >>> from sympy import Symbol
    >>> from haarpy import haar_integral_unitary
    >>> d = Symbol("d")
    >>> sequences = ((0, 1, 2), (0, 0, 1))
    >>> sequences_conjugate = ((0, 1, 2), (0, 1, 0))
    >>> haar_integral_unitary(sequences, sequences_conjugate, 5)
    Fraction(-1, 840)
    >>> haar_integral_unitary(sequences, sequences_conjugate, d)
    -1/(d*(d - 1)*(d + 1)*(d + 2))
    >>> power = ((1, 0, 0), (1, 0, 0), (0, 1, 0))
    >>> power_conjugate = ((1, 0, 0), (0, 1, 0), (1, 0, 0))
    >>> haar_integral_unitary(power, power_conjugate, d, algorithm = "Gorin", structure = "matrix")
    -1/(d*(d - 1)*(d + 1)*(d + 2))

    See Also
    --------
    :func:`haarpy.symmetric.stabilizer_coset`
        Returns all permutations sending a first sequence to a second sequence
    :func:`haarpy.unitary.weingarten_unitary`
        Returns the unitary Weingarten function
    """
    if (
        not isinstance(algorithm, str)
        or not isinstance(structure, str)
        or not isinstance(unitary_dimension, (Symbol, int))
    ):
        raise TypeError

    algorithm, structure = algorithm.lower(), structure.lower()

    if algorithm not in ("collins", "gorin") or structure not in ("matrix", "sequences"):
        raise ValueError(
            "The 'algorithm' must be either 'Collins' or 'Gorin'.\n"
            "The 'structure' must be either 'matrix' or 'sequences'"
        )

    # trivial case
    if structure == "sequences":
        if (
            len(monomial) != 2
            or len(monomial[0]) != len(monomial[1])
            or len(monomial_conjugate) != 2
            or len(monomial_conjugate[0]) != len(monomial_conjugate[1])
        ):
            raise ValueError("Wrong tuple format")

        if sorted(monomial[0]) != sorted(monomial_conjugate[0]) or sorted(monomial[1]) != sorted(
            monomial_conjugate[1]
        ):
            return 0

    # trivial case
    if structure == "matrix":
        if not _is_power_matrix(monomial) or not _is_power_matrix(monomial_conjugate):
            raise ValueError("Wrong power matrix format")

        row_sum_list = [sum(row) for row in monomial]
        row_sum_conj_list = [sum(row) for row in monomial_conjugate]
        if any(
            row_sum != row_sum_conj
            for row_sum, row_sum_conj in zip(row_sum_list, row_sum_conj_list)
        ):
            return 0

        col_sum_list = [
            sum(monomial[i][j] for i in range(len(monomial))) for j in range(len(monomial[0]))
        ]
        col_sum_conj_list = [
            sum(monomial_conjugate[i][j] for i in range(len(monomial_conjugate)))
            for j in range(len(monomial_conjugate[0]))
        ]
        if any(
            col_sum != col_sum_conj
            for col_sum, col_sum_conj in zip(col_sum_list, col_sum_conj_list)
        ):
            return 0

    if algorithm == "gorin":
        if structure == "sequences":
            power_matrix_m, power_matrix_n = _sequence_to_matrix(
                (monomial[0], monomial_conjugate[0]),
                (monomial[1], monomial_conjugate[1]),
            )
            power_matrix_m, power_matrix_n = _compressed_unitary_pair(
                (power_matrix_m, power_matrix_n)
            )
        else:
            power_matrix_m, power_matrix_n = _compressed_unitary_pair(
                (monomial, monomial_conjugate)
            )
        # returns 1 (the integral over the Haar measure) if the power matrix is empty
        return (
            _haar_integral_unitary_gorin(power_matrix_m, power_matrix_n, unitary_dimension)
            if power_matrix_m
            else 1
        )
    # algorithm == "collins"
    seq_i, seq_j = _matrix_to_sequence(monomial) if structure == "matrix" else monomial
    seq_i_prime, seq_j_prime = (
        _matrix_to_sequence(monomial_conjugate) if structure == "matrix" else monomial_conjugate
    )
    return _haar_integral_unitary_collins(
        (seq_i, seq_j, seq_i_prime, seq_j_prime), unitary_dimension
    )
