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

    [4] Gorin, T., & López, G. V. (2008). Monomial integrals on the classical groups. Journal of
    mathematical physics, 49(1).
"""

from math import prod, comb
from fractions import Fraction
from itertools import product
from functools import lru_cache
from collections import Counter
from sympy import Symbol, Expr, factorial, rf, Integer, Rational
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
from ._utils import (
    _simplify,
    _generate_matrices_with_row_sums,
    _vector_multinomial,
    _matrix_to_sequence,
    _sequence_to_matrix,
    _is_power_matrix,
)


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
def _haar_integral_orthogonal_collins(
    sequences: tuple[tuple[int, ...], ...], orthogonal_dimension: Symbol
) -> Expr:
    """Returns the integral over orthogonal group polynomial sampled at random from the Haar measure
    using Weingarten calculus

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
    """
    seq_i, seq_j = sequences
    degree = len(seq_i)

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


@lru_cache
def _column_integral_orthogonal(col_vector: tuple[int, ...], group_dimension: Symbol) -> Expr:
    """Integral over a single column of an orthogonal matrix

    Parameters
    ----------
    col_vector : tuple[int, ...]
        A vector of power of the orthogonal entries

    group_dimension : Symbol
        The dimension of the orthogonal group

    Returns
    -------
    Expr
        The integral under the Haar measure
    """
    if any(x % 2 for x in col_vector):
        return 0

    half_total = sum(col_vector) // 2

    numerator = Integer(1)
    for col_element in col_vector:
        numerator *= rf(Rational(1, 2), col_element // 2)

    denominator = rf(
        (
            group_dimension / 2
            if isinstance(group_dimension, Symbol)
            else Rational(group_dimension, 2)
        ),
        half_total,
    )

    return _simplify(numerator / denominator)


@lru_cache
def _haar_integral_orthogonal_gorin(
    power_matrix: tuple[tuple[int, ...], ...],
    group_dimension: Symbol | int,
) -> Expr:
    """Returns the integral over orthogonal group polynomial sampled at random from the Haar measure
    using Gorin's algorithm

    Parameters
    ----------
    power_matrix : tuple[tuple[int, ...], ...]
        Power matrix of non-negative integers

    orthogonal_dimension : Symbol
        The dimension of the orthogonal group

    Returns
    -------
    Expr
        The integral under the Haar measure
    """
    row_count, col_count = len(power_matrix), len(power_matrix[0])
    if col_count == 1:
        column = tuple(power_matrix[i][0] for i in range(row_count))
        return _column_integral_orthogonal(column, group_dimension)

    # last column recursion
    last_col = tuple(power_matrix[i][col_count - 1] for i in range(row_count))
    last_col_sum = sum(last_col)

    # most probably remove since already tested that all rows and columns are even
    if last_col_sum % 2:
        return 0

    power_matrix_crop = tuple(tuple(row[: col_count - 1]) for row in power_matrix)

    # the following can be removed if previously remove 0 rows and columns
    if last_col_sum == 0:
        return _haar_integral_orthogonal_gorin(power_matrix_crop, group_dimension)

    integral = 0
    # kappa[i] is even and 0 <= kappa[i] <= last_col[i]
    kappa_vector_options = [list(range(0, m + 1, 2)) for m in last_col]

    # iterate on last column
    for kappa_vector in product(*kappa_vector_options):
        kappa_vector = tuple(kappa_vector)
        kappa_sum = sum(kappa_vector)

        # most probably can remove, kappa should never be odd
        if kappa_sum % 2:
            continue

        vector_binomial = prod(comb(m, k) for m, k in zip(last_col, kappa_vector))
        kappa_integral = _column_integral_orthogonal(kappa_vector, group_dimension)

        # I don't believe that should ever happen if kappa is never zero
        if kappa_integral == 0:
            continue

        a, b = last_col_sum // 2, kappa_sum // 2
        z1 = (
            group_dimension / 2
            if isinstance(group_dimension, Symbol)
            else Rational(group_dimension, 2)
        )
        z2 = Rational(col_count - 1, 2)
        b_function = _simplify((-1) ** (a - b) * rf(z1, b) * rf(z1, a - b) / rf(z1 - z2, a))

        col_coefficient = vector_binomial * kappa_integral * b_function

        prescribed_row_sum = tuple(m - k for m, k in zip(last_col, kappa_vector))

        # iterate over power truncated power matrices
        reduced_integral = 0
        for power_matrix_k in _generate_matrices_with_row_sums(prescribed_row_sum, col_count - 1):
            kcs_vector = tuple(
                sum(power_matrix_k[i][j] for i in range(len(power_matrix_k)))
                for j in range(len(power_matrix_k[0]))
            )

            # Only even column sums contribute, because of the one-vector average
            if any(x % 2 for x in kcs_vector):
                continue

            kcs_integral = _column_integral_orthogonal(kcs_vector, group_dimension)

            # I don't believe the following should happen
            if kcs_integral == 0:
                continue

            next_power_matrix = tuple(
                tuple(a + b for a, b in zip(row_m, row_k))
                for row_m, row_k in zip(power_matrix_crop, power_matrix_k)
            )

            recursive_integral = _haar_integral_orthogonal_gorin(next_power_matrix, group_dimension)

            # should never happen
            if recursive_integral == 0:
                continue

            reduced_integral += (
                Integer(_vector_multinomial(prescribed_row_sum, power_matrix_k))
                * kcs_integral
                * recursive_integral
            )

        integral += col_coefficient * reduced_integral

    return _simplify(integral)


@lru_cache
def haar_integral_orthogonal(
    monomial : tuple[tuple[int, ...], ...],
    orthogonal_dimension : Symbol | int, 
    algorithm : str = "collins",
    input : str = "sequence",
) -> Expr:
    """Returns the integral over orthogonal group polynomial sampled at random from the Haar measure

    Parameters
    ----------
    monomial : tuple[tuple[int, ...], ...]
        Sequences of matrix elements or a power matrix of non-negative integers

    orthogonal_dimension : Symbol
        The dimension of the orthogonal group

    algorithm : str
        The algorithm to be used to compute the integral. Either ``Collins`` or ``Gorin``

    input : str
        The input type of ``monomial`` can be either ``sequences`` or ``matrix``

    Returns
    -------
    Expr
        The integral under the Haar measure

    Raises
    ------
    TypeError
        If ``algorithm`` or ``orthogonal_dimension`` are not of type ``str``
    ValueError
        If ``algorithm`` is neither ``Collins`` nor ``Gorin``
    ValueError
        If ``input`` is neither ``matrix`` nor ``sequences``
    ValueError
        If the argument ``monomial`` does not contain 2 tuples with ``input`` type ``sequences``
    ValueError
        If the sequences are of different lengths with ``input`` type ``sequences``
    ValueError
        If ``monomial`` is not a proper power matrix with ``input`` type ``matrix``

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
    #CHANGE SEQUENCES AND POWER_MATRIX TO MONOMIAL!!!!!!!!
    #ADD EXAMPLES WITH THE COLLINS ALGORITHM
    if not isinstance(algorithm, str) or not isinstance(input, str):
        raise  TypeError
    
    if not isinstance(orthogonal_dimension, (Symbol, int)):
        raise TypeError
    
    algorithm, input = algorithm.lower(), input.lower()

    if not algorithm in ("collins", "gorin"):
        raise ValueError
    
    if not input in ("matrix", "sequences"):
        raise ValueError
    
    # trivial case
    if input == "sequences":
        if len(monomial) != 2 or len(monomial[0]) != len(monomial[1]):
            raise ValueError("Wrong tuple format")
        if len(monomial[0]) % 2:
            return 0

    #ADD ERRORS IF THE POWER MATRIX IS NOT A PROPER POWER MATRIX
    #TEST THAT BOTH ALGORITHMS AGREE
    #TEST THAT THE GORIN INTEGRAL MATCH THE WEINGARTEN FUNCTION WITH SETUP CORRECTLY
    #trivial case
    if input == "matrix":
        if not _is_power_matrix(monomial):
            raise ValueError("Wrong power matrix format")
        
        if not monomial or not monomial[0]:
            return 1
        
        row_sum_list = [sum(row) for row in monomial]
        col_sum_list = [sum(monomial[i][j] for i in range(len(monomial))) for j in range(len(monomial[0]))]
        if any(s % 2 for s in row_sum_list) or any(s % 2 for s in col_sum_list):
            return 0
    
    if algorithm == "gorin":
        #DO _matrix_to_sequence(sequence_to_matrix(monomial)) to remove zero or define a 0 removal function
        power_matrix = _sequence_to_matrix((monomial[0],), (monomial[1],))[0] if input == "sequence" else monomial
        return _haar_integral_orthogonal_gorin(power_matrix, orthogonal_dimension)
    elif algorithm == "collins":
        sequences = _matrix_to_sequence(monomial) if input == "matrix" else monomial
        return _haar_integral_orthogonal_collins(sequences, orthogonal_dimension)
