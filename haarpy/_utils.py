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

from collections.abc import Iterable
from fractions import Fraction
from sympy import Add, Mul, together, fraction, factor, factor_list, Expr, PolificationFailed


def _simplify(expr_iter: Iterable[Expr], constant: Fraction = Fraction(1, 1)) -> Expr:
    """Factorizes a sum of rational fraction into a
    single factorized, simplified fraction

    Parameters
    ----------
        expr (Expr) : the expression to be simplified
        constant (Fraction) : A constant fraction multiplying the simplification result

    Returns
    -------
        Expr : the simplified fraction
    """
    equation = together(Add(*expr_iter))
    #automatically factorises the denominator
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
