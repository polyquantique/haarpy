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
Permutation matrices Python interface
"""

from typing import Generator


def set_partition(collection: tuple) -> Generator[tuple[tuple], None, None]:
    """Returns the partitionning of a given collection (set) of objects
    into non-empty subsets.

    Args:
        collection (tuple): The collection (set) to be partitionned

    Returns:
        generator(tuple): all partitions of the input collection 
    """
    if len(collection) == 1:
        yield (collection,)
        return
    
    first = collection[0]
    for smaller in set_partition(collection[1:]):
        for index, subset in enumerate(smaller):
            yield smaller[:index] + ((first,) + subset,) + smaller[index + 1:]
        yield ((first,),) + smaller


#considerer only coding the partitial order function as a bool function
#I believe this can be done easily with sets objects in Python!!!
def zeta_function():
    return


def mobius_function():
    """as seen in Collin & Nagatsu "Weingarten calculus for centered
    random permutation matrices"
    """
    return


def weingarten_permutation():
    #1. generate sigma^tau
    #2. Generate set partitions of each idividual block
    #3. Itertool product all block to find all partial order
    return


def weingarten_centered_random_permutation():
    return


def haar_integral_permutation():
    return


def haar_integral_centered_random_permutation():
    return
