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
Partition Python interface
"""

from __future__ import annotations
from typing import Generator
from itertools import product
from sympy.combinatorics.partitions import Partition as SympyPartition
from sympy.utilities.iterables import multiset_partitions
from sympy.sets.sets import FiniteSet


# pylint: disable=abstract-method
class Partition(SympyPartition):
    """
    Custom subclass of Sympy's Partition class
    This class represents an abstract partition
    A partition is a set of disjoint sets whose union equals a given set
    See  sympy.utilities.iterables.partitions

    Attributes:
        partition (list[list[int]]): Return partition as a sorted list of lists
        members (tuple[int]): Elements of the Partition - Integer from 0 to size-1
        size (int): size of the partition
        is_crossing (bool): True for crossing partitions, False otherwise
        is_perfect_matching (bool): True for perfect matching partitions, False otherwise
    """

    size: int  # type hint for pylint
    _is_crossing = None
    _is_perfect_matching = None

    def __new__(cls, *partition: FiniteSet) -> SympyPartition:
        """
        Initializes a Partition

        Args:
            partition (Finiteset): Each argument to Partition should be a list, set, or a FiniteSet

        Returns:
            SympyPartition: An instance of Sympy's Partition

        Raise:
            ValueError: Integers 0 through len(Partition)-1 must be present.
        """
        obj = super().__new__(cls, *partition)
        if obj.members != tuple(range(obj.size)):
            raise ValueError(
                f"Integers 0 through {obj.size - 1} should be present given the partition size."
            )

        return obj

    def __le__(self, other: Partition) -> bool:
        """
        Checks if a partition is less than or equal to the other based on partial order

        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self <= other

        Raise:
            TypeError: if other is not an instance of Partition
            ValueError: if other and self are of different size
        """
        if not isinstance(other, Partition):
            raise TypeError

        if self.size != other.size:
            raise ValueError("Cannot compare partitions of different sizes.")

        for self_block in self:
            if not any(self_block.issubset(other_block) for other_block in other):
                return False

        return True

    def __lt__(self, other: Partition) -> bool:
        """
        Checks if a partition is less than the other based on partial order

        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self < other
        """
        return self <= other and self != other

    def __ge__(self, other: Partition) -> bool:
        """
        Checks if a partition is greater or equal than the other based on partial order

        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self >= other
        """
        return other <= self

    def __gt__(self, other: Partition) -> bool:
        """
        Checks if a partition is greater than the other based on partial order

        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self > other
        """
        return other < self

    def __and__(self, other: Partition) -> Partition:
        """Returns the greatest lower bound of the two input partitions

        Args:
            other (Partition): Partition to meet

        Return:
            Partition: Greatest lower bound Partition
        """
        meet_list = [
            self_block & other_block
            for self_block, other_block in product(self, other)
            if self_block & other_block
        ]

        return Partition(*meet_list)

    def __or__(self, other: Partition) -> Partition:
        """Returns the least upper bound of the two input partitions

        Args:
            other (Partition): Partition to join

        Return:
            Partition: Least lower bound Partition
        """
        parent = [
            {
                index
                for element in self_block
                for index, other_block in enumerate(other)
                if element in other_block
            }
            for self_block in self
        ]

        merged = []
        for index_set in parent:
            overlap = [m for m in merged if m & index_set]
            for m in overlap:
                index_set |= m
                merged.remove(m)
            merged.append(index_set)

        return Partition(
            *[
                [element for index in block for element in other.partition[index]]
                for block in merged
            ]
        )

    def meet(self, other: Partition) -> Partition:
        """Returns the greatest lower bound of the two input partitions

        Args:
            other (Partition): Partition to meet

        Return:
            Partition: Greatest lower bound Partition
        """
        return self & other

    def join(self, other: Partition) -> Partition:
        """Returns the least upper bound of the two input partitions

        Args:
            other (Partition): Partition to join

        Return:
            Partition: Least lower bound Partition
        """
        return self | other

    def partial_order(self, other: Partition) -> bool:
        """
        Checks if a partition is less than or equal to the other based on partial order

        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self <= other
        """
        return self <= other

    @property
    def is_crossing(self) -> bool:
        """
        Checks if the partition is a crossing partition
        Computed lazily and stored in _is_crossing

        Returns:
            bool: True if the partition is crossing, False otherwise
        """
        if self._is_crossing is None:
            filtered_partition = [
                block
                for block in self.partition
                if len(block) != 1 and block[0] + 1 != block[-1]
            ]

            for index, previous_block in enumerate(filtered_partition[:-1]):
                for next_block in filtered_partition[index + 1 :]:
                    if previous_block[-1] < next_block[0]:
                        break
                    for element in previous_block[1:]:
                        if next_block[0] < element < next_block[-1]:
                            self._is_crossing = True
                            return self._is_crossing

            self._is_crossing = False

        return self._is_crossing
    
    @property
    def is_perfect_matching(self) -> bool:
        """
        Checks if the partition is a crossing partition
        Computed lazily and stored in _is_crossing

        Returns:
            bool: True if the partition is crossing, False otherwise
        """
        if self._is_perfect_matching is None:
            self._is_perfect_matching = all(len(block) == 2 for block in self.partition)
        return self._is_perfect_matching

    def fattening(self) -> Partition:
        """
        """
        return
    
    def thinning(self) -> Partition:
        """
        """
        if not self.is_perfect_matching:
            raise ValueError("Thinning only applies to perfect matching partitions.")
        return


def set_partitions(size: int) -> Generator[Partition, None, None]:
    """Returns the partitionning of a set [size] into non-empty subsets.

    Args:
        size (int): size of the sequence [size]

    Returns:
        Generator[Partition]: all partitions of set [size]

    Raise:
        TypeError: if size is not int
        ValueError: if size <= 0
    """
    if not isinstance(size, int):
        raise TypeError
    
    if  size <= 0:
        raise ValueError("size must be an integer greater than 0.")
    
    return (Partition(*partition) for partition in multiset_partitions(range(size)))
