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

from sympy.combinatorics.partitions import Partition as SympyPartition
from sympy.sets.sets import FiniteSet

# where to put caching here?
# don't define __ge__ for now. Make sure __le__ works both ways.
# define __lt__ and make sure both partitions are different. (self.partition != other.partition)
# Discuss permutation class with Nicolas - Pretty cool but could be part of sympy as is (I overwrite their operators)
# haar_integral_permutation TypeError
# haar_integral_permutation ValueError
# haar_integral_centered_permutation TypeError
# haar_integral_centered_permutation ValueError
# raise error in both weingarten if partition elements are not unique
# add crossing partition method (sort(sort(input))) - test is done in montrealer
# "The partitions must be composed of unique elements" for all functions in partition
# Add typing
# I believe my < overwrite < method!
# All functions of Permutation are isomorphic (use Union) if Sequence, turn into Partition with *partition
# if isinstance(partition1, Sequence) and isinstance(partition1, Sequence)
#   parition1 = Partition(*partition1)
#   parition2 = Partition(*partition2)
# elif not (isinstance(partion1, Partition) and isinstance(partion2, Partition)):
#   raise TypeError

class Partition(SympyPartition):
    """
    Custom subclass of Sympy's Partition class
    This class represents an abstract partition
    A partition is a set of disjoint sets whose union equals a given set
    See  sympy.utilities.iterables.partitions

    Attributes:
        partition (list[list[int]]): Return partition as a sorted list of lists
        members (tuple[int]): 
        size (int): size of the partition
    """

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
            raise ValueError('Integers 0 through %s must be present.' %
            (obj.size-1))
        
        return obj
    
    def __le__(self, other):
        """
        Checks if a partition is less than or equal to the other based on partial order

        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self <= other
        """
        #self.parition
        "value error if different size - Cannot compare Partition of different sizes."
        if not isinstance(other, Partition):
            raise TypeError
        
        return True
    
    def __lt__(self, other):
        return self.__le__(other) and self != other
    
    def __ge__(self, other):
        return other.__le__(self)
    
    def __gt__(self, other):
        return other.__lt__(self)

    
    #def __lt__(self, other):
    #    return self.__le__(other) and self != other

    #def __ge__(self, other):
        """
        
        Args:
            other (Partition): Partition to compare

        Returns:
            bool: True if self is of lower or equal partial order than other

        Raise:
            TypeError: if other is not an instance of Partition
            ValueError: if other and self are of different size
        """
    #    return other.__le__(self)
     
    def __and__(self, other):
        "meet operation & symbol"
        return
    
    def __or__(self, other):
        "join operation & symbol"
        return 

    def meet(self, other):
        return self.__and__(other)  

    def join(self, other):
        return self.__or__(other)
      
    def partition_order(self, other):
        return self.__le__(other)
    
    def is_crossing(self, other):
        return
    

#def set_partition(size: int) -> Generator[Partition, None, None]:
#    return multiset(range())