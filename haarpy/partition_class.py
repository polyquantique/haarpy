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

# Discuss permutation class with Nicolas
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
    This class represents an abstract partition.
    A partition is a set of disjoint sets whose union equals a given set.
    See  sympy.utilities.iterables.partitions
    """
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, **kwargs)


    def __le__(self, other):
        "partition order"
        #self.parition
        return


    def __ge__(self, other):
        return other.__le__(self)
    
    
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
    

    def iscrossing(self, other):
        return
    
    