'''
303. Range Sum Query - Immutable
Easy

691

927

Add to List

Share
Given an integer array nums, find the sum of the 
elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
Note:
You may assume that the array does not change.
There are many calls to sumRange function.
'''

class NumArray(object):
    
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        
        # You can use sqrt decomposition
        # You can use a segment tree
        # How about a map?
        
        # Just use a partial sum array
        self.partial = []
        a = 0
        for i in nums:
            a += i
            self.partial.append(a)
        
        print("partial sum", self.partial)
        
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        
        # So its the difference between sum up to J, minus 
        # sum up to but right before i (because partial i 
        # should be included!)!
        
        if(i == 0):
            return self.partial[j]
        else:      
            return self.partial[j] - self.partial[i-1]
        
        
        
        


# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)