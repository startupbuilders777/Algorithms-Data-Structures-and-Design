'''
283. Move Zeroes
Easy

3816

124

Add to List

Share
Given an array nums, write a function to move all 0's to the end of it while 
maintaining the relative order of the non-zero elements.

Example:

Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
Note:

You must do this in-place without making a copy of the array.
Minimize the total number of operations.


'''

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        # just move 0s forward?
        # minimize total ops
        
        # one pointer points at a zero, other one points at nonzero, swap to front,
        # then find next zero, and next nonzero, swap. 
        
        zeroPtr = 0
        nonzeroPtr = 0
        
        while nonzeroPtr != len(nums) and zeroPtr != len(nums):
            # FIND NON ZEROS AFTER ZERO PTR!       
            if nums[zeroPtr] != 0:
                zeroPtr += 1
                nonzeroPtr = max(zeroPtr, nonzeroPtr)
            elif nums[nonzeroPtr] == 0:
                nonzeroPtr = max(zeroPtr, nonzeroPtr + 1)
            else: 
                nums[zeroPtr], nums[nonzeroPtr] = nums[nonzeroPtr], nums[zeroPtr]                
                zeroPtr += 1
                nonzeroPtr += 1

