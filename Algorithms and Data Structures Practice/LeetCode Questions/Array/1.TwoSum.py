# DONE


'''
Given an array of integers, return indices of the two numbers such that they add 
up to a specific target.

You may assume that each input would have exactly one solution, and you may not 
use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].

'''


class Solution:
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        partial = {}
        
        for (idx, i) in enumerate(nums): 
            partial[target - i] = idx
            
        for (idx, i) in enumerate(nums):
            if(partial.get(i) is not None):
                index1 = partial.get(i);
                if(idx == index1):
                    continue
                else: 
                    return [index1, idx]
                