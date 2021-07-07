'''
911. Maximum Size Subarray Sum Equals k
English
Given an array nums and a target value k, find the maximum length of 
a subarray that sums to k. If there isn't one, return 0 instead.

Example
Example1

Input:  nums = [1, -1, 5, -2, 3], k = 3
Output: 4
Explanation:
because the subarray [1, -1, 5, -2] sums to 3 and is the longest.
Example2

Input: nums = [-2, -1, 2, 1], k = 1
Output: 2
Explanation:
because the subarray [-1, 2] sums to 1 and is the longest.
Challenge
Can you do it in O(n) time?

Notice
The sum of the entire nums array is guaranteed to fit within the 32-bit signed integer range.


SIMILAR TO 
'''

# %100 DONE. 
class Solution:
    """
    @param nums: an array
    @param k: a target value
    @return: the maximum length of a subarray that sums to k
    """
    def maxSubArrayLen(self, nums, k):
        '''
        sums to k.
        
        ok create cumulative array. 
        Store 
        
        k - sum in map -> map to index.
        
        when you see it again, 
        since you process from right to left, 
        store 
        sumI - k in map. 
        
        sumJ - sumI = k
        sumI + k == sumJ
        we are trying to find sumJ which is computed as you move left. 
        '''
        res = 0
        m = {}
        cum = 0
        N = len(nums)
        m[k] = -1
        
        for idx, i in enumerate(nums):
            cum += i
            # print("idx, i, cum, M IS ", idx, i, cum,  m)
            
            if m.get(cum) is not None:
                # print("saw ", idx - m.get(cum) + 1)
                res = max(res, idx - m.get(cum))
            
            if m.get(cum + k) is None:
                # print("inserting kv", (cum+k, idx))
                m[cum+k] = idx # we dont overwrite if we have saw it previously!
                               # because we want to keep our ranges as wide as possible
                               # to be a possible max
        return res
