'''
643. Maximum Average Subarray I
Easy

780

121

Add to List

Share
Given an array consisting of n integers, find the contiguous subarray of given length k that has the maximum average value. And you need to output the maximum average value.

Example 1:

Input: [1,12,-5,-6,50,3], k = 4
Output: 12.75
Explanation: Maximum average is (12-5-6+50)/4 = 51/4 = 12.75
 

Note:

1 <= k <= n <= 30,000.
Elements of the given array will be in the range [-10,000, 10,000].
'''

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        
        
        if len(nums) < k:
            return sum(nums)/k
        
        s = 0
        for i in range(k):
            s += nums[i]
        
        m = s
        i = 0
        while i + k < len(nums):
            s += nums[i+k] - nums[i]
            m = max(s, m)
            i += 1 
        return m/k

# Using prefix sums (where sums[i] is the sum of the first i numbers) to compute subarray sums.

def findMaxAverage(self, nums, k):
    sums = [0] + list(itertools.accumulate(nums))
    return max(map(operator.sub, sums[k:], sums)) / k

# NumPy version (requires import numpy as np):

def findMaxAverage(self, nums, k):
    sums = np.cumsum([0] + nums)
    return int(max(sums[k:] - sums[:-k])) / k
