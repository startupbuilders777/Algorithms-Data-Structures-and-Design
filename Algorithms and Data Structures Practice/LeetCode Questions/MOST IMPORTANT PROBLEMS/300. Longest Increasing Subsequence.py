'''
300. Longest Increasing Subsequence
Medium

4302

102

Add to List

Share
Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:

Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 
Note:

There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.
Follow up: Could you improve it to O(n log n) time complexity?

Accepted
'''

import bisect

class Solution(object):
    def lengthOfLIS(self, nums):
        if(len(nums) == 0):
            return 0
        
        LIS = [1 for i in range(len(nums))]
        i = 0
        
        arr = sorted(enumerate(nums), key=lambda x: x[1] )
        print(arr)
        N = len(nums)
                    
        '''
        for nlogn performance, need to index the smallest largest element 
        required for each length of subsequnce and binsearch on it. 
        en
        '''
        
        # each index represents lenght of subsequence, and value is largest value. 
        # tails = [nums[p] if p == 0 else float("inf") for p in range(N)]
        tails = []
        while i < N:
            val = nums[i]
            idx = bisect.bisect_left(tails, val)  
            # idx is insertion point. 
            LIS[i] = idx + 1
            
            if(len(tails) == idx):
                tails.append(val)
            else:
                tails[idx + 1 - 1] = min(tails[idx + 1  - 1], val)
            i += 1
        
        return max(LIS)