# COMPLETED SOLUTION

'''
You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, 
the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and 
it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of 
money you can rob tonight without alerting the police.

Credits:
Special thanks to @ifanchu for adding this problem and creating all test cases. Also thanks to @ts for adding 
additional test cases.
'''

# MY DP SOLUTION
class Solution:
    def rob(self, nums: List[int]) -> int:      
        '''
        Transition function: 
        
        FREE state + ROB action -> FROZEN 
        Free state + DONT ROB -> Free State
        FROZEN state + Dont Rob -> Free State.  
        '''
        
        COUNT = len(nums)
        
        FROZEN = 0
        FREE = 0 
        
        NXT_FROZEN = 0
        NXT_FREE = 0
        
        for val in nums:
            # Rob
            # OPT[i+1] = max(OPT[i+1], OPT[i] + nums[i]) 
            NXT_FROZEN = FREE  + val
            NXT_FREE = max(FREE, FROZEN)
            
            FROZEN = NXT_FROZEN
            FREE = NXT_FREE
        
        return max(FREE, FROZEN)


class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        maxRobDict = {}  # Save the max you can earn at an array index

        def robMax(nums, index, maxRobDict, prevBrokenInto):
            if (index == len(nums)):
                return 0
            elif prevBrokenInto:
                return robMax(nums, index + 1, maxRobDict, False)
            else:
                if (maxRobDict.get(index) is None):
                    maxRobDict[index] = max(nums[index] + robMax(nums, index + 1, maxRobDict, True),
                                            robMax(nums, index + 1, maxRobDict, False))

                return maxRobDict[index]

        return robMax(nums, 0, maxRobDict, False)

##FASTER PRRACTICE THE BOTTOM UP BRO

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ppre, pre = 0, 0
        for i in range(len(nums)):
            ppre, pre = pre, max(ppre + nums[i], pre)
        return pre


#FASTESTTTTTT

class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0
        if len(nums) < 2:
            return nums[0]

        dp = [0] * len(nums)
        dp[1] = nums[0]
        res = dp[1]
        for i in range(2, len(nums)):
            dp[i] = max(dp[i - 2] + nums[i - 1], dp[i - 1])
            res = max(res, dp[i])
        res = max(res, dp[-2] + nums[-1])
        return res