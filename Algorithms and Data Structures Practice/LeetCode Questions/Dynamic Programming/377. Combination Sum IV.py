'''
377. Combination Sum IV
Medium

1332

171

Add to List

Share
Given an integer array with all positive numbers and no duplicates, 
find the number of possible combinations that add up to a positive integer target.

Example:

nums = [1, 2, 3]
target = 4

The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)

Note that different sequences are counted as different combinations.

Therefore the output is 7.
 

Follow up:
What if negative numbers are allowed in the given array?
How does it change the problem?
What limitation we need to add to the question to allow negative numbers?

'''

class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:        
        amounts = [0 for _ in range(target + 1)]
        amounts[0] = 1 # When you reach amount 0, yield 1, for the coin, base case
        
        for amt in range(1, target+1):
            for coin in nums: 
                if coin <= amt: 
                    amounts[amt] += amounts[amt-coin]
                    
        return amounts[target]
    


'''
The problem with negative numbers is that now the combinations could be potentially of infinite length. Think about nums = [-1, 1] and target = 1. We can have all sequences of arbitrary length that follow the patterns -1, 1, -1, 1, ..., -1, 1, 1 and 1, -1, 1, -1, ..., 1, -1, 1 (there are also others, of course, just to give an example). So we should limit the length of the combination sequence, so as to give a bound to the problem.

This is a recursive Python code that solves the above follow-up problem, so far it's passed all my test cases but comments are welcome.

class Solution(object):
    def combinationSum4WithLength(self, nums, target, length, memo=collections.defaultdict(int)):
        if length <= 0: return 0
        if length == 1: return 1 * (target in nums)
        if (target, length) not in memo: 
            for num in nums:
                memo[target, length] += self.combinationSum4(nums, target - num, length - 1)
        return memo[target, length]
'''