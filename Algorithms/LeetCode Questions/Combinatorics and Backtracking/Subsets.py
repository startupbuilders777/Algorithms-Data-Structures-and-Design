'''

Given a set of distinct integers, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

For example,
If nums = [1,2,3], a solution is:

[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]


'''


class Solution:
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        numberOfSubsets = len(nums)
        subsets = []
        for i in range(0, 2 ** numberOfSubsets):
            bits = bin(i)
            subset = []
            for j in range(0, numberOfSubsets):
                if i >> j & 1:  # Check if the first bit is on, then check if second bit is on, then check third bit is on, and keep going
                    subset.append(nums[j])

            subsets.append(subset)

        return subsets
