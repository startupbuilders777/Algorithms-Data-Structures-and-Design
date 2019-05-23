'''
Given a collection of distinct numbers, return all possible permutations.

For example,
[1,2,3] have the following permutations:
[
  [1,2,3],
  [1,3,2],
  [2,1,3],
  [2,3,1],
  [3,1,2],
  [3,2,1]
]

'''


class Solution:
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        '''
        Its a general tree of choices ! 
        We can use DFS to go through all the possibilities or BFS
        '''

        if len(nums) == 1:
            return [nums]
        allPerms = []
        for i in range(0, len(nums)):
            value = nums[i]
            permutations = self.permute(nums[:i] + nums[i + 1:])

            for j in range(0, len(permutations)):
                print(permutations[j])
                permutations[j] = [value] + permutations[j]

            allPerms += permutations

        return allPerms



