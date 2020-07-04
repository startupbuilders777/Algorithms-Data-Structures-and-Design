'''
Given a collection of numbers that might contain duplicates, 
return all possible unique permutations.

For example,
[1,1,2] have the following unique permutations:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]

'''


class Solution: #THIS SOLUTION DOESNT WORK BECAUSE LISTS ARE UNHASHABLE BECAUSE THEY ARE MUTABLE, DO IN SOME OTHER WAY?
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]

        Put them in a set to get unique permutations!
        """

        if (len(nums) == 1):
            return [nums]

        allPermutations = set()

        for i in range(0, len(nums)):
            val = nums[i]
            permutations = self.permuteUnique(nums[:i] + nums[i + 1:])

            for j in range(0, len(permutations)):
                allPermutations.add([val] + permutations[j])

        return list(allPermutations)


'''
WAY BETTER AND FASTER SOLUTION
'''


class Solution:
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]

        Put them in a set to get unique permutations!

        OR

        track in a map if a digit has been at position i and if it has, dont do the 
        recursive step and skip over it, otherwises do the
        recursion!

        """

        if (len(nums) == 1):
            return [nums]

        allPermutations = []
        seenDigit = set()

        for i in range(0, len(nums)):
            val = nums[i]

            if (val in seenDigit):
                continue

            seenDigit.add(val)

            permutations = self.permuteUnique(nums[:i] + nums[i + 1:])

            for j in range(0, len(permutations)):
                allPermutations.append([val] + permutations[j])

        return allPermutations


class Solution:
    # another one
    
    def permuteUnique(self, nums):
        def backtrack(tmp, size):
        if len(tmp) == size:
            ans.append(tmp[:])
        else:
            for i in range(size):
                if visited[i] or (i > 0 and nums[i-1] == nums[i] and not visited[i-1]):
                    continue
                visited[i] = True
                tmp.append(nums[i])
                backtrack(tmp, size)
                tmp.pop()
                visited[i] = False
    ans = []
    visited = [False] * len(nums)
    nums.sort()
    backtrack([], len(nums))
    return ans
