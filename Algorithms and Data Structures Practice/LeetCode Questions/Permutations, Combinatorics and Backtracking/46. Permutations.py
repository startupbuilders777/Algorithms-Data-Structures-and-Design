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

class Solution2:
    # DFS
    def permute(self, nums):
        res = []
        self.dfs(nums, [], res)
        return res
        
    def dfs(self, nums, path, res):
        if not nums:
            res.append(path)
            # return # backtracking
        for i in xrange(len(nums)):
            self.dfs(nums[:i]+nums[i+1:], path+[nums[i]], res)


# Solution 1: Recursive, take any number as first

# Take any number as the first number and append any permutation of the other numbers.

def permute(self, nums):
    return [[n] + p
            for i, n in enumerate(nums)
            for p in self.permute(nums[:i] + nums[i+1:])] or [[]]

#Solution 2: Recursive, insert first number anywhere

# Insert the first number anywhere in any permutation of the remaining numbers.

def permute(self, nums):
    return nums and [p[:i] + [nums[0]] + p[i:]
                     for p in self.permute(nums[1:])
                     for i in range(len(nums))] or [[]]

# Solution 3: Reduce, insert next number anywhere

# Use reduce to insert the next number anywhere in the already built permutations.

def permute(self, nums):
    return reduce(lambda P, n: [p[:i] + [n] + p[i:]
                                for p in P for i in range(len(p)+1)],
                  nums, [[]])

# Solution 4: Using the library

def permute(self, nums):
    return list(itertools.permutations(nums))

# That returns a list of tuples, but the OJ accepts it anyway. 
# If needed, I could easily turn it into a list of lists:

def permute(self, nums):
    return map(list, itertools.permutations(nums))



class Solution:
    def permute(self, nums):
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



