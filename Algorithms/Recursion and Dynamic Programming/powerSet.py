'''Write a method to return all subsets of a set'''

#This is a set
A = {"a", "b", "c"}

# Result -> {{}, {a}, {b}, {c}, {a,b} {a,c} {b,c} {a,b,c}}
# what you can do is this trick => 3 elements in set => ok find 2^3 sets to get all of them its like this:
# 000, 001, 010, 011, 100, 101, 110, 111
# -> choose those elements to keep or not keep like that.


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
