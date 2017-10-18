'''
Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3
Note:
You may assume that the array does not change.
There are many calls to sumRange function.

Would you need a 2D map
'''


class NumArray(object):
    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.nums = nums
        self.partialSums = {}
        sum = 0
        for i in range(0, len(nums)):
            sum += nums[i]
            print(sum)
            self.partialSums[i] = sum

    '''
    Save intervals in the map,
    and if a sum requires that interval, then do that + extra computation from there

    Map works like this:

    recursively call sumRange and lookups in map for intervals that exist

    map[start][end] -> Gives one range
    if start doesnt exist, try to do a lookup with startVal + sumrange(start + 1, end)
     (this will recursively check if that value is in the map)
    requires O(n^2) space probably
    '''

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.partialSums[j + 1] - self.partialSums[i]


        # Your NumArray object will be instantiated and called as such:
        # obj = NumArray(nums)
        # param_1 = obj.sumRange(i,j)