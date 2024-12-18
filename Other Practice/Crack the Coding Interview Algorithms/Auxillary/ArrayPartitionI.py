'''
Given an array of 2n integers, your task is to group these integers into n pairs of integer, 
say (a1, b1), (a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i 
from 1 to n as large as possible.

Example 1:
Input: [1,4,3,2]

Output: 4
Explanation: n is 2, and the maximum sum of pairs is 4 = min(1, 2) + min(3, 4).
Note:
n is a positive integer, which is in the range of [1, 10000].
All the integers in the array will be in the range of [-10000, 10000].
'''


class Solution(object):
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        nums.sort()

        # Adjacent sorted elements sum to the max sum
        element = None
        tuples = []
        first = None
        Second = None
        for i in nums:
            if first == None:
                first = i
                continue
            else:
                second = i
                tuples.append((first, second))
                first = None
                second = None
                continue

        # print(tuples)

        def sumMins(arrOfTuples):
            sum = 0
            for i in arrOfTuples:
                if (i[0] <= i[1]):
                    sum += i[0]
                else:
                    sum += i[1]
            return sum

        return sumMins(tuples)