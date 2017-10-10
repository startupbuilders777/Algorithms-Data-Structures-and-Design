'''
Given a binary array, find the maximum number of consecutive 1s in this array.

Example 1:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.
Note:

The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000
'''


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        inConsec = False
        max = 0
        count = 0
        for i in nums:
            if i == 1 and inConsec == True:
                count += 1
                if count >= max:
                    max = count
            elif i == 1 and inConsec == False:
                inConsec = True
                count = 1
                if count >= max:
                    max = count
            elif i == 0:
                inConsec = False
                count = 0

        return max

## FASTER


class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_con, current_con = 0, 0
        for num in nums:
            if num == 1:
                current_con += 1
            else:
                if current_con > max_con:
                    max_con = current_con
                current_con = 0

        if current_con > max_con:
            max_con = current_con

        return max_con

## Fastest:

class Solution(object):
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        highest = -1
        count = 0
        for num in nums:
            if num == 1:
                count += 1
            else:
                if count > highest:
                    highest = count
                count = 0

        if count > highest:
            highest = count

        return highest
