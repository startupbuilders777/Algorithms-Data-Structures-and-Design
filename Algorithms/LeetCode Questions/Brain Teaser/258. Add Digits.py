'''
Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

For example:

Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.

Follow up:
Could you do it without any loop/recursion in O(1) runtime?

Credits:
Special thanks to @jianchao.li.fighter for adding this problem and creating all test cases.
'''


class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """

        # Every number must map to one of the following: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        # 11 -> 2
        # 12 -> 3
        # 13 -> 4
        # 9 is -> 1001 -> sum is 9
        # 10 is -> 1010 -> conv to 1
        # 11 is -> 1011 ->
        # 12 is -> 1100 -> 3

        # 41 -> keep subtracting 9 and youll get -> 9 + 9 + 9 + 9 + 5 so it maps to 5!!!!! THATS THE ALGO!!!!!
        if (num == 0):
            return 0
        elif (num % 9 == 0):
            return 9
        else:
            return num % 9

