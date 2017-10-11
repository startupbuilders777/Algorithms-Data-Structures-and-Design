'''
Given two binary strings, return their sum (also a binary string).

For example,
a = "11"
b = "1"
Return "100".
'''


class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """

        aInt = int(a, 2)
        bInt = int(b, 2)
        print(aInt)
        print(bInt)
        return bin(aInt + bInt)[2:]