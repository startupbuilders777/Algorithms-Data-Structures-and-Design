'''
Related to question Excel Sheet Column Title

Given a column title as appear in an Excel sheet, return its corresponding column number.

For example:

    A -> 1
    B -> 2
    C -> 3
    ...
    Z -> 26
    AA -> 27
    AB -> 28 

'''


class Solution(object):
    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        # Base 26 system:
        # AA -> 1 * 26 ^1 + 1 * 26^0


        dict = {}
        minCharASCIIValueMinus1 = ord("A") - 1

        for i in range(ord("A"), ord("Z") + 1):  # Go from A to Z
            dict[chr(i)] = i - minCharASCIIValueMinus1

        value = 0
        s = s[::-1]
        for i in range(0, len(s)):
            value += dict[s[i]] * (26 ** i)

        return value