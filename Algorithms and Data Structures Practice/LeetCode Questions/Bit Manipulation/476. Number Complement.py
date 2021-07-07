'''
Given a positive integer, output its complement number. The complement strategy is to flip the bits of its binary representation.

Note:
The given integer is guaranteed to fit within the range of a 32-bit signed integer.
You could assume no leading zero bit in the integer’s binary representation.
Example 1:
Input: 5
Output: 2
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
Example 2:
Input: 1
Output: 0
Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.
'''

class Solution(object):
    def findComplement(self, num):
        """
        :type num: int
        :rtype: int
        """
        '''
        Complement fo 101 is 010

        101^111 -> 010
        So xor with all 1's

        You can convert between a string representation of the binary using bin() and int()

        >>> bin(88)
        '0b1011000'
        >>> int('0b1011000', 2)
        88
        >>>

        >>> a=int('01100000', 2)
        >>> b=int('00100110', 2)
        >>> bin(a & b)
        '0b100000'
        >>> bin(a | b)
        '0b1100110'
        >>> bin(a ^ b)
        '0b1000110'

        '''
        strBin = bin(num)
        print(strBin)
        allOnesBin = int(len(strBin[2:])*"1", 2)
        return num^allOnesBin

