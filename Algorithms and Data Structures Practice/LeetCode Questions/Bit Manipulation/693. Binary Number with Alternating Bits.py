'''
Given a positive integer, check whether it has alternating bits: namely, 
if two adjacent bits will always have different values.

Example 1:
Input: 5
Output: True
Explanation:
The binary representation of 5 is: 101
Example 2:
Input: 7
Output: False
Explanation:
The binary representation of 7 is: 111.
Example 3:
Input: 11
Output: False
Explanation:
The binary representation of 11 is: 1011.
Example 4:
Input: 10
Output: True
Explanation:
The binary representation of 10 is: 1010.

'''


class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """

        '''
        Algo, the xor for adjacent bits in the binary represetntation
        will always yield 1 

        (BECAUSE XOR YIELDS 1 WHEN BITS ARE DIFF AHLIE BRO)

        maybe an easy way is to xor with its c


        '''

        bits = bin(n)[2:]  # the first 2 array indices is 0b(useless)
        print(bits)

        def xor(bit1, bit2):
            if (bit1 != bit2):
                return True
            else:
                return False

        i = 0
        j = 1
        while True:
            if j == len(bits):
                break
            if (not xor(bits[i], bits[j])):
                return False
            i += 1
            j += 1
            continue

        return True

#FASTER
class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        if n in (0, 1):
            return True

        def isAlign(n):
            while n > 0:
                if (n & 1) ^ ((n >> 1) & 1) == 0:
                    return False
                n >>= 1
            return True

        return isAlign(n)

## FASTEST SOLUTION YOU RETARD

class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        first = n%2
        n /= 2
        while n:
            if first == n%2:
                return False
            else:
                first = n%2
            n /=2
        return True