'''
Divide two integers without using multiplication, division and mod operator.

If it is overflow, return MAX_INT.
'''

class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        quotient = 0
        dividend = abs(dividend)
        divisor = abs(divisor)

        while(dividend > divisor):
            dividend = dividend - divisor
            quotient += 1