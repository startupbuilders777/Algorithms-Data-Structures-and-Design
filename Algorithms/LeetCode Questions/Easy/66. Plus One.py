'''
Given a non-negative integer represented as a non-empty array of digits, plus one to the integer.

You may assume the integer do not contain any leading zero, except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.

'''


class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        reversedDigits = digits[::-1]
        carryIn = 1
        for i in range(0, len(reversedDigits)):
            sum = carryIn + reversedDigits[i]
            if (sum >= 10):
                reversedDigits[i] = sum % 10
                carryIn = sum / 10
                if (i == len(reversedDigits) - 1):
                    reversedDigits.append(carryIn)
            else:
                reversedDigits[i] = sum
                break

        return reversedDigits[::-1]


