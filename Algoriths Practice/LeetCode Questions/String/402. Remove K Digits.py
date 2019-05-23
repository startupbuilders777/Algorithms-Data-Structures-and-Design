'''
Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.

Note:
The length of num is less than 10002 and will be ≥ k.
The given num does not contain any leading zero.
Example 1:

Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
Example 2:

Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
Example 3:

Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.

'''


class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """

        digits = list(str(num))

        # Greedily remove the largest digits from left to right
        # Go from left to right, remove it if the next digit is smaller than currDigit
        # When next digit is not smaller than currDigit, you cant remove it, so keep going. BUT PUSH IT ON THE STACK
        # Then keep going

        if (k >= len(num)):
            return "0"

        st = []
        for i in range(0, len(digits)):
            st.append(i)
            if (i + 1 == len(digits)):
                break

            while True:
                topOfStack = st[len(st) - 1]

                if (int(digits[topOfStack]) > int(digits[i + 1]) and k != 0):
                    st.pop()
                    digits[topOfStack] = ""
                    k -= 1
                    if (len(st) == 0):
                        break
                else:
                    break

        # Now the string is in increasing order, any k leftover should be removed from the end of the string

        counter = len(digits) - 1
        while k > 0:
            if digits[counter] == "":
                continue
            else:
                digits[counter] = "";
                counter -= 1
                k -= 1

        # Remove useless 0s from the front
        print(digits)
        for i in range(0, len(digits)):
            if (digits[i] == '0' or digits[i] == ""):
                digits[i] = ""
            else:
                break

        result = "".join(digits)

        if (result == ""):
            return "0"

        return result





class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        stack = []
        for i in num:
            while k and stack and stack[-1] > i:
                stack.pop(-1)
                k -= 1
            stack.append(i)
        while k and stack:
            stack.pop(-1)
            k -= 1
        while stack and stack[0] == '0':
            stack.pop(0)
        return ''.join(stack) if stack else '0'

class Solution(object):
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        out = []
        for d in num:
            while k and out and out[-1] > d:
                out.pop()
                k -= 1
            out.append(d)
        return ''.join(out[:-k or None]).lstrip('0') or '0'