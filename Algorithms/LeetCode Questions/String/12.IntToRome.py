# DONE

'''
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.

Example 1:

Input: 3
Output: "III"
Example 2:

Input: 4
Output: "IV"
Example 3:

Input: 9
Output: "IX"
Example 4:

Input: 58
Output: "LVIII"
Explanation: C = 100, L = 50, XXX = 30 and III = 3.
Example 5:

Input: 1994
Output: "MCMXCIV"
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

'''

# my solution:

class Solution:
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        # greedily take largest and subtract out of num, then repeat. 
        # How to deal with 4 => IV  
        # => GREEDILY TAKE THE FOLLOWING, TAKE 
        val = num
        roman = ""
        while(True):
            if(val == 0):
                break
            elif(val >= 1000):
                roman += "M"
                val -= 1000
            elif(val >= 900):
                roman += "CM"
                val -= 900
                
            elif(val >= 500):
                roman += "D"
                val -= 500
            elif(val >= 400):
                roman += "CD"
                
                val -= 400
                
            elif(val >= 100):
                roman += "C"
                
                val -= 100
            elif(val >= 90):
                roman += "XC"
                val -= 90
            elif(val >= 50):
                roman += "L"
                val -= 50
            elif(val >= 40):
                roman += "XL"
                val -= 40
            elif(val >= 10):
                roman += "X"
                val -= 10
            elif(val >= 9):
                roman += "IX"
                val -= 9
            elif(val >= 5):
                roman += "V"
                val -= 5
            elif(val >= 4):
                roman += "IV"
                val -= 4
            elif(val >= 1):
                roman += "I"
                val -= 1
                
        return roman

# EASIER WAY TO DO IT:

class Solution:
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        res = ""
        v = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
        roman = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        
        for i in range(len(v)):
            while num >= v[i]:
                num -= v[i]
                res += roman[i]
        return res

# A LIL BIT FASTER THAN ABOVE


class Solution:
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        t = {1:'I', 4:'IV', 5:'V', 9:'IX', 10:'X', 40:'XL', 50:'L', 90:'XC', 100:'C', 400:'CD', 500:'D', 900:'CM', 1000:'M'}
        res = ''
        index = len(str(num))
        while index > 0:
            b = 10**(index - 1)
            n = num // b
            if n == 4 or n == 9:
                res += t.get(n*b)
            elif n >= 5:
                res += t.get(5*b) + (n - 5)*t.get(b)
            else:
                res += n*t.get(b)
            num = num - (n * b)
            index -= 1
        return res
                