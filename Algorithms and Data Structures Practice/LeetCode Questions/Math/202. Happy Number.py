'''
202. Happy Number
Easy

2052

417

Add to List

Share
Write an algorithm to determine if a number n is "happy".

A happy number is a number defined by the following process: 
Starting with any positive integer, replace the number by the sum of the 
squares of its digits, and repeat the process until the number equals 1 
(where it will stay), or it loops endlessly in a cycle which does not include 1. 
Those numbers for which this process ends in 1 are happy numbers.

Return True if n is a happy number, and False if not.

Example: 

Input: 19
Output: true
Explanation: 
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1

'''

class Solution:
    def isHappy(self, n: int) -> bool:
        '''
        
        21 -> 5 -> 25 -> 29 -> 85 -> 89 -> 159 -> 
        
        it equals 1 when 
        digits are either 1, or 0. 
        
        so 1, 10, 100, 1000, 1100
        
        
        We do cycle detection. 
        if we see a number in seen again, then return false.
        
        '''
        st = set()

        new_numb = 0
        while True:
            for i in str(n):
                new_numb += int(i)**2   
            
            if new_numb == 1:
                return True
            else:
                if new_numb in st:
                    break
                else:
                    st.add(new_numb)
                n = new_numb
                new_numb = 0
        
        return False        