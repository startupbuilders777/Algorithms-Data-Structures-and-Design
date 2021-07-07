'''
647. Palindromic Substrings
Medium

1753

88

Favorite

Share
Given a string, your task is to count how many 
palindromic substrings in this string.

The substrings with different start indexes or 
end indexes are counted as different substrings 
even they consist of same characters.

Example 1:

Input: "abc"
Output: 3
Explanation: Three palindromic strings: "a", "b", "c".
 

Example 2:

Input: "aaa"
Output: 6
Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
 

Note:

The input string length won't exceed 1000.
'''

class Solution(object):
    def countSubstrings(self, s):
        # Find all even and odd substrings. 
        '''
        THis is also known as the expand around center solution.        
        '''
        
        i = 0
        count = 0
        for i in range(len(s)):            
            left = i
            right = i
            
            # Count odd palins
            def extend(left, right, s):
                count = 0
                while True:
                    if left < 0 or right >= len(s) or s[left] != s[right]:
                        break   
                    count += 1
                    left = left - 1
                    right = right + 1
                return count
            
            count += extend(left, right, s)
            count += extend(left, right+1, s)
          
        return count
        
        
        
        
        