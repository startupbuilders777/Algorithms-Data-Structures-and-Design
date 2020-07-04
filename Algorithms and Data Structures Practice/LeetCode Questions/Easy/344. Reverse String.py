'''
Write a function that takes a string as input and returns the string reversed.

Example:
Given s = "hello", return "olleh".

'''

class Solution(object):
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]
    
class Solution:
    def reverseString(self, s: List[str]) -> None:
        """
        Do not return anything, modify s in-place instead.
        """
        if s == "":
            return ""
        
        i = 0
        j = len(s) - 1
        
        while i < j:
            s[i], s[j] = s[j], s[i]
            j -= 1
            i += 1
        
        return s
        