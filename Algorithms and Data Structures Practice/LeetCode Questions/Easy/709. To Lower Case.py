
'''
Implement function ToLowerCase() that has a string parameter str, and returns the same string in lowercase.


'''

class Solution:
    def toLowerCase(self, str): 
        return "".join(chr(ord(c) + 32) if 65 <= ord(c) <= 90 else c for c in str)
class Solution:
    def toLowerCase(self, str): 
        return "".join(chr(ord(c) + 32) if "A" <= c <= "Z" else c for c in str)
class Solution:
    def toLowerCase(self, str): 
        return str.lower()
        