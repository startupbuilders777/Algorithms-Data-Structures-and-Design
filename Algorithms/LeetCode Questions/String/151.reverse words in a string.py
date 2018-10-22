#COMPLETED

'''

Given an input string, reverse the string word by word.

Example:  

Input: "the sky is blue",
Output: "blue is sky the".
Note:

A word is defined as a sequence of non-space characters.
Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
You need to reduce multiple spaces between two words to a single space in the reversed string.
Follow up: For C programmers, try to solve it in-place in O(1) space.

'''


class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        splitwords = s.split()
        return " ".join(splitwords[::-1])



# YOU CAN USE A DEQUE, SLIGHTLY FASTER:


class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        if not s:
            return s
        
        curr_word = None
        stack = collections.deque()
        
        for c in s:
            if c == " " and curr_word is not None:
                stack.append(curr_word)
                curr_word = None
            elif c != " ":
                if curr_word is None:
                    curr_word = c
                else:
                    curr_word += c
        if curr_word:
            stack.append(curr_word)
        result = ""
        while stack:
            if result != "":
                result += " "
            result += stack.pop()
        return result


# solution similar to yours but faster:

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        return " ".join([string for string in s.split(" ")[::-1] if string])


# EVEN FASTEr:

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        '''
       
The method strip() returns a copy of the string in which all chars 
have been stripped from the beginning and the end of the string (default whitespace characters).


        '''

        return " ".join(s.strip().split()[::-1])


# Fastest SOLUTION::

class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        # return ' '.join((s.strip().split())[::-1]) 
        return ' '.join(reversed(s.split()))