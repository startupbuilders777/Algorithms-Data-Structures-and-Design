'''
Given a string and an integer k, you need to reverse the first k characters for every 2k characters counting from the start of the string. If there are less than k characters left, reverse all of them. If there are less than 2k but greater than or equal to k characters, then reverse the first k characters and left the other as original.
Example:
Input: s = "abcdefg", k = 2
Output: "bacdfeg"
Restrictions:
The string consists of lower English letters only.
Length of the given string and k will in the range [1, 10000]

'''


class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        i = 0
        j = k

        while True:
            if j <= len(s):
                start = s[:i]
                middle = s[i: j]
                end = s[j:]
                s = start + middle[::-1] + end
                i += 2 * k
                j += 2 * k
            elif i <= len(s):  # Revere everythign after i
                start = s[:i]
                middle = s[i:]
                s = start + middle[::-1]
                break
            else:
                break

        return s

#FASTER
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        i = 0
        ret = ""
        flag = True
        while i+k <= len(s):
            if flag:
                ret += s[i:i+k][::-1]
            else:
                ret += s[i:i+k]
            flag = not flag
            i += k
        if flag:
            ret += s[i:][::-1]
        else:
            ret += s[i:]
        return ret

#fasest SOLUTION


class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        s1=''
        i=0
        while i<len(s):
            if len(s)-i-1>=2*k:
                s1+=s[i:i+k][::-1]
                s1+=s[i+k:i+2*k]
                i=i+2*k
            if len(s)-i-1>=k and len(s)-i-1< 2*k:
                s1+=s[i:i+k][::-1]
                s1+=s[i+k:]
                break
            if len(s)-i-1<k:
                s1+=s[i::][::-1]
                break
        return s1