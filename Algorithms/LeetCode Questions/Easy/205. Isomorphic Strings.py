'''
Given two strings s and t, determine if they are isomorphic.

Two strings are isomorphic if the characters in s can be replaced to get t.

All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.

For example,
Given "egg", "add", return true.

Given "foo", "bar", return false.

Given "paper", "title", return true.

Note:
You may assume both s and t have the same length.
'''


class Solution(object):
    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """

        dic = {}
        setUsed = {}

        if (len(s) != len(t)):
            return False

        for i in range(0, len(s)):
            print(s[i])
            if (dic.get(s[i]) is None):
                print("SAVED")
                if (setUsed.get(t[i]) is None):
                    dic[s[i]] = t[i]
                    setUsed[t[i]] = 1
                else:
                    return False  # Double mapping
            elif (dic.get(s[i]) != t[i]):
                return False
        return True