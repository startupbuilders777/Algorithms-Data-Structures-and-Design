'''
Given an arbitrary ransom note string and another string containing letters from all the magazines, 
write a function that will return true if the ransom note can be constructed from the magazines ; 
otherwise, it will return false.

Each letter in the magazine string can only be used once in your ransom note.

Note:
You may assume that both strings contain only lowercase letters.

canConstruct("a", "b") -> false
canConstruct("aa", "ab") -> false
canConstruct("aa", "aab") -> true


'''


class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """

        count = {}
        for i in magazine:
            if (count.get(i) is None):
                count[i] = 1
            else:
                count[i] += 1

        for i in ransomNote:
            if (count.get(i) is None):
                return False
            else:
                count[i] -= 1
                if (count[i] == 0):
                    del count[i]

        return True

#REALLY FAST

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        return all(ransomNote.count(c) <= magazine.count(c) for c in set(ransomNote))

class Solution(object):
    def canConstruct(self, ransomNote, magazine):
        """
        :type ransomNote: str
        :type magazine: str
        :rtype: bool
        """
        for i in set(ransomNote):
            if ransomNote.count(i) > magazine.count(i):
                return False
        return True


