# DONE
'''
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
Seen this question in a real interview before?  YesNo

'''

# My accepted solution

class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        # ok go from left to right, start adding characters, and keeping a set of all character currently in the substring => actually a map
        # from character to index location in string. 
        # if you find a character that is in the current substring, delete characters from the front until that character is gone. 
        # determing how many to remove using the map, and the index of the found character.
        # keep track of the longest string and max characters. 
        m = {}
        
        currMaxSubStr=""
        
        currStr = ""
        startOfCurStr = 0
        endOfCurStr = 0
        
        
        
        for i, c in enumerate(s):
            if(m.get(c) is None):
                m[c] = i
                currStr += c
                endOfCurStr += 1
                
            else: #Not a max, take out right side and put in this char
                # store current version of longest substring
            
                lastSeenAt = m[c]
                # print("lastSeenAt", lastSeenAt)
                # print("bad char", c)
                # print("currStr", currStr)
                # print("startOfCurStr", startOfCurStr)
                # print("endOfCurStr", endOfCurStr)
              
                if(len(currStr) > len(currMaxSubStr)):
                    currMaxSubStr = currStr
                # print("m before deleting bad char", m)
                for k in range(lastSeenAt - startOfCurStr + 1):
                    del m[currStr[k]] # delete popped characters from map
                
                # print("m after deleteing bad char", m)
                
                endOfCurStr += 1 #add that char
                currStr = s[lastSeenAt+1 : endOfCurStr]
                m[c] = i #add new char
                startOfCurStr = lastSeenAt + 1
                # print("curr str after, ", currStr)
                # print()
                
        # check if the final string is bigger than max:
        if(len(currStr) > len(currMaxSubStr)):
            currMaxSubStr = currStr
        
                
            
        # print(currMaxSubStr)
        return len(currMaxSubStr)

# LEET CODE SOLUTIONS:
# BRUTE FORCE 
'''

Suppose we have a function boolean allUnique(String substring) which will return true if 
the characters in the substring are all unique, otherwise false. We can iterate through 
all the possible substrings of the given string s and call the function allUnique. If it 
turns out to be true, then we update our answer of the maximum length of substring without duplicate characters.

'''

# FASTER SOLUTIONS THAN MY SLIDING WINDOW:


class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        start = 0
        dict_ = {}
        max_ = 0
        for i, ch in enumerate(s): #DP
            if ch in dict_:
                max_ = max(max_, i - start)
                start = max(start, dict_[ch]+1)
            
            
            dict_[ch] = i
        return max(max_, len(s)-start)


# Faster: 

class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        substr = ''
        l = 0
        for a in s:
            if a not in substr:
                substr += a
            else:
                substr = substr.split(a)[1] + a
            if l < len(substr):
                    l += 1
        return l


# a little bit faster than above:

class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        dic = {}
        if len(s)<1:return 0
        if len(s)==1:return 1
        s_max = 1
        tmp = ''
        for i in s:
            if i in tmp:
                if len(tmp)>s_max:s_max = len(tmp)+1
                tmp = tmp[tmp.index(i)+1:]+i
            else:
                tmp+=i
                if len(tmp)>s_max:s_max = len(tmp)
        return s_max
        

# SUPER FAST:


class Solution:
    def lengthOfLongestSubstring(self, str):
        """
        :type s: str
        :rtype: int
        """
        if len(str) == 0: return 0
        map = {}
        l, h = 0, 0
        map[str[0]] = 0
        result = 1
        for i, c in enumerate(str[1:], 1):
            if c not in map:
                map[c] = i
                h = i
            else:
                # c in map
                j = map[c]
                if j >= l and j <= h:
                    l = j + 1
                
                map[c] = i
                h = i 
            temp = h - l + 1
            if temp > result: result = temp               
        return result
