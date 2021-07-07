# DONE

'''
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
Example 2:

Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
Note:

All given inputs are in lowercase letters a-z.



'''



import functools

def lcp(s1, s2):
    for c1, c2 in zip(s1, s2):
        if c1 == c2:
            yield c1
        else:
            break

class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        return ''.join(functools.reduce(lcp, strs)) if strs else ''



class Solution2:
    # @return a string
    def longestCommonPrefix(self, strs):
        if not strs:
            return ""
            
        for i, letter_group in enumerate(zip(*strs)):
            if len(set(letter_group)) > 1:
                return strs[0][:i]
        else:
            return min(strs)


class Solution3:
        
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        def commonCharacters(str1, str2): 
            count = 0
            
            smallerLength = min(len(str1), len(str2))
            
            for i in range(0, smallerLength):
                if(str2[i] == str1[i]):
                    count += 1
                else: 
                    break
            
            return str1[0:count]
        
        
        if(len(strs) == 0):
            return ""
        
        currCommonString = strs[0]
        for i in range(1, len(strs)):
            currCommonString = commonCharacters(currCommonString, strs[i])
            
            
        return currCommonString




'''
SOLUTION SET:

Horizontal Scanning:

 public String longestCommonPrefix(String[] strs) {
    if (strs.length == 0) return "";
    String prefix = strs[0];
    for (int i = 1; i < strs.length; i++)
        while (strs[i].indexOf(prefix) != 0) {
            prefix = prefix.substring(0, prefix.length() - 1);
            if (prefix.isEmpty()) return "";
        }        
    return prefix;
}


Approach 2: Vertical scanning

public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0) return "";
    for (int i = 0; i < strs[0].length() ; i++){
        char c = strs[0].charAt(i);
        for (int j = 1; j < strs.length; j ++) {
            if (i == strs[j].length() || strs[j].charAt(i) != c)
                return strs[0].substring(0, i);             
        }
    }
    return strs[0];
}

Even though the worst case is still the same as Approach 1, in the best case there are at most 
n*minLenn comparisons where minLenminLen is the length of the shortest string in the array.



Use a Trie




'''


