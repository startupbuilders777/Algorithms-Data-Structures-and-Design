# COMPLETED

'''
Given an array of strings, group anagrams together.

Example:

Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]
Note:

All inputs will be in lowercase.
The order of your output does not matter.

'''

class Solution:
        
    def groupAnagrams(self, strs):
        from collections import Counter
        import hashlib

        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        
        # how to do it fast
        # need to get character count for each string, 
        # then check if character count for that string matches character count of another string (can do a key look up for this)
        # if it does add to that list
        # if it doesnt match any in the list, then apppend as new item
        
        '''
        To check if 2 dictionaries are equal do this;
        
        >>> a = dict(one=1, two=2, three=3)
        >>> b = {'one': 1, 'two': 2, 'three': 3}
        >>> c = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
        >>> d = dict([('two', 2), ('one', 1), ('three', 3)])
        >>> e = dict({'three': 3, 'one': 1, 'two': 2})
        >>> a == b == c == d == e
        True
        '''
 
        
        m = {}
        for i in strs:
            yo = Counter(i)
            if(m.get(tuple(sorted(yo.items()) )) is None):
                m[tuple(sorted(yo.items()))] = [i]
            else:
                m.get(tuple(sorted(yo.items()))).append(i)
        
        return list(m.values())
        
        