'''

Given two strings s and t , write a function to determine if t is an anagram of s.

Example 1:

Input: s = "anagram", t = "nagaram"
Output: true
Example 2:

Input: s = "rat", t = "car"
Output: false
Note:
You may assume the string contains only lowercase alphabets.

Follow up:
What if the inputs contain unicode characters? How would you adapt your solution to such case?



One more solution.
I'm using XOR in combination with a simple hash function.
XOR ensures that every letter has a duplicate.
a ^ a === 0
a ^ b !== 0
But it will fail in a case when s == 'aabb' and t == 'ccdd'.
That's why there is also a simple hash function checking if these two strings are similar.

var isAnagram = function(s, t) {
    const len = s.length;
    if (len !== t.length) return false;
    
    let x = 0, h = 0;
    for (let i = 0; i < len; i++) { 
        const _1 = s.charCodeAt(i);
        const _2 = t.charCodeAt(i);
        x ^= _1 ^ _2;
        h += _1 * (_1 % 26) - _2 * (_2 % 26);
    }
    return h === 0 && x === 0;
};
'''

class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        result = 0
        h = 0
        
        if(len(s) != len(t)):
            return False
        
        for i in range(0, len(s)):
            a = ord(s[i])
            b = ord(t[i])
            result ^= a
            result ^= b
            
            h += a * (a % 26 ) - b * ( b % 26 ) 
            
        if(result == 0 and h == 0):
            return True
        else:
            return False



