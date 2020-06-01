'''
91. Decode Ways
Medium

2447

2657

Add to List

Share
A message containing letters from A-Z is being 
encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26
Given a non-empty string containing only digits, 
determine the total number of ways to decode it.

Example 1:

Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).
Example 2:

Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
'''

# MY SOLUTION SUPER SLOW:

class Solution:
    def numDecodings(self, s):        
        
        m = {}
        def waysDP(idx):
            nonlocal m 
            if(m.get(idx)):
                return m[idx]
            
            if idx == len(s):
                return 1
            
            A = 0
            val1 = int(s[idx])
            if(val1 != 0):
                A = waysDP(idx + 1)
            
            
            # take 2 chars and progress twice
            B = 0     
    
            val2 = None
            
            if(idx + 1 < len(s)):
                val2 = int(s[idx + 1])
            
            if(val2 != None and (val1 == 1 or (val1 == 2 and val2 < 7) )):
                # print("decoded,", (val1, val2))
                B = waysDP(idx + 2)
            
            m[idx] = A + B
            return m[idx]
        
        return waysDP(0)

# BOTTOM UP WAY:

class Solution:
    def numDecodings(self, s: str) -> int:
        if len(s) == 0: return 0
        dp = [0]*(len(s)+1)
        dp[0] = 1
        dp[1] = 1 if s[0] != '0' else 0
        for i in range(2, len(dp)):
            # single
            if '1' <= s[i-1] <= '9':
                dp[i] = dp[i-1]
            # double
            if '10' <= s[i-2:i] <= '26':
                dp[i] += dp[i-2]
        return dp[-1]

# Use two variables for O(1) space:

class Solution:
    def numDecodings(self, s: str) -> int:
        if len(s) == 0: return 0
        single = 1 if s[0] != '0' else 0
        double = 0
        for i in range(1, len(s)):
            oldSingle = single
            single = 0 if s[i] == '0' else (single + double)
            double = oldSingle if '10' <= s[i-1:i+1] <= '26' else 0
        return single + double