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

# MY BOTTOM UP SOLUTION

class Solution:
    def numDecodings(self, s):        
        # BOTTOM UP ONLY!
        '''        
        ADD UP ALL THE WAYS WE ARRIVED TO A STATE FROM OTHER STATES!!
        USE IF STATEMENTS TO DETECT IF WE CAN ARRIVE TO THE STATE. 
        
        OPT[i] = OPT[i-1]   (TAKE ONE ALWAYS POSSIBLE!)
                 + OPT[i-2]   (TAKE 2 MAY NOT BE POSSIBLE)
         
        s = "12"
        
        "0" does not map to anything -> can only be used with 10 or 20
        
        but you can take 2 singles. 
        or take a double -> it should add to same array index. 
        does it matter if we go left to right or right to left? NO
        
        226
        2 -> 1  b
        22 -> 1 bb 
        2 26
        3 ways:
        2 2 6
        22 6
        2 26
        
        Base case empty string = 1?
        take 1 
        2 
        take 2:
        22 
        next timestamp?
        we take 1 
        
        OPT[i] -> all the ways to decode up to index i. 
        process index 0 -> only 1 way to decode unless its 0. 
        can take 2 charcters if OPT[i-1] exists. 
        
        In other words solution relies on 3 timesteps to build finite automata
        '''
        
        OPT = [0 for i in range(len(s) +1)]
        
        # BASE CASE
        OPT[0] = 1 
        prevCh = None
        seeNonezero = False
        
        # BTW its easy to optimize this to O(N) space, using 2 variables for 
        # previous 2 timestamps. 
        
        for idx in range(1, len(s)+1):
            # 0 cannot become anything!
            # take current character as single. 
            ch = int(s[idx-1])
            if ch != 0:
                OPT[idx] += OPT[idx-1]    
            # only way we can use 0 is if we see prev.                         
            # if you see 2 zeros in a row you cant decode it -> answer is 0. 
            if prevCh != None: 
                # take current character + prev char!
                if (prevCh == 1 and ch < 10) or (prevCh == 2 and ch < 7):
                    OPT[idx] += OPT[idx-2]
            # loop end set prevCharacter
            prevCh = ch            
            
        return OPT[len(s)]
            
            


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
    
    
'''
cleaner way:
'''

def mapDecoding(msg):
    a, b = 1, 0
    M = 10 ** 9 + 7
    for i in range(len(msg)-1, -1, -1):
        if msg[i] == "0":
            a, b = 0, a
        else:
            a, b = (a + (i+2 <= len(msg) and msg[i:i+2] <= "26") * b) % M, a
    return a