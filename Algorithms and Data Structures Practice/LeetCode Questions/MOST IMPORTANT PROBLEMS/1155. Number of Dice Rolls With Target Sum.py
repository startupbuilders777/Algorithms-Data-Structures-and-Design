'''
1155. Number of Dice Rolls With Target Sum
Medium

661

36

Add to List

Share
You have d dice, and each die has f faces numbered 1, 2, ..., f.

Return the number of possible ways (out of fd total ways) modulo 10^9 + 7 to roll the dice so the sum of the face up numbers equals target.

 

Example 1:

Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.
Example 2:

Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.
Example 3:

Input: d = 2, f = 5, target = 10
Output: 1
Explanation: 
You throw two dice, each with 5 faces.  There is only one way to get a sum of 10: 5+5.
Example 4:

Input: d = 1, f = 2, target = 3
Output: 0
Explanation: 
You throw one die with 2 faces.  There is no way to get a sum of 3.
Example 5:

Input: d = 30, f = 30, target = 500
Output: 222616187
Explanation: 
The answer must be returned modulo 10^9 + 7.
 

Constraints:

1 <= d, f <= 30
1 <= target <= 1000

'''

class Solution:
    def numRollsToTarget(self, d: int, f: int, target: int) -> int:
        '''
        Cant define the backward way to hard?
        
        OPT[target] -> count ways to reach target. 
            OPT[target - 1 ] + OPT[target-2] + ... + OPT[target - f]
        
        IMPORTANT RULE -> 
        
        The only rolls that count are the ones after roll d is completed!
        That is why we need a 2d array, or a 1d array and we always carry the
        previous row as a space optimization TO BE USED FOR THE NEXT ROW. 
        
        '''
        COUNT = [0 for i in range(target+1)]  
        COUNT[0] = 1
        PRE_COUNT = COUNT
        
        for roll in range(d):
            COUNT = [0 for i in range(target+1)]
            for i in range(1, target+1): 
                for face in range(f):
                    if i - (face+1) >= 0:
                        COUNT[i] += (PRE_COUNT[i - (face+1)] )
            PRE_COUNT = COUNT
        return (COUNT[target] % (10**9 + 7) )
        
    def numRollsToTargetTopDown(self, d: int, f: int, target: int) -> int:
        # Accepted Top Down Solution:
        @lru_cache(maxsize=None)
        def dp(i, leftover):
            # if leftover < 0:
            if i == d and leftover == 0:
                # reached end!
                # you have to roll all dice. 
                return 1 # one way to do it. 
            
            elif i == d:
                return 0
            
            count = 0
             
            for face in range(f):
                if leftover >= (face+1):
                    count += dp(i+1, leftover - (face+1) )
            return count
        
        return (dp(0, target) % (10**9 + 7))
