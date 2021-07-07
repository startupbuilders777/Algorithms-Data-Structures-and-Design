'''
1223. Dice Roll Simulation
Medium

292

96

Add to List

Share
A die simulator generates a random number from 1 to 6 for each roll. 
You introduced a constraint to the generator such that it cannot roll 
the number i more than rollMax[i] (1-indexed) consecutive times. 

Given an array of integers rollMax and an integer n, return the number of 
distinct sequences that can be obtained with exact n rolls.

Two sequences are considered different if at least one element differs from 
each other. Since the answer may be too large, return it modulo 10^9 + 7.

 

Example 1:

Input: n = 2, rollMax = [1,1,2,2,2,3]
Output: 34
Explanation: There will be 2 rolls of die, if there are no constraints on the 
die, there are 6 * 6 = 36 possible combinations. In this case, looking at rollMax array, 
the numbers 1 and 2 appear at most once consecutively, therefore sequences (1,1) and (2,2) 
cannot occur, so the final answer is 36-2 = 34.


Example 2:

Input: n = 2, rollMax = [1,1,1,1,1,1]
Output: 30
Example 3:

Input: n = 3, rollMax = [1,1,1,2,2,3]
Output: 181

'''

# HARMANS BOTTOM UP SOLN

'''

First think of top down quickly, to determine how each state should be encoded. 
AND THE Base Case.

Determine if its perms question or combs question and deal with each scenerio seperately
by arraning for loops properly. 

Figure out how the old states relate to the new states. 
Figure out how the grid should be filled

SPACE OPTIMIZE AFTER!

'''

class Solution:        
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        
        ''' 
        You can save space by only keep the previous i and the current i, 
        and overwriting previous with current. 
        '''
        max_roll = max(rollMax)
        
        COUNT = [[[0 for _ in range(max_roll)] for _ in range(6)] for _ in range(n) ]
        
        COUNT[0][0][0] = 1
        COUNT[0][1][0] = 1
        COUNT[0][2][0] = 1
        COUNT[0][3][0] = 1
        COUNT[0][4][0] = 1
        COUNT[0][5][0] = 1
        
        for i in range(1, n):   
            for roll in range(6):
                for prev_roll in range(6):
                    for seq in range(max_roll):
                        if roll != prev_roll:
                            COUNT[i][roll][0] += (COUNT[i-1][prev_roll][seq] % (10 ** 9 + 7))
                        elif seq < rollMax[roll] and seq > 0:  
                            COUNT[i][roll][seq] = COUNT[i-1][roll][seq-1] % (10 ** 9 + 7) 
                            
        RESULT = sum(map(sum, COUNT[n-1]))
        
        return RESULT % (10 ** 9 + 7)


# HARMANS TOP DOWN SOLN
class Solution:
    def dieSimulator(self, n: int, rollMax: List[int]) -> int:
        '''
        rollMax, -> vals > n dont matter floor to n. 
        
        First roll can be anything, 
        second roll, choose whats available -> 
            keep recursively doing. 
            in right direction. 
        
        Or you can do the other way. 
        
        All possible ways - all illegal ways. 
        6^n - illegal. 
        '''    
        
        @lru_cache(None)
        def dfs(i, active, count):
            # do we need to know previous roll? yes
            if i == n:
                return 1
            
            tot = 0
            for c in range(6):
                # when you choose it, subtract one from it!
                if c == active and count < rollMax[c]:
                    tot += dfs(i + 1, c, count + 1)          
                elif c != active and rollMax[c] > 0:
                    tot += dfs(i + 1, c, 1) 
            return tot
        
        # theres is oh an element that isnt me thats active, 
        # there is oh 
        return dfs(0, None, 0) % (10**9 + 7)
    
