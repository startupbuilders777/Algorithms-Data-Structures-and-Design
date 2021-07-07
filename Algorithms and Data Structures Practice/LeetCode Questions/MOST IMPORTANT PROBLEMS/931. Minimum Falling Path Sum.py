'''
931. Minimum Falling Path Sum
Medium

726

61

Add to List

Share
Given a square array of integers A, we want the minimum sum of a falling path through A.

A falling path starts at any element in the first row, and chooses one element from each row.  The next row's choice must be in a column that is different from the previous row's column by at most one.

 

Example 1:

Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: 12
Explanation: 
The possible falling paths are:
[1,4,7], [1,4,8], [1,5,7], [1,5,8], [1,5,9]
[2,4,7], [2,4,8], [2,5,7], [2,5,8], [2,5,9], [2,6,8], [2,6,9]
[3,5,7], [3,5,8], [3,5,9], [3,6,8], [3,6,9]
The falling path with the smallest sum is [1,4,7], so the answer is 12.

 

Constraints:

1 <= A.length == A[0].length <= 100
-100 <= A[i][j] <= 100

'''


class Solution:
    # BACKWARD DYNAMIC PROGRAMMING
    def minFallingPathSumBackwardDP(self, A: List[List[int]]) -> int:
        '''
        Just save the min falling path 
        
        (i, j) -> minimum value
        start from bottom row as base case, build next layer above based on bottom row. 
        
        '''
        ROWS = len(A)
        COLS = len(A[0])
        
        grid = [[0 for _ in range(COLS)] for _ in range(2)]
        for i in range(ROWS-1, -1, -1):
            for j in range(COLS):
                if i == ROWS-1:
                    grid[i&1][j] = A[i][j]
                else:
                    # + A[i][j]
                    grid[i&1][j] = grid[(i+1)&1][j] 
                    if j + 1 < COLS:
                        grid[i&1][j] = min(grid[i&1][j], grid[(i+1)&1][j+1])
                    if j-1 >= 0:
                        grid[i&1][j] = min(grid[i&1][j], grid[(i+1)&1][j-1])
                    
                    grid[i&1][j] += A[i][j]
                    
        return min(grid[0])
        
        
    # FORWARD DYNAMIC PROGRAMMING
    # CAN ALSO SPACE OPTIMIZE THIS LIKE ABOVE
    def minFallingPathSum(self, A: List[List[int]]) -> int:
        '''
        Just save the min falling path 
        
        (i, j) -> minimum value
        start from bottom row as base case, build next layer above based on bottom row. 
        
        '''
        ROWS = len(A)
        COLS = len(A[0])
        
        # RELAX INFINITY!
        grid = [[float("inf") for i in range(len(A[0]))] for j in range(len(A))]
        
        # BASE CASE
        grid[0] = A[0]
        
        for i in range(ROWS-1):
            for j in range(COLS):
                
                # relax the 3 columns in the next row given us
                
                grid[i+1][j] = min(grid[i+1][j], grid[i][j] + A[i+1][j])
                
                if j + 1 < COLS:
                    grid[i+1][j+1] = min(grid[i+1][j+1], grid[i][j] + A[i+1][j+1])
                
                if j-1 >= 0:
                    grid[i+1][j-1] = min(grid[i+1][j-1], grid[i][j] + A[i+1][j-1])
    
        return min(grid[-1]) 
    
# Another cool soln:

def minFallingPathSum(self, A):
    """
    :type A: List[List[int]]
    :rtype: int
    """
    dp = A[0]
    for row in A[1:]:
        dp = [value + min([dp[c], dp[max(c - 1, 0)], dp[min(len(A) - 1, c + 1)]]) for c, value in enumerate(row)]
    return min(dp)
