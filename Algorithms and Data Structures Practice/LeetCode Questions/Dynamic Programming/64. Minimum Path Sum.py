'''

64. Minimum Path Sum
Medium



Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:

Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.


'''

class Solution(object):
    def minPathSumTopDown(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        
        
        '''
       
        TOP DOWN MEMOIZATION SOLUTION
        '''
        m = {}
        endi = len(grid)
        
        # i, j is our current spot.
        # go left, gor down. take min
        def minPathSum(i, j):
            if(m.get( (i, j) )):
                return m.get( ( i, j) )
                
            cost = grid[i][j]
            
            if(i == len(grid)-1 and j == len(grid[0]) - 1):
                return cost
            elif(i == len(grid) - 1):
                # go down
                m[ (i,j)  ] = cost + minPathSum(i, j + 1) 
            
            elif(j == len(grid[0]) - 1):
                m[ (i,j)  ] = cost + minPathSum(i+1, j)
            
            else:
                 m[ (i,j)  ] =  cost + min(minPathSum(i+1, j) , minPathSum(i, j+1))
                                                                           
            return m[(i,j)]
                    
        return minPathSum(0,0)
        '''
        BOTTOM UP:
        
        
        Base Cases:
        
        min cost of (0,0) is itself,
        min cost of top row is cumaltive sum
        min cost of left column is cumulative sum
        get min steps to block (i, j) using prev blocks!
        
        
        
        
        '''
    def minPathSum(self, grid):
        m = len(grid)
        n = len(grid[0])
        for i in range(1, n):
            grid[0][i] += grid[0][i-1]
        for i in range(1, m):
            grid[i][0] += grid[i-1][0]
        for i in range(1, m):
            for j in range(1, n):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]

        # O(m*n) space
    def minPathSum(self, grid):
        if not grid:
            return 
        r, c = len(grid), len(grid[0])
        dp = [[0 for _ in xrange(c)] for _ in xrange(r)]
        dp[0][0] = grid[0][0]
        for i in xrange(1, r):
            dp[i][0] = dp[i-1][0] + grid[i][0]
        for i in xrange(1, c):
            dp[0][i] = dp[0][i-1] + grid[0][i]
        for i in xrange(1, len(grid)):
            for j in xrange(1, len(grid[0])):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
        return dp[-1][-1]

    # O(2*n) space
    def minPathSum2(self, grid):
        if not grid:
            return 
        r, c = len(grid), len(grid[0])
        pre = cur = [0] * c
        pre[0] = grid[0][0] 
        for i in xrange(1, c):
            pre[i] = pre[i-1] + grid[0][i]
        for i in xrange(1, r):
            cur[0] = pre[0] + grid[i][0]
            for j in xrange(1, c):
                cur[j] = min(cur[j-1], pre[j]) + grid[i][j]
            pre = cur
        return cur[-1]

    # O(n) space
    def minPathSum(self, grid):
        if not grid:
            return 
        r, c = len(grid), len(grid[0])
        cur = [0] * c
        cur[0] = grid[0][0] 
        for i in xrange(1, c):
            cur[i] = cur[i-1] + grid[0][i]
        for i in xrange(1, r):
            cur[0] += grid[i][0]
            for j in xrange(1, c):
                cur[j] = min(cur[j-1], cur[j]) + grid[i][j]
        return cur[-1]

    # change the grid itself  
    def minPathSum4(self, grid):
        if not grid:
            return 
        r, c = len(grid), len(grid[0])
        for i in xrange(1, c):
            grid[0][i] += grid[0][i-1]
        for i in xrange(1, r):
            grid[i][0] += grid[i-1][0]
        for i in xrange(1, r):
            for j in xrange(1, c):
                grid[i][j] += min(grid[i-1][j], grid[i][j-1])
        return grid[-1][-1]    

