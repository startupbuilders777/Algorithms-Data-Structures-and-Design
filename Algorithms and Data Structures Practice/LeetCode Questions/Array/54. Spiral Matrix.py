'''
54. Spiral Matrix
Medium

1738

492

Add to List

Share
Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

Example 1:

Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:

Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]

'''

class Solution(object):
    
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        
        output = []
        if len(matrix) == 0:
            return []
        
        # rightborder, left, up, down
        def dfs(mat, i, j, rBorder, lBorder, uBorder, dBorder, direction):
            
            # print((i, j, rBorder, lBorder, uBorder, dBorder))
            
            # print("a", i <= uBorder)
            # print("b", i >= dBorder)
            # print("c", j >= rBorder)
            # print("d", j <= lBorder)
            if(i <= uBorder or 
               i >= dBorder or 
               j >= rBorder or 
               j <= lBorder):
                
                
                 return 
            
            output.append(mat[i][j])
            
            if direction == "r":
                if j+1 < rBorder:
                    dfs(mat, i, j+1, rBorder, lBorder, uBorder, dBorder, direction)
                else: # Go down, add border to up
                     dfs(mat, i+1, j, rBorder, lBorder, uBorder+1, dBorder, "d")
                    
            elif direction == "l":
                if(j-1 > lBorder): 
                    dfs(mat, i, j-1, rBorder, lBorder, uBorder, dBorder, direction)
                else: # Go up
                    dfs(mat, i-1, j, rBorder, lBorder, uBorder, dBorder-1, "u")
            
            elif direction == "u":
                if(i-1 > uBorder):
                    dfs(mat, i-1, j, rBorder, lBorder, uBorder, dBorder, direction)
                else:
                    dfs(mat, i, j+1, rBorder, lBorder+1, uBorder, dBorder, "r")

            
            else:
                if i+1 < dBorder:
                    dfs(mat, i+1, j, rBorder, lBorder, uBorder, dBorder, direction)
                else:
                    dfs(mat, i, j-1, rBorder-1, lBorder, uBorder, dBorder, "l")

                    
        dfs(matrix, 0,0, len(matrix[0]), -1, -1, len(matrix), "r")
        # print(output)
        return output