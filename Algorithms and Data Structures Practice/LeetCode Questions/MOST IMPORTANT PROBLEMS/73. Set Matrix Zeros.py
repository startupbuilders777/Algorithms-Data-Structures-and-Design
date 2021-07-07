'''
73. Set Matrix Zeroes
Medium

2081

299

Add to List

Share
Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

Example 1:

Input: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]
Example 2:

Input: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
Output: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]
Follow up:

A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
'''


# CONSTANT SPACE SOLUTION!
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        
        rows = len(matrix)
        if rows == 0:
            return matrix
        
        cols = len(matrix[0])
        
        # Treat the row and column we gather into,
        # DIFFERENT FROM THE OTHER ROWS AND COLS. 
        # DONT LOSE THE INFORMATION. 
        top_row_null = False
        left_col_null = False

        for c in range(cols):
            if matrix[0][c] == 0:
                # set top row null
                top_row_null = True
        
        for i in range(rows):
            if matrix[i][0] == 0:
                left_col_null = True

        for i in range(1, rows):
            for j in range(1, cols):
                if matrix[i][j] == 0:                    
                    matrix[0][j] = 0
                    matrix[i][0] = 0 
                    
        for c in range(1, cols):
            if matrix[0][c] == 0:
                for r in range(1, rows):
                    matrix[r][c] = 0
                
        for r in range(1, rows):
            if matrix[r][0] == 0:
                for c in range(1, cols):
                    matrix[r][c] = 0
                
        if(top_row_null):
            for c in range(cols):
                matrix[0][c] = 0
        
        if(left_col_null):
            for r in range(rows):
                matrix[r][0] = 0

'''
SUPER SHORT C++ SOLUTION:

My idea is simple: store states of each row in the first of that row, 
and store states of each column in the first of that column. 
Because the state of row0 and the state of column0 would occupy the 
same cell, I let it be the state of row0, and use another variable 
"col0" for column0. In the first phase, use matrix elements to set 
states in a top-down way. In the second phase, use states to set 
matrix elements in a bottom-up way.

void setZeroes(vector<vector<int> > &matrix) {
    int col0 = 1, rows = matrix.size(), cols = matrix[0].size();

    for (int i = 0; i < rows; i++) {
        if (matrix[i][0] == 0) col0 = 0;
        for (int j = 1; j < cols; j++)
            if (matrix[i][j] == 0)
                matrix[i][0] = matrix[0][j] = 0;
    }

    for (int i = rows - 1; i >= 0; i--) {
        for (int j = cols - 1; j >= 1; j--)
            if (matrix[i][0] == 0 || matrix[0][j] == 0)
                matrix[i][j] = 0;
        if (col0 == 0) matrix[i][0] = 0;
    }
}
'''