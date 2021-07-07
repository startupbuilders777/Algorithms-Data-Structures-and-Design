'''
221. Maximal Square
Medium

3591

93

Add to List

Share
Given a 2D binary matrix filled with 0's and 1's, find the largest 
square containing only 1's and return its area.

Example:

Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4
Accepted
294,418
Submissions
774,302

'''

def maximalSquare(matrix):
    
    '''
    then do maximal rectangle. 
    Go right and go down. 
    question -> how many 1's below me?
    
    1 1 1 1
    1 2 2 2 
    1 2 3
    
    take max of top, left, and top-left to produce current. 
    3 coordinates
    keep track of largest square.     
    
    space wise we only need prev row to compute
    current row, as well as previously computed value
    aka left adjacent. 
    
    Recurrence:
    dp(i,j) = min(dp(i−1, j), dp(i−1, j−1), dp(i, j−1)) + 1

    BASE CASE: 
    matrix[i,j] == '0' THEN return 0
    
    '''
    R = len(matrix)
    if R == 0:
        return 0
        
    C = len(matrix[0])
    
    
    prevRow = [0 for j in range(C+1)]
    maxSquare = 0
    
    for i in range(R):
        # we have to zero pad. 
        currRow = [0]
        
        for j in range(1, C+1):
            # if current value is 0, put 0.
            val = matrix[i][j-1]
            if val == "0":
                currRow.append(0)
            else:
                minOfTopAndLeft = min(currRow[-1], prevRow[j-1], prevRow[j])
                cellVal = minOfTopAndLeft + 1
                maxSquare = max(maxSquare, cellVal**2)
                currRow.append(cellVal)
                
        prevRow = currRow[::]
    return maxSquare
