'''
Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

For example, given the following matrix:

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Return 6.

'''

class Solution(object):
    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        #Algo Explanation:
        '''
        Go through each element in the matrix, and count how many elements are 2 the right and are below the current 
        element and save it in a map
        this operation is O(N^2) if implemented properly       
        
        
        Then go through every element in the map again and find the element with the largest width*height value 
        and return that width*height
        
        '''
        dict = {}

        for i in matrix:
                self.countElementsToTheRight(i, 0, matrix[i], 0, dict) #Processing a row of the matrix each time

    def countElementsToTheRight(self, row, col, matrixRow, currentMax, map):
        if(col == len(matrixRow)):
            return currentMax
        elif(matrixRow[col] == 1):
            blocksToRight = self.countElementsToTheRight(row, col+1, matrixRow, currentMax+1, map)
            map[row][col] = blocksToRight

        elif(matrixRow[col] == 0):
                self.countElementsToTheRight(row, col + 1, matrixRow, currentMax + 1, map)
            map[row][col] = 0
            return currentMax


            map[row][j] = currentMax
        else:
            currentMax += 1






