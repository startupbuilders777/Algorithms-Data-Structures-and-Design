'''
You are given a map in form of a two-dimensional integer grid where 1 represents land 
and 0 represents water. Grid cells are connected horizontally/vertically (not diagonally). 
The grid is completely surrounded by water, and there is exactly one island (i.e., one or 
more connected land cells). The island doesn't have "lakes" (water inside that isn't 
connected to the water around the island). One cell is a square with side length 1. 
The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

Example:

[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]

Answer: 16
Explanation: The perimeter is the 16 yellow stripes in the image below:

TECHNIQUE:

Look at the 0's that are adjacent to the 1's
If the 0 is adjacent to no 1's -> add 0
If it is adjacent to 2 -> add 2
If it is adjact to 3 -> add 3
If it is adjacent to 4 -> add 4

ETC.

'''


class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """

        def checkAdjacency(indexRow, indexCol, grid):
            numberOfOnesAdjacentToOne = 0
            if (indexCol + 1 < len(grid[indexRow]) and grid[indexRow][indexCol + 1] == 1):
                numberOfOnesAdjacentToOne += 1
            if (indexCol - 1 >= 0 and grid[indexRow][indexCol - 1] == 1):
                numberOfOnesAdjacentToOne += 1

            if (indexRow + 1 < len(grid) and grid[indexRow + 1][indexCol] == 1):
                numberOfOnesAdjacentToOne += 1

            if (indexRow - 1 >= 0 and grid[indexRow - 1][indexCol] == 1):
                numberOfOnesAdjacentToOne += 1
            print(numberOfOnesAdjacentToOne)
            return 4 - numberOfOnesAdjacentToOne

        perimeter = 0
        for row in range(0, len(grid)):
            for col in range(0, len(grid[0])):
                if grid[row][col] == 1:
                    perimeter += checkAdjacency(row, col, grid)

        return perimeter

'''
FASTEST SOLUTION
'''

class Solution(object):
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid:
            return 0
        s = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if grid[i][j]==1:
                    s+=4
                    if i!=0:
                        if grid[i-1][j]>0:
                            s -=2
                    if j!=0:
                        if grid[i][j-1]>0:
                            s -=2
        return s


