'''
566. Reshape the Matrix
DescriptionHintsSubmissionsDiscussSolution
Discuss Pick One
In MATLAB, there is a very useful function called 'reshape', which can reshape a matrix into a new one with different size but keep its original data.

You're given a matrix represented by a two-dimensional array, and two positive integers r and c representing the row number and column number of the wanted reshaped matrix, respectively.

The reshaped matrix need to be filled with all the elements of the original matrix in the same row-traversing order as they were.

If the 'reshape' operation with given parameters is possible and legal, output the new reshaped matrix; Otherwise, output the original matrix.

Example 1:
Input: 
nums = 
[[1,2],
 [3,4]]
r = 1, c = 4
Output: 
[[1,2,3,4]]
Explanation:
The row-traversing of nums is [1,2,3,4]. The new reshaped matrix is a 1 * 4 matrix, fill it row by row by using the previous list.
Example 2:
Input: 
nums = 
[[1,2],
 [3,4]]
r = 2, c = 4
Output: 
[[1,2],
 [3,4]]
Explanation:
There is no way to reshape a 2 * 2 matrix to a 2 * 4 matrix. So output the original matrix.
Note:
The height and width of the given matrix is in range [1, 100].
The given r and c are all positive.

'''


class Solution(object):
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        if (nums is [] or nums is None):
            return nums

        rowsOfMatrix = len(nums)

        try:
            colsOfMatrix = len(nums[0])
        except:
            print("Not 2D Array")
            return nums

        lst = []

        if (r * c != rowsOfMatrix * colsOfMatrix):
            return nums
        else:
            count = 0
            tempLst = []
            for i in nums:
                for j in i:
                    print(j)
                    tempLst.append(j)
                    count += 1
                    if (count == c):
                        count = 0
                        lst.append(tempLst)
                        # lst.append(tempLst[:]) <- you could also put this if youre scared of copying a reference and want a copy
                        tempLst = []

        return lst