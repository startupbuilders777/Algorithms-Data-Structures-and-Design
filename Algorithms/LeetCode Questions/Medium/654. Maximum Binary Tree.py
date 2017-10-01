'''
Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:

The root is the maximum number in the array.
The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
Construct the maximum tree by the given array and output the root node of this tree.

Example 1:
Input: [3,2,1,6,0,5]
Output: return the tree root node representing the following tree:

      6
    /   \
   3     5
    \    / 
     2  0   
       \
        1
Note:
The size of the given array will be in the range [1,1000].

'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    # Could Probably do this inplace but fck that
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """

        def getMaxAndIndexOfMax(nums):
            if (len(nums) <= 0):
                return (None, None)
            else:
                max = nums[0]
                maxIndex = 0
                for i in range(1, len(nums)):
                    if (max < nums[i]):
                        max = nums[i]
                        maxIndex = i
                return (max, maxIndex)

        if (len(nums) <= 0):
            return None
        else:
            (max, maxIndex) = getMaxAndIndexOfMax(nums)
            node = TreeNode(max)
            node.left = self.constructMaximumBinaryTree(nums[0:maxIndex])
            node.right = self.constructMaximumBinaryTree(nums[maxIndex + 1:])
            return node

