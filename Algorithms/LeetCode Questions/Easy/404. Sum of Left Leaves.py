'''

Find the sum of all left leaves in a given binary tree.

Example:

    3
   / \
  9  20
    /  \
   15   7

There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.
Seen this question in a real interview before?   Yes  No
Subscribe to see which companies asked this question.

Related Topics 
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def sumOfLeftLeavesRecur(root, isLeft):
            if (root is None):
                return 0
            if (root.left is None and root.right is None and isLeft):
                return root.val
            elif (root.left is None and root.right is None):
                return 0
            else:
                left = sumOfLeftLeavesRecur(root.left, True)
                right = sumOfLeftLeavesRecur(root.right, False)
                return left + right

        return sumOfLeftLeavesRecur(root, False)
