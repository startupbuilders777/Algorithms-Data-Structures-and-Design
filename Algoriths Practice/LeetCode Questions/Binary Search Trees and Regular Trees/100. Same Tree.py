'''
Given two binary trees, write a function to check if they are equal or not.

Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if(p is not None and q is not None):
            if(p.val != q.val):
                return False
            else:
                leftSame = self.isSameTree(p.left, q.left)
                rightSame = self.isSameTree(p.right, q.right)
                return leftSame and rightSame
        elif(p is not None and q is None or p is None and q is not None):
            return False
        else:
            return True