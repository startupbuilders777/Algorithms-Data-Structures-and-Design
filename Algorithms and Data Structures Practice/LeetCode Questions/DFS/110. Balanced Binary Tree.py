'''
Given a binary tree, determine if it is height-balanced.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of 
every node never differ by more than 1.

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: TreeNode) -> bool:
        
        
        def balance(node):
            if node is None:
                return (True, 0)
            
            
            leftBal, h1 = balance(node.left)
            rightBal, h2 = balance(node.right)

            if (not leftBal or not rightBal):
                return (False, 0)
            
            if(abs(h1 - h2) <= 1):
                return (True, max(h1, h2) + 1)
            else: 
                return (False, 0)
        
        return balance(root)[0]
