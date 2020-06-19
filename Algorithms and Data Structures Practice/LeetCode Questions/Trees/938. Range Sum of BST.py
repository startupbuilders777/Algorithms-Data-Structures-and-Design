# DONE
'''
Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        
        '''
        if root is within L, and R, add it to sum.
        Otherwise, choose a side and recurse
        
        '''
        
        def rec(node, L, R):
            if(node is None):
                return 0

            if(node.val >= L and node.val <= R):
                print("add ", node.val)
                s = node.val 
                leftSum = rec(node.left, L, R)
                rightSum = rec(node.right, L, R)
                return s + leftSum + rightSum 
            
            elif(node.val > R): 
                # root val really big, recurse on left
                return rec(node.left, L, R)
                
            else: # root val less than L
                return rec(node.right, L, R)
        
        return rec(root, L, R)
        
       