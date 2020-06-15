'''
Given a binary tree, return the postorder traversal of its nodes' values.

Example:

Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [3,2,1]
Follow up: Recursive solution is trivial, could you do it iteratively?

'''

# TO DO IT, ITERATIVELY, YOU NEED A FLAG THAT MAKES YOUR VISIT YOUR CHILDREN FIRST BEFORE YOURSELF:

class Solution:
    # @param {TreeNode} root
    # @return {integer[]}
    def postorderTraversal(self, root):
        traversal, stack = [], [(root, False)]
        while stack:
            node, visited = stack.pop()
            if node:
                if visited:
                    # add to result if visited
                    traversal.append(node.val)
                else:
                    # post-order
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))

        return traversal



# ANTHER SOLUTION THAT USES A DEQUE, AND APPENDLEFT TO SIMULATE IT

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
from collections import deque

class Solution:
    def postorderTraversal(self, root: TreeNode) -> List[int]:
        d = deque()
        stack = [ (root, d )]
        
        while stack:
            el, r = stack.pop()
            
            if(el is None):
                continue
            
            r.appendleft(el.val)
            stack.append( (el.left, r) )
            stack.append( (el.right, r) )
            
        return list(d)