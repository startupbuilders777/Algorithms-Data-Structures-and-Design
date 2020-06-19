'''
572. Subtree of Another Tree
Easy

2198

111

Add to List

Share
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
 

Example 2:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
'''

class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def contains(s, t):
            # t has to be in s, not in a discontinuous way
            if t == None and s == None:
                return True

            elif s == None or t == None:
                return False

            if(s.val == t.val and contains(s.left, t.left) and contains(s.right, t.right)):
                return True
            else:
                return False
        
        if t == None:
            return True
        
        if s == None:
            return False
        
        if s.val == t.val and contains(s, t):
            return True
        else:
            return any( [self.isSubtree(s.left, t), self.isSubtree(s.right, t) ]) 


# Fastest soln:

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        def traverse(s):
            if not s:
                return None
            return f"# {traverse(s.left)} {s.val} {traverse(s.right)}"
            
            
        s1 = traverse(s)
        s2 = traverse(t)
        return s2 in s1
