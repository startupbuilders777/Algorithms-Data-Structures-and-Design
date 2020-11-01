'''
993. Cousins in Binary Tree
Easy

1109

61

Add to List

Share
In a binary tree, the root node is at depth 0, and children of each depth k node are at depth k+1.

Two nodes of a binary tree are cousins if they have the same depth, but have different parents.

We are given the root of a binary tree with unique values, and the values x and y of two different nodes in the tree.

Return true if and only if the nodes corresponding to the values x and y are cousins.

 

Example 1:


Input: root = [1,2,3,4], x = 4, y = 3
Output: false
Example 2:


Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true
Example 3:



Input: root = [1,2,3,null,4], x = 2, y = 3
Output: false
 

Constraints:

The number of nodes in the tree will be between 2 and 100.
Each node has a unique integer value from 1 to 100.


'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

# NONLOCAL IS REQUIRED IN THIS CASE TO WORK!!!

class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:        
        depthX = None
        parentX = None
        
        depthY = None
        parentY = None
        
        
        def find(node, parent, h):
            nonlocal depthX, depthY, parentX, parentY
            
            if node is None:
                return 
            
            if node.val == x:
                depthX = h
                parentX = parent
                return 
            
            if node.val == y:
                depthY = h
                parentY = parent
                return
            
            find(node.left, node, h+1)
            find(node.right, node, h+1)
        
        find(root, -1, 0)
                
        if depthX is not None and depthY is not None and depthX == depthY and parentX != parentY:
            return True
        
        return False

# Better  way to do it without nonlocal


class Solution:
    def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
        m = {}
        def _solve(node, parent, depth):
            if node:
                if node.val == x:
                    m[x] = (depth, parent.val)
                elif node.val == y:
                    m[y] = (depth, parent.val)
                    
                if x not in m or y not in m:
                    _solve(node.left, node, depth+1)
                    _solve(node.right, node, depth+1)
        
        _solve(root, root, 0)
        return m[x][0] == m[y][0] and m[x][1] != m[y][1]