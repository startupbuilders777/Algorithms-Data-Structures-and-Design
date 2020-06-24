'''
124. Binary Tree Maximum Path Sum
Hard

3556

285

Add to List

Share
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes 
from some starting node to any node in the tree along the parent-child connections. 
The path must contain at least one node and does not need to go through the root.

Example 1:

Input: [1,2,3]

       1
      / \
     2   3

Output: 6
Example 2:

Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        
        # find the max path sum for each path, then connect it to a apex node. 
        # max sum left. max sum rite. max sum using this node. 
        m = float(-inf)
        
        def helper(node):
            nonlocal m
            if node is None:
                return 0
            
            # Only keep paths, not the max sum
            maxRightPath =  helper(node.right)
            maxLeftPath = helper(node.left)
            
            # if using node creates a path of sum < 0, just recurse up 0
            # but cant the max sum be a negative number??            
            right = maxRightPath + node.val
            left = maxLeftPath + node.val
            connected = maxRightPath + maxLeftPath + node.val
            
            # but m cannot be 0, because it needs to have at least 1 node.
            m = max(m, right, left, connected, node.val)
            
            
            # 0 to node include node
            maxPath = max(right, left, node.val, 0)
            return maxPath
            
        helper(root)
        return m
            
