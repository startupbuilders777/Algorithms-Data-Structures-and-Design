'''
652. Find Duplicate Subtrees
Medium

1261

198

Add to List

Share
Given a binary tree, return all duplicate subtrees. For each kind of duplicate subtrees, you only need to return the root node of any one of them.

Two trees are duplicate if they have the same structure with same node values.

Example 1:

        1
       / \
      2   3
     /   / \
    4   2   4
       /
      4
The following are two duplicate subtrees:

      2
     /
    4
and

    4
Therefore, you need to return above trees' root in the form of a list.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        
        # Well when you process each node, you build a merkel hash for it. 
        # then you see if the hashes match, and return the roots!
        # Problem: hash collisions.
        
        tree_m = {}
        result = set()
        
        def dfs(node):
            
            if node is None:
                return "#"
            
            
            l = dfs(node.left)
            r = dfs(node.right)
            
            h = str(node.val) + "," + l + "," + r 
            
            if tree_m.get(h):
                result.add(tree_m.get(h))
            else:
                tree_m[h] = node
            
            return h
        
        dfs(root)
        return result
    
# FASTEST SOLUTION AVOID USING SET PLEASE!


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findDuplicateSubtrees(self, root: TreeNode) -> List[TreeNode]:
        counter = collections.Counter()
        res = [] 
        def dfs(node):
            if not node:
                return ' '
            serialize = str(node.val) + ',' + dfs(node.left) + ',' + dfs(node.right)
            if counter[serialize] == 1:
                res.append(node)
            counter[serialize] += 1    
            return serialize
        dfs(root)
        return res
