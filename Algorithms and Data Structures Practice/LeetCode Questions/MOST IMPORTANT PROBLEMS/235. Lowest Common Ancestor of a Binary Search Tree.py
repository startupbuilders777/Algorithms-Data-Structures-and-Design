'''
235. Lowest Common Ancestor of a Binary Search Tree
Easy

1939

104

Add to List

Share
Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined 
between two nodes p and q as the lowest node in T that has both p and q as 
descendants (where we allow a node to be a descendant of itself).”

Given binary search tree:  root = [6,2,8,0,4,7,9,null,null,3,5]


 

Example 1:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
Example 2:

Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
 

Note:

All of the nodes' values will be unique.
p and q are different and both values will exist in the BST.

'''

# My soln:

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def LCA(node, leftVal, rightVal):    
            if node is None:
                return None
            if(node.val < leftVal and node.val < rightVal):
                return LCA(node.right, leftVal, rightVal)
        
            if(node.val > leftVal and node.val > rightVal):
                return LCA(node.left, leftVal, rightVal)
            
            # we found the split point! node can either be left or right, or whatever. 
            return node
        
        return LCA(root, p.val, q.val)
    
# Other cool ways to do it:

def lowestCommonAncestor(self, root, p, q):
    while (root.val - p.val) * (root.val - q.val) > 0:
        root = (root.left, root.right)[p.val > root.val]
    return root

def lowestCommonAncestor(self, root, p, q):
    a, b = sorted([p.val, q.val])
    while not a <= root.val <= b:
        root = (root.left, root.right)[a > root.val]
    return root

def lowestCommonAncestor(self, root, p, q):
    next = p.val < root.val > q.val and root.left or \
           p.val > root.val < q.val and root.right
    return self.lowestCommonAncestor(next, p, q) if next else root

def lowestCommonAncestor(self, root, p, q):
    return root if (root.val - p.val) * (root.val - q.val) < 1 else \
           self.lowestCommonAncestor((root.left, root.right)[p.val > root.val], p, q)

