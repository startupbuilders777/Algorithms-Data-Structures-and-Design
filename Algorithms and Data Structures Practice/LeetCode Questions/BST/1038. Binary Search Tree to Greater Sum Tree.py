
# Done

'''
Given the root of a binary search tree with distinct values, modify it so 
that every node has a new value equal to the sum of the values of the original 
tree that are greater than or equal to node.val.

As a reminder, a binary search tree is a tree that satisfies these constraints:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
 

Example 1:



Input: [4,1,6,0,2,5,7,null,null,null,3,null,null,null,8]
Output: [30,36,21,36,35,26,15,null,null,null,33,null,null,null,8]
 

Note:

The number of nodes in the tree is between 1 and 100.
Each node will have value between 0 and 100.
The given tree is a binary search tree.

'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def bstToGst(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
    
        if(root is None):
            return TreeNode(root.val)
        
        def recursive_sum(root, parent_sum):
            
            # Get sum of all the right nodes. 
            # if i am a left node get sum of parent
            
            newTreeNode = TreeNode(0)
            if(root is None):
                return (0, None)
            
            #if(root.left is None and root.right is None):
            # newTreeNode.val = root.val + parent_sum
            #    return (root.val, newTreeNode)
            
           
            (right_sum, right_node) = recursive_sum(root.right, parent_sum)
            
            newTreeNode.val = root.val + parent_sum + right_sum
            
            newTreeNode.right = right_node
            
            (left_sum, left_node) = recursive_sum(root.left, newTreeNode.val)
            
            
            newTreeNode.left = left_node
            
            
            return (root.val + left_sum + right_sum, newTreeNode)
    
        return recursive_sum(root, 0)[1]



# VERY SMALL SOLUTION

class Solution(object):
    val = 0
    def bstToGst(self, root):
        if root.right: self.bstToGst(root.right)
        root.val = self.val = self.val + root.val
        if root.left: self.bstToGst(root.left)
        return root

# Faster 

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def bstToGst(self, root: TreeNode) -> TreeNode:
        self.bst(root,0)
        return root
    def bst(self, r, s):
        if not r:
            return s
        r.val += self.bst(r.right,s)
        return self.bst(r.left, r.val)