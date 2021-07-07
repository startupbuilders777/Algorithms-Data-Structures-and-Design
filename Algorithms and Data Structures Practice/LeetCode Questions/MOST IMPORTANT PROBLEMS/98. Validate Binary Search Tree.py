'''
98. Validate Binary Search Tree
Medium

3608

514

Add to List

Share
Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.
 

Example 1:

    2
   / \
  1   3

Input: [2,1,3]
Output: true
Example 2:

    5
   / \
  1   4
     / \
    3   6

Input: [5,1,4,null,null,3,6]
Output: false
Explanation: The root node's value is 5 but its right child's value is 4.

'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        
        
        # check if left key is smaller, right node is bigger or none.
        # then call is valid on left and right. return result. 
        
        if root == None:
            return True
        
        def helper(root, minimum, maximum):
            
            if(root is None):
                return True
            
            if((root.left != None and root.left.val >= root.val) or 
               (root.right != None and root.right.val <= root.val )):
                return False
            
            if root.left != None and root.left.val <= minimum:
                return False
            
            if root.right != None and root.right.val >= maximum:
                return False
            return (helper(root.left, minimum, root.val) and helper(root.right, root.val, maximum))
        
        return helper(root, float("-inf"), float("inf"))
    
    
# Faster solutions:

# YOU CAN DO INORDER TRAVERSAL TO VALIDATE BST: 
# MEMORIZE THIS TECHNIQUE


def isValidBST(self, root):
    res, self.flag = [], True
    self.helper(root, res)
    return self.flag
    
def helper(self, root, res):
    if root:
        self.helper(root.left, res)
        if res and root.val <= res[-1]:
            self.flag = False
            return
        res.append(root.val)
        self.helper(root.right, res)

# Or iterative inorder tree traversal:


# iteratively, in-order traversal
# O(n) time and O(n)+O(lgn) space
class Solution:
# @param root, a tree node
# @return a boolean
def isValidBST(self, root):
    pre, cur, stack = None, root, []
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.left
        s = stack.pop()
        if pre and s.val <= pre.val:
            return False
        pre, cur = s, s.right
    return True