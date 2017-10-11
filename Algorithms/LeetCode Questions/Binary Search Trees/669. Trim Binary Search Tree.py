'''
Given a binary search tree and the lowest and highest boundaries as L and R, trim the tree so that all its elements lies in [L, R] (R >= L). You might need to change the root of the tree, so the result should return the new root of the trimmed binary search tree.

Example 1:
Input: 
    1
   / \
  0   2

  L = 1
  R = 2

Output: 
    1
      \
       2
Example 2:
Input: 
    3
   / \
  0   4
   \
    2
   /
  1

  L = 1
  R = 3

Output: 
      3
     / 
   2   
  /
 1


'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """

        if root is None:
            return None
        elif (root.val >= L and root.val <= R):
            root.left = self.trimBST(root.left, L, R)
            root.right = self.trimBST(root.right, L, R)
            return root
        elif (root.val >= L):
            return self.trimBST(root.left, L, R)
        elif (root.val <= R):
            return self.trimBST(root.right, L, R)
        else:
            print("Shouldnt be in this case")

#FASTERRRRR


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        if root == None:
            return None

        if root.val < L:
            root.left = None
            return self.trimBST(root.right, L, R)
        elif root.val == L:
            root.left = None
            root.right = self.trimBST(root.right, L, R)
            return root

        if root.val > R:
            root.right = None
            return self.trimBST(root.left, L, R)
        elif root.val == R:
            root.right = None
            root.left = self.trimBST(root.left, L, R)
            return root

        l_tree = self.trimBST(root.left, L, R)
        r_tree = self.trimBST(root.right, L, R)
        root.left = l_tree
        root.right = r_tree
        return root