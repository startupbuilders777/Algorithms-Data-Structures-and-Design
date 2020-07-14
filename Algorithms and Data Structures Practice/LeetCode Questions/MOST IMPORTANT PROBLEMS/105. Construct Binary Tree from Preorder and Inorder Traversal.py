'''
105. Construct Binary Tree from Preorder and Inorder Traversal
Medium

3331

92

Add to List

Share
Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
Accepted
357,562
Submissions
742,788
'''


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        
        # use preorder, determine sizes with inorder!
        # use indexes to reduce copying!
        preorder_idx = 0
        def build(inorder_i, inorder_j):
            nonlocal preorder_idx
            
            if preorder_idx == len(preorder) or inorder_i > inorder_j:
                return None
            
            pivot = preorder[preorder_idx]
            # we need to build left and right!
            node = TreeNode(pivot)
            
            # OK WE CAN BUILD A NODE IF THE ABOVE CASE PASSES!
            # must increment here. 
            preorder_idx += 1
            
            left_i = inorder_i
            left_j = None
            right_i = None
            right_j = inorder_j
            
            temp = inorder_i
            
            while inorder[temp] != pivot:
                temp += 1
            
            left_j = temp - 1
            right_i = temp + 1
            
            leftSide = build(left_i, left_j)
            rightSide = build(right_i, right_j)        
            
            node.left = leftSide
            node.right = rightSide
            return node
            
        return build(0, len(inorder) - 1)

