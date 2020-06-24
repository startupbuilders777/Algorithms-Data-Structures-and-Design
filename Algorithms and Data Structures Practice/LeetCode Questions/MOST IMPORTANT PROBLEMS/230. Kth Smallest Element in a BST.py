'''
230. Kth Smallest Element in a BST
Medium

Share
Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Example 1:

Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
Example 2:

Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3
Follow up:

What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? 
How would you optimize the kthSmallest routine?
'''

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # LETS DO INORDER THIS TIME. 
        # you can also just convert to sorted array, and get kth element. 
        # in other words, just do inorder traversal of tree. 
        # and keep subtracting 1 from k?
        
        found = None
        count = 0
        
        def inorder(node):
            nonlocal found
            nonlocal count
            
            # Ok found left side. 
            if node is None:
                return 
            
            inorder(node.left)
            count += 1
            if count == k:
                found = node.val
                return 
            inorder(node.right)
        inorder(root)
        return found
        
# Follow up Answer:
# USE A B-TREE
