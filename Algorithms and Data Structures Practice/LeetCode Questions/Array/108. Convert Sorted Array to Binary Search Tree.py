'''
108. Convert Sorted Array to Binary Search Tree
Easy

2285

205

Add to List

Share
Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in 
which the depth of the two subtrees of every node never differ by more than 1.

Example:

Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
Accepted
394,871
Submissions
692,664

'''

class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        # l, r are indexes. 
        # r is right boundary (not included)
        
        def build_tree(l, r):
            if(l == r):
                return None
            
            mid =  l + (r-l)//2
            root = nums[mid]
            
            # you never include right value
            left = build_tree(l, mid)
            right = build_tree(mid+1, r)
            return TreeNode(val=root, left=left, right=right)
            
        return build_tree(0, len(nums))

