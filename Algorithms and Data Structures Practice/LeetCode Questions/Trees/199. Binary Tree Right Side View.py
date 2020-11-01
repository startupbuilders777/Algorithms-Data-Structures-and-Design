'''
199. Binary Tree Right Side View
Medium

2857

162

Add to List

Share
Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

Example:

Input: [1,2,3,null,5,null,4]
Output: [1, 3, 4]
Explanation:

   1            <---
 /   \
2     3         <---
 \     \
  5     4       <---

'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: TreeNode) -> List[int]:
        
        # get righmost values. 
        # user can only see 1 node per level. 
        '''
        traverse tree right first, 
        then left.
        
        assign to global variable. 
        keep track of height you traversed. 
        
        '''
        
        res = []
        
        def helper(node, h):
            nonlocal res 
            
            if node is None:
                return
            
            if len(res) < h:
                res.append(node.val) 
            helper(node.right, h+1)
            helper(node.left, h+1)
        
        helper(root, 1)
        return res
    

