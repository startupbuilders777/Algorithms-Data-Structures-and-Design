# DONE
'''
Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.
'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        
        '''
        if root is within L, and R, add it to sum.
        Otherwise, choose a side and recurse
        
        '''
        
        def rec(node, L, R):
            if(node is None):
                return 0

            if(node.val >= L and node.val <= R):
                print("add ", node.val)
                s = node.val 
                leftSum = rec(node.left, L, R)
                rightSum = rec(node.right, L, R)
                return s + leftSum + rightSum 
            
            elif(node.val > R): 
                # root val really big, recurse on left
                return rec(node.left, L, R)
                
            else: # root val less than L
                return rec(node.right, L, R)
        
        return rec(root, L, R)
        
'''

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        
        if(root == nullptr) {
            return 0;
        }     
        int sum = 0;
        if(root->val >= low && root->val <= high) {
            return root->val + rangeSumBST(root->left, low, high) + rangeSumBST(root->right, low, high);
        } else if(root->val < low) {
            return rangeSumBST(root->right, low, high);
        } else {
            return rangeSumBST(root->left, low, high);
        }
    }
};

'''  