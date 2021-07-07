/*

543. Diameter of Binary Tree
Easy

1731

108

Favorite

Share
Given a binary tree, you need to compute the length of the 
diameter of the tree. The diameter of a binary tree is the 
length of the longest path between any two nodes in a tree. 
This path may or may not pass through the root.

Example:
Given a binary tree
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].

Note: The length of path between two nodes is 
represented by the number of edges between them.


*/

/*
Fastest:
*/

/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    int diameterOfBinaryTree(TreeNode* root, int& height) {
        if (!root) {
            height = -1;
            return 0;
        }
        
        int left_height, right_height, left_diameter, right_diameter;
        left_diameter = diameterOfBinaryTree(root->left, left_height);
        right_diameter = diameterOfBinaryTree(root->right, right_height);
        
        height = max(left_height, right_height) + 1;
        return max({left_height + right_height + 2,
                    left_diameter,
                    right_diameter});
    }
public:
    int diameterOfBinaryTree(TreeNode* root) {
        int height;
        return diameterOfBinaryTree(root, height);
    }
};


/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    
    
    int helper(TreeNode* root, int & currMax) {
       
        if(root == nullptr) {
           return 0;
       } 
        
        auto leftH = helper(root->left, currMax);
        auto rightH = helper(root->right, currMax);
        
        if(leftH + rightH > currMax) {
            currMax = leftH + rightH;
        } 
        
        return 1 + max(leftH, rightH);    
    }
    
    int diameterOfBinaryTree(TreeNode* root) {
        
        
        // find the max left and max right from a node. add, if thats bigger than currMax, update
        
        
        /*
        
        BFS from root node. 
        
        get dist of all kids. 
        
        dist from any 2 kids is computed through dist from root?
        
        nah
        has to be done through dist from common parent of 2 nodes!
        so need to go through parents array as well as dist array:
        
        
        */
        
        int max = 0;
        helper(root, max);
        
        return max;
        
    }
};