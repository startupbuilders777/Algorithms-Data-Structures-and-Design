/*
1457. Pseudo-Palindromic Paths in a Binary Tree
Medium

331

12

Add to List

Share
Given a binary tree where node values are digits from 1 to 9. A path in the binary tree is said to be pseudo-palindromic if at least one permutation of the node values in the path is a palindrome.

Return the number of pseudo-palindromic paths going from the root node to leaf nodes.

 

Example 1:



Input: root = [2,3,1,3,1,null,1]
Output: 2 
Explanation: The figure above represents the given binary tree. There are three paths going from the root node to leaf nodes: the red path [2,3,3], the green path [2,1,1], and the path [2,3,1]. Among these paths only red path and green path are pseudo-palindromic paths since the red path [2,3,3] can be rearranged in [3,2,3] (palindrome) and the green path [2,1,1] can be rearranged in [1,2,1] (palindrome).
Example 2:



Input: root = [2,1,1,1,3,null,null,null,null,null,1]
Output: 1 
Explanation: The figure above represents the given binary tree. There are three paths going from the root node to leaf nodes: the green path [2,1,1], the path [2,1,3,1], and the path [2,1]. Among these paths only the green path is pseudo-palindromic since [2,1,1] can be rearranged in [1,2,1] (palindrome).
Example 3:

Input: root = [9]
Output: 1
 

Constraints:

The given binary tree will have between 1 and 10^5 nodes.
Node values are digits from 1 to 9.

*/


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

#import <unordered_map>

using namespace std; 

class Solution {
public:
    int pseudoPalindromicPaths (TreeNode* root) {
        // count it up!
        unordered_map<int, int> m;
        int count = helper(root, m);
        return count; 
    }
    
    int palinCheck(const unordered_map<int, int>& m) {
        bool odd = false; // only  
        for(const auto & [key, value] : m) {
            if(value % 2 != 0 && odd) {
                return false;
            }  else if(value % 2 != 0 ) {
                odd = true;
            }
        }
        return true;    
    }
    
    int helper(TreeNode* node, unordered_map<int, int>& m) {
        m[node->val] = m.find(node->val) != m.end() ? m[node->val] + 1 : 1;
        int res = 0;
        if(node->left == nullptr && node->right == nullptr) {            
           res = palinCheck(m); 
        }
        
        if(node->left) {
            res += helper(node->left, m);
        }
        
        if(node->right) {
            res += helper(node->right, m);
        }
        
        m[node->val] -= 1; 
        return res; 
    }
    
};


/*
BEST SOLUTION WITH XOR: 

class Solution:
    def pseudoPalindromicPaths (self, root: TreeNode) -> int:
        def preorder(node, path):
            nonlocal count
            if node:
                # compute occurences of each digit 
                # in the corresponding register
                path = path ^ (1 << node.val)
                # if it's a leaf, check if the path is pseudo-palindromic
                if node.left is None and node.right is None:
                    # check if at most one digit has an odd frequency
                    if path & (path - 1) == 0:
                        count += 1
                else:                    
                    preorder(node.left, path)
                    preorder(node.right, path) 
        
        count = 0
        preorder(root, 0)
        return count


*/
