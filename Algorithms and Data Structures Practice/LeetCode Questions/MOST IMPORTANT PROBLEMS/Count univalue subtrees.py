'''
921. Count Univalue Subtrees
中文English
Given a binary tree, count the number of uni-value subtrees.

A Uni-value subtree means all nodes of the subtree have the same value.

Example
Example1

Input:  root = {5,1,5,5,5,#,5}
Output: 4
Explanation:
              5
             / \
            1   5
           / \   \
          5   5   5
Example2

Input:  root = {1,3,2,4,5,#,6}
Output: 3
Explanation:
              1
             / \
            3   2
           / \   \
          4   5   6

'''
# solution

class Solution:
    """
    @param root: the given tree
    @return: the number of uni-value subtrees.
    """
    def countUnivalSubtrees(self, root):
        '''
        leafs 
        
        count it, 
        
        return up -> 
        
        process kids 
            then process node. 
            bottom up processing saves computation
            because if a child couldnt yield anything, then its parents cant yield anything either. 
            as you process each node add to sum!
        '''
        count = 0
        def helper(node):
            nonlocal count 
            if node is None:
                return None
            left_result = helper(node.left)
            right_result = helper(node.right)
            
            if left_result == False:
                return False
            if right_result == False:
                return False
            if left_result and left_result != node.val:
                return False    
            if right_result and right_result != node.val:
                return False
            count += 1
            return node.val
        
        helper(root)
        return count
