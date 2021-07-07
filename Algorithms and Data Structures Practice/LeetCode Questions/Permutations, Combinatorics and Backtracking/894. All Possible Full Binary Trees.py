#DONE 
'''
894. All Possible Full Binary Trees
Medium

409

35

Favorite

Share
A full binary tree is a binary tree where each node has exactly 0 or 2 children.

Return a list of all possible full binary trees with N nodes.  Each element of the answer is the root node of one possible tree.

Each node of each tree in the answer must have node.val = 0.

You may return the final list of trees in any order.

 

Example 1:

Input: 7
Output: [[0,0,0,null,null,0,0,null,null,0,0],[0,0,0,null,null,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,null,null,null,null,0,0],[0,0,0,0,0,null,null,0,0]]
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def allPossibleFBT(self, N: int) -> List[TreeNode]:
        m = {}
        
        def create(node_count): # Returns list of tree nodes. 
            if(m.get(node_count) is not None):
                return m.get(node_count)
            
            # You can memoize these results!!!!
            
            if(node_count == 1): 
                return [TreeNode(0)] # Just return 1 node!!
            # Now we can either add left and right, and choose a side to recurse
            
            
            # we can give left side, either 1 node to build with, 3, 5, 9
            # we then have to give right side the opposite, but right side has to be atleast 1!!
            result = []
            for i in range(1, node_count, 2):                
                left_side = create(i)
                right_side = create(node_count - 1 - i)
                
                for ls in left_side:
                    for rs in right_side:
                        n = TreeNode(0)
                        n.left = ls
                        n.right = rs
                        result.append(n)
                        
            m[node_count] = result
            return result
    
        
        return create(N)
            
