'''
437. Path Sum III
Easy

3293

268

Add to List

Share
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, 
but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def verifySum(self, node, val):         
        if node is None:
            return 0
        count = 0
        if node.val == val:
            count = 1    
        verifyRight = self.verifySum(node.right, val - node.val)
        verifyLeft = self.verifySum(node.left, val - node.val)
        count += (verifyRight + verifyLeft)    
        return count
    
    def pathSum(self, root: TreeNode, sum: int) -> int: 
        if root is None:
            return 0   
        count = 0
        count = self.verifySum(root, sum)
        right = self.pathSum(root.right, sum)
        left =  self.pathSum(root.left, sum)
        count += right
        count += left
        return count


# MEMOIZATION SOLUTION:

class Solution:
    def dfs(self,node,isum,target):
        if node ==None:
            return
        nxtSum = isum + node.val
        
          
        if nxtSum - target in self.map:
            self.count += self.map[nxtSum - target]
        
        if nxtSum not in self.map:
            self.map[nxtSum] = 1
        else:    
            self.map[nxtSum] += 1
        
        self.dfs(node.left,nxtSum,target)
        self.dfs(node.right,nxtSum,target)
        
        self.map[nxtSum] -= 1
            
    
    def pathSum(self, root: TreeNode, sum: int) -> int:
        self.map = {}
        self.map[0] = 1
        self.count = 0
        self.dfs(root,0,sum)
        return self.count