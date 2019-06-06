#DONE

'''
993. Cousins in Binary Tree
Easy

165

8

Favorite

Share
In a binary tree, the root node is at depth 0, and children of each depth k node are at depth k+1.

Two nodes of a binary tree are cousins if they have the same depth, but have different parents.

We are given the root of a binary tree with unique values, and the values x and y of two different nodes in the tree.

Return true if and only if the nodes corresponding to the values x and y are cousins.

 

Example 1:


Input: root = [1,2,3,4], x = 4, y = 3
Output: false
Example 2:


Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true
Example 3:



Input: root = [1,2,3,null,4], x = 2, y = 3
Output: false
 

Note:

The number of nodes in the tree will be between 2 and 100.
Each node has a unique integer value from 1 to 100.
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

from collections import deque 

class Solution(object):
    def isCousins(self, root, x, y):
        """
        :type root: TreeNode
        :type x: int
        :type y: int
        :rtype: bool
        """
        
        '''
        Three solutions:
        
        BFS with dist map, visited set, queue
        ensure both nodes have same dist value. 
        
        OR use 2 queues. 
        
        
        
        '''
        
        if(root is None):
            return False
        
        currDepth = deque([root])
        nextDepth = deque()
        parents = {}
        parents[root] = None
        
        # to do bfs, do append() and popleft()
        
        foundX = False
        foundY = False
        
        while currDepth: 
            node = currDepth.popleft()
            print("node", node)
            children = [node.left, node.right]
            
            for i in children:
                if(i): 
                    if(i.val == x):
                        foundX = True
                    if(i.val == y):
                        foundY = True

                    parents[i.val] = node.val
                    nextDepth.append(i)

            if(foundX and foundY):
                break
                
            if(len(currDepth) == 0 and len(nextDepth) > 0):
                currDepth = nextDepth
                nextDepth = deque()
            
        
        if(foundX and foundY):
            # Check they are in same depth by checking if they in nextDepth
            sameLevelX = False
            sameLevelY = False
            
            while nextDepth:
                
                element = nextDepth.popleft()
                if(element.val == x):
                    sameLevelX = True
                if(element.val == y):
                    sameLevelY = True
            if(sameLevelX and sameLevelY):
                print("parents x", parents[x])
                print("parents y", parents[y])
                
                if(parents[x] != parents[y]):
                    
                    return True
            return False
                
            
        else:
            return False
            
