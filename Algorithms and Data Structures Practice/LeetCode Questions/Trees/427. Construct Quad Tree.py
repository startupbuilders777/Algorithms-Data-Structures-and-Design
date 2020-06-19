'''

427. Construct Quad Tree
Medium

215

393

Add to List

Share
We want to use quad trees to store an N x N boolean grid. Each cell 
in the grid can only be true or false. The root node represents the whole 
grid. For each node, it will be subdivided into four children nodes until 
the values in the region it represents are all the same.

Each node has another two boolean attributes : isLeaf and val. isLeaf is 
true if and only if the node is a leaf node. The val attribute for a leaf 
node contains the value of the region it represents.

Your task is to use a quad tree to represent a given grid. The following 
example may help you understand the problem better:

Given the 8 x 8 grid below, we want to construct the corresponding quad tree:



It can be divided according to the definition above:



 

The corresponding quad tree should be as following, where each node 
is represented as a (isLeaf, val) pair.

For the non-leaf nodes, val can be arbitrary, so it is represented as *.


Note:

N is less than 1000 and guaranteened to be a power of 2.
If you want to know more about the quad tree, you can refer to its wiki.


'''

"""
# Definition for a QuadTree node.
class Node(object):
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight
"""
class Solution(object):
    def helper(self,topLeft, botRight, grid):
        x0, y0 = topLeft 
        x1, y1 = botRight
        
        if(x0 >= x1 and y0  >= y1 ):
            print("HIT A WIERD CASE")
            return None
        
        if(x0 == x1 - 1 and y0  == y1 - 1 ):
            
            cell = grid[x0][y0]
            return Node(cell, True,  None, None, None, None)    
        
        else: 
            midX = (x0 + x1)/2
            midY = (y0 + y1)/2
            
            topLeft = self.helper( (x0, y0), (midX, midY), grid)
            bottomLeft = self.helper( (midX, y0), (x1, midY), grid)
            topRight = self.helper((x0, midY), (midX, y1), grid)
            bottomRight = self.helper((midX, midY), (x1, y1), grid)
            
            
            if(topLeft and topLeft.isLeaf and 
               topRight and topRight.isLeaf and bottomLeft and 
               bottomLeft.isLeaf  and
               bottomRight and bottomRight.isLeaf and  
               topLeft.val == topRight.val and 
               topRight.val == bottomLeft.val and
               bottomLeft.val == bottomRight.val):
                # MERGE!
                return Node(topLeft.val, True, None, None, None, None)
            
            else:
                n = Node()
                n.topLeft = topLeft
                n.topRight = topRight
                n.bottomLeft = bottomLeft
                n.bottomRight = bottomRight
                
                return n
                        
        
    def construct(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: Node
        """
        
        # Similar to merge, 
        # return if its leaf or not
        # if all 4 are leaf, 
        # check if all 4 represent the same
        # thing if they do, then leaf again. 
        # O(Size of grid)
        
        return self.helper((0, 0), (len(grid), len(grid)), grid )
        