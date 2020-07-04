'''
114. Flatten Binary Tree to Linked List
Medium

2673

327

Add to List

Share
Given a binary tree, flatten it to a linked list in-place.

For example, given the following tree:

    1
   / \
  2   5
 / \   \
3   4   6
The flattened tree should look like:

1
 \
  2
   \
    3
     \
      4
       \
        5
         \
          6
'''

# BEST SOLUTION

'''
O(1) SPACE SOLUTION

So what this solution is basically doing is putting the 
right subtree next to the rightmost node on the left subtree 
and then making the left subtree the right subtree and 
then making the left one null. Neat!
'''

class Solution:
    # @param root, a tree node
    # @return nothing, do it in place
    def flatten(self, root):
        if not root:
            return
        
        # using Morris Traversal of BT
        node=root
        
        while node:
            if node.left:
                pre=node.left
                while pre.right:
                    pre=pre.right
                pre.right=node.right
                node.right=node.left
                node.left=None
            node=node.right
            
   
          

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """  
        stack = []
        prev = root
        
        if not root:
            return root
        
        if root.right:
            stack.append(root.right)
        
        if root.left:
            stack.append(root.left)

        root.left = None
        
        while stack:    
            tn = stack.pop()
            if tn.right:
                stack.append(tn.right)
            if tn.left:
                stack.append(tn.left)

            tn.left = None
            prev.right = tn
            prev = prev.right 



            


# RECURSIVE FASTEST SOLN

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        self.previous = TreeNode()
        self.traverse(root)
        return root
    def traverse(self, root):
        if root == None:
            return
        left, right = root.left, root.right
        self.previous.right = root
        self.previous.left = None
        self.previous = root
        self.traverse(left)
        self.traverse(right)
            
            
            
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        if root:
            self.flatten(root.right)

            if root.left:
                self.flatten(root.left)
                temp = root.left
                while(temp.right):
                    temp = temp.right
                temp.right = root.right    
                root.right = root.left
                root.left = None

            
        

        """
        Do not return 
        """

'''
REVERSE PREORDER TRAVERSAL SOLUTION:

'''

class Solution:
    def __init__(self):
        self.prev = None
    
    def flatten(self, root):
        if not root:
            return None
        self.flatten(root.right)
        self.flatten(root.left)
        
        root.right = self.prev
        root.left = None
        self.prev = root