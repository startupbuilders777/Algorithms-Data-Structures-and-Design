'''
Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down 
to the farthest leaf node.

'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def maxHeight(node, count):
            if (node.left is None and node.right is None):
                return count
            elif (node.left is None and node.right is not None):
                return maxHeight(node.right, count + 1)
            elif (node.right is None and node.left is not None):
                return maxHeight(node.left, count + 1)
            else:
                return max(maxHeight(node.left, count + 1),
                           maxHeight(node.right, count + 1))

        if (root is None):
            return 0
        else:
            return maxHeight(root, 1)

#FASTER SOLUTIONS

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0

        return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0

        level = 0
        cur_nodes = [root]
        next_nodes = []

        while len(cur_nodes) > 0:
            level += 1
            for cur_node in cur_nodes:
                if cur_node.left is not None:
                    next_nodes.append(cur_node.left)
                if cur_node.right is not None:
                    next_nodes.append(cur_node.right)

            cur_nodes = next_nodes
            next_nodes = []

        return level