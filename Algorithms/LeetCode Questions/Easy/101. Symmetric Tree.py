'''
Given a binary tree, check whether it is a mirror 
of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3
But the following [1,2,2,null,3,null,3] is not:
    1
   / \
  2   2
   \   \
   3    3
Note:
Bonus points if you could solve it both 
recursively and iteratively.

'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        lstLeft = []
        lstRight = []

        def getLeft(node):
            if (node is None):
                lstLeft.append(None)
                return
            else:
                lstLeft.append(node.val)
                getLeft(node.left)
                getLeft(node.right)

        def getRight(node):
            if (node is None):
                lstRight.append(None)
                return
            else:
                lstRight.append(node.val)
                getRight(node.right)
                getRight(node.left)

        if (root is None):
            return True
        getLeft(root.left)
        getRight(root.right)
        print(lstLeft)
        print(lstRight)

        return lstLeft == lstRight


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#FASTER
class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        def isSym(left, right):
            if left is None or right is None:
                return left == right
            if left.val != right.val:
                return False
            return isSym(left.left, right.right) and isSym(left.right, right.left)

        if root == None:
            return True
        return isSym(root.left, root.right)

#EVEN FASTER:
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSame(self, first, second):
        if first is None or second is None:
            return first == second  # be careful about this condition

        if first.val != second.val:
            return False
        return self.isSame(first.left, second.right) and self.isSame(first.right, second.left)

    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return root is None or self.isSame(root.left, root.right)


#FASTEST

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        if root is None:
            return True

        def helper(left, right):
            if left is None and right is None:
                return True
            if left is None or right is None:
                return False

            return left.val == right.val and helper(left.left, right.right) and helper(left.right, right.left)

        return helper(root.left, root.right)