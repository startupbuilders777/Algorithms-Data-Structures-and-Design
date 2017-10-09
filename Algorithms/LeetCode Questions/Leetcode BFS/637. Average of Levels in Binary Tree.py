'''
Given a non-empty binary tree, return the average value of the nodes on each level in the form of an array.

Example 1:
Input:
    3
   / \
  9  20
    /  \
   15   7
Output: [3, 14.5, 11]
Explanation:
The average value of nodes on level 0 is 3,  on level 1 is 14.5, and on level 2 is 11. Hence return [3, 14.5, 11].
Note:
The range of node's value is in the range of 32-bit signed integer.


'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """

        depth = {}

        def averageOfLevelsRecur(node, height):
            if (node is None):
                return

            if depth.get(height) is None:
                depth[height] = [node.val]
            else:
                depth[height].append(node.val)

            averageOfLevelsRecur(node.right, height + 1)
            averageOfLevelsRecur(node.left, height + 1)

        averageOfLevelsRecur(root, 0)
        result = []

        for i in sorted(depth.keys()):
            result.append(sum(depth[i]) / float(len(depth[i])))

        return result

#### SLIGHTLY FASTER:

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        q = []
        list_of_levels = []

        if root is None:
            return q
        else:
            q.append(root)

        while len(q) > 0:
            list_of_levels.append(q)
            new_level = []
            for item in q:
                if item.left:
                    new_level.append(item.left)
                if item.right:
                    new_level.append(item.right)
            q = new_level

        list_of_avgs = []
        for lst in list_of_levels:
            sum = 0
            for item in lst:
                sum += item.val
            list_of_avgs.append(float(sum) / len(lst))

        return list_of_avgs

#### FASTER BFS HOLY MOLY

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

import collections

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        q = collections.deque()
        q.append(root)
        res = []
        while q:
            levelNum = len(q)
            sumNum = 0
            for i in range(levelNum):
                node = q.popleft()
                sumNum += node.val
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            res.append(sumNum / (levelNum * 1.0))
        return res
