'''
Given a binary tree, return the zigzag level order traversal of its nodes' values. 
(ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]

'''

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


import Queue


class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        visited, queue = set(), Queue.Queue()
        queue.put((root, 0))
        alternate = False
        result = []
        while queue.qsize() != 0:
            (node, depth) = queue.get()

            if (node is not None):
                print("node val visited" + str(node.val))
                print(depth)
                arr = result[depth] if depth < len(result) else None
                if (arr is None):
                    result.append([node.val])
                else:
                    arr.append(node.val)
                depth += 1
                if (node.left is not None):
                    print("put left")
                    queue.put((node.left, depth))
                if (node.right is not None):
                    print("put right")
                    queue.put((node.right, depth))

        for i in range(0, len(result)):
            if (i % 2 == 0):
                continue
            else:
                result[i] = result[i][::-1]

        return result


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        q = [root]
        k = 0
        output = []
        while q:
            k += 1
            outputi = []
            p = []
            for item in q:
                outputi.append(item.val)
                if item.left:
                    p.append(item.left)
                if item.right:
                    p.append(item.right)
            if (k % 2) == 0:
                output.append(list(reversed(outputi)))
            else:
                output.append(outputi)
            q = p[:]
        return output

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        level = [root]
        ans = []
        flag = True
        while len(level) > 0:
            res = [x.val for x in level]
            ans.append(res)
            newLevel = []
            if flag:
                for i in range(len(level)-1,-1,-1):
                    if level[i].right:
                        newLevel.append(level[i].right)
                    if level[i].left:
                        newLevel.append(level[i].left)
            else:
                for i in range(len(level)-1,-1,-1):
                    if level[i].left:
                        newLevel.append(level[i].left)
                    if level[i].right:
                        newLevel.append(level[i].right)
            level = newLevel
            flag = not flag
        return ans