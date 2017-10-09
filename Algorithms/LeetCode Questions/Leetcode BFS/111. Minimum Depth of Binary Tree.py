'''
Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

USE BFS
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


class Queue():
    def __init__(self):
        self.lst = []

    def enqueue(self, item):
        self.lst.append(item)

    def dequeue(self):
        if (len(self.lst) == 0):
            return None
        item = self.lst.pop(0)
        return item

    def length(self):
        return len(self.lst)

    def top(self):
        if (len(self.lst) == 0):
            return None
        else:
            return lst.get(len(lst) - 1)

    def isEmpty(self):
        return len(self.lst) == 0

    def printAll(self):
        for i in self.lst:
            print(str(i[0].val) + " " + str(i[1]))


class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        '''
        BFS is fastest way and you terminate search 
        on a path if it gets longer than the other paths
        '''

        queue = Queue()
        visited = set()

        if (root is None):
            return 0;

        queue.enqueue((root, 1))

        while (not queue.isEmpty()):
            queue.printAll()
            (node, currDepth) = queue.dequeue()
            if node not in visited:
                visited.add(node)

                print("VISITED THE NODE" + str(node.val))
                left = node.left
                right = node.right
                if (left is not None and right is not None):
                    queue.enqueue((left, currDepth + 1))
                    queue.enqueue((right, currDepth + 1))
                    continue
                elif (left is None and right is None):
                    return currDepth
                    continue
                elif (left is None):
                    queue.enqueue((right, currDepth + 1))
                    continue
                elif (right is None):
                    queue.enqueue((left, currDepth + 1))
                    continue
                else:
                    print("THis case shouldnt exist")


'''

FASTER
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0

        left_min = self.minDepth(root.left)
        right_min = self.minDepth(root.right)

        if left_min and right_min:
            return min(left_min, right_min) + 1
        elif left_min == 0:
            return right_min + 1
        elif right_min == 0:
            return left_min + 1

'''
FASTEST SOLUTION
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        if root is None: return 0

        queue = [root]
        depth = 1
        minimum = float('inf')

        while True:
            tmp = []
            while queue:
                node = queue.pop(0)
                if node.left is None and node.right is None:
                    return depth
                if node.left is not None: tmp.append(node.left)
                if node.right is not None: tmp.append(node.right)
            if tmp:
                depth += 1
                queue = tmp
                # else:
                #     return minimum