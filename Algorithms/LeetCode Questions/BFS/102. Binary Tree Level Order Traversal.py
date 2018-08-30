'''
Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

For example:

Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]

By the way

1 in the first level
2 in the second level
4 in the third level
8 in the fourth level
...
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        Note that if you are trying to do that operation often, especially in loops, a list is the wrong data structure.
        Lists are not optimized for modifications at the front, and somelist.insert(0, something) is an O(n) operation.
        somelist.pop(0) and del somelist[0] are also O(n) operations.
        The correct data structure to use is a deque from the collections module. deques expose an interface that is similar to those of lists, but are o       ptimized for modifications from both endpoints. They have an appendleft method for insertions at the front.
        
        """
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class StackNode():
    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class Stack():
    def __init__(self):
        self.topNode = None
        self.count = 0

    def push(self, data):
        self.count += 1
        newNode = StackNode(data)
        if (self.topNode is None):
            self.topNode = newNode
        else:
            self.topNode.next = newNode
            newNode.prev = self.topNode
            self.topNode = newNode

    def top(self):
        if self.topNode is None:
            return None
        else:
            return self.topNode.data

    def size(self):
        return self.count

    def pop(self):
        if self.topNode is None:
            return ReferenceError
        else:
            self.count -= 1
            prev = self.topNode.prev
            self.topNode = prev

    def empty(self):
        return self.topNode is None


class Queue():
    '''
    Invarient :
    Front will always contain elements if the back contains elements and peeking at front is always O(1)
    '''

    def __init__(self):
        self.front = Stack()
        self.back = Stack()

    def size(self):
        return self.front.size() + self.back.size()

    def add(self, item):
        self.back.push(item)
        if (self.front.empty()):
            self.transfer()

    def top(self):
        return self.front.top()

    def empty(self):
        return self.front.empty()

    def remove(self):
        if (self.empty()):
            return Exception
        self.front.pop()
        if (self.front.empty()):
            self.transfer()

    def transfer(self):
        while not self.back.empty():
            self.front.push(self.back.top())
            self.back.pop()


class Solution(object):
    def levelOrder(self, root):

        if root is None:
            return []

        visited, queue = set(), Queue()
        result = []
        queue.add((root, 0))

        while queue.size() != 0:
            (node, depth) = queue.top()
            queue.remove()
            if (node is None):
                continue
            else:
                arr = result[depth] if depth < len(result) else None
                if (arr is None):
                    result.append([node.val])
                else:
                    arr.append(node.val)
            if node not in visited:
                visited.add(node)
                if (node is None):
                    continue
                if (node.left is not None):
                    queue.add((node.left, depth + 1))
                elif (node.left is None):
                    queue.add((None, depth + 1))

                if (node.right is not None):
                    queue.add((node.right, depth + 1))
                elif (node.right is not None):
                    queue.add((None, depth + 1))

        return result


class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []

        result = []
        cur_level = [root]

        while cur_level:
            tmp1, tmp2 = [], []
            for node in cur_level:
                tmp1.append(node.val)
                if node.left:
                    tmp2.append(node.left)
                if node.right:
                    tmp2.append(node.right)
            result.append(tmp1[:])
            cur_level = tmp2

        return result


##FASTEST

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """

        if not root:
            return []
        res = []
        level = [root]
        while level:
            res.append([n.val for n in level])

            tmp = []
            for n in level:
                tmp.append(n.left)
                tmp.append(n.right)
            level = [leaf for leaf in tmp if leaf]
        return res
