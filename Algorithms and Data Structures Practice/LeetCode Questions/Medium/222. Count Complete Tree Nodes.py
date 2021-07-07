'''
Given a complete binary tree, count the number of nodes.

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the 
last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def countNodes(self, root):
        '''

        This way is too slow
        sum = [0]

        def countNodesAcc(root, sum):
            if(root is None):
                return 
            else:
                sum[0] += 1
                countNodesAcc(root.left, sum)
                countNodesAcc(root.right, sum)

        countNodesAcc(root, sum) 
        return sum[0]
        '''

        '''
        Proper way:
        Basically my solution contains 2 steps.
        (1) Firstly, we need to find the height of the binary tree and count the nodes above the last level.
        (2) Then we should find a way to count the nodes on the last level.
        '''

        def getHeightLeft(node):
            if (node is None):
                return 0
            return 1 + getHeightLeft(node.left)

        def getHeightRight(node):
            if (node is None):
                return 0
            return 1 + getHeightRight(node.right)

        hl = getHeightLeft(root)
        hr = getHeightRight(root)

        if hl == hr:
            return 2 ** hl - 1
        else:
            return self.countNodes(root.left) + self.countNodes(root.right) + 1




