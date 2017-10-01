'''
The thief has found himself a new place for his thievery again. There is only one entrance to this area, 
called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart 
thief realized that "all houses in this place forms a binary tree". It will automatically contact the police 
if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:
     3
    / \
   2   3
    \   \ 
     3   1
Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
Example 2:
     3
    / \
   4   5
  / \   \ 
 1   3   1
Maximum amount of money the thief can rob = 4 + 5 = 9.
Credits:
Special thanks to @dietpepsi for adding this problem and creating all test cases.

'''


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rob(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        # Is this dynamic programming?

        # At each node 2 decisions, Steal, Dont Steal
        # Unless the previous node was steal, then this node cant steal
        # Then next node have decision again, Steal or Dont steal
        dict = {}

        def steal(node, stoleInPreviouseNode, dict):
            if (node is None):
                return 0
            if (stoleInPreviouseNode == True):
                # Cant steal on this node
                stealLeft = steal(node.left, False, dict)
                stealRight = steal(node.right, False, dict)
                totalSteal = stealLeft + stealRight
                return totalSteal
            if (stoleInPreviouseNode == False):
                # Check to see if answer was memoized from before

                memoizedValue = dict.get(node)
                if (memoizedValue is not None):
                    return memoizedValue
                # Otherwise its none
                # We can either Steal from this node or dont steal and get loot from other nodes
                # take the max of these two situations and return that

                # Steal this node
                totalIfWeStealInThisNode = node.val + steal(node.left, True, dict) + steal(node.right, True, dict)
                # Dont steal
                totalIfWeDontStealInThisNode = steal(node.left, False, dict) + steal(node.right, False, dict)
                dict[node] = max(totalIfWeStealInThisNode, totalIfWeDontStealInThisNode)
                return dict[node]

        return steal(root, False, dict)

