'''
Given a sorted linked list, delete all duplicates such that each element appear only once.

For example,
Given 1->1->2, return 1->2.
Given 1->1->2->3->3, return 1->2->3.

'''


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        if (head is None):
            return None

        def deleteDupsRecur(node, prevNodeVal):
            if (node is None):
                return None
            elif (node.val == prevNodeVal):
                return deleteDupsRecur(node.next, prevNodeVal)
            else:
                node.next = deleteDupsRecur(node.next, node.val)
                return node

        head.next = deleteDupsRecur(head.next, head.val)
        return head


