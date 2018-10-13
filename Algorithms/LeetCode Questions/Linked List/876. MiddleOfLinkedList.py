# COMPLETED


'''
876. Middle of the Linked List
Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.

 

Example 1:

Input: [1,2,3,4,5]
Output: Node 3 from this list (Serialization: [3,4,5])
The returned node has value 3.  (The judge's serialization of this node is [3,4,5]).
Note that we returned a ListNode object ans, such that:
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, and ans.next.next.next = NULL.
Example 2:

Input: [1,2,3,4,5,6]
Output: Node 4 from this list (Serialization: [4,5,6])
Since the list has two middle nodes with values 3 and 4, we return the second one.
 

Note:

The number of nodes in the given list will be between 1 and 100.


'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        numOfNodes = 0
        node = head
        while True:
            node = node.next
            numOfNodes += 1
            if(node is None):
                break
        
        node = head
        middle = int((numOfNodes /2))
                     
        for i in range(0, middle):
            node = node.next
        
        return node
            
            
## BETTER SOLUTION IS FAST AND SLOW POINTER!!!!!!!!!!!!!!!!!!!!!!

'''
When traversing the list with a pointer slow, make another pointer fast that traverses twice as fast. When fast reaches the end of the list, slow must be in the middle.
'''


class Solution(object):
    def middleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

''' Fastest solution:

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        '''
        slow = fast = head
  # iterate over the list and check if the fast becomes null
  # the moment the fast becomes null, the slow points to the middle of the list
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            
        return slow
        '''
        lst =[]
        while head:
            lst.append(head.val)
            head = head.next
        return lst[int(len(lst)/2):]

    
