'''
19. Remove Nth Node From End of List
Medium

3068

224

Share
Given a linked list, remove the n-th node from the end of list and return its head.

Example:

Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.
Note:

Given n will always be valid.

Follow up:

Could you do this in one pass?

Accepted
594,903
Submissions
1,701,051

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        
        # 2 pointers,
        # keep them apart
        # progress both of them until right pointer hits null,
        # perform delete operation. 
            
        dummy = ListNode(0)
        dummy.next = head
        
        left = dummy
        right = dummy

        for i in range(n+1):
            right = right.next

        while True:            
            if(right == None):
                left.next = left.next.next
                break     
            right = right.next
            left = left.next
        
        return dummy.next
    
            
'''
Value-Shifting - AC in 64 ms

My first solution is "cheating" a little. 
nstead of really removing the nth node, I remove the nth value. 
I recursively determine the indexes (counting from back), then 
shift the values for all indexes larger than n, 
and then always drop the head.
'''        

class Solution:
    def removeNthFromEnd(self, head, n):
        def index(node):
            if not node:
                return 0
            i = index(node.next) + 1
            if i > n:
                node.next.val = node.val
            return i
        index(head)
        return head.next

'''
Index and Remove - AC in 56 ms

In this solution I recursively determine the indexes again, 
but this time my helper function removes the nth node. It 
returns two values. The index, as in my first solution, 
and the possibly changed head of the remaining list.
'''

class Solution:
    def removeNthFromEnd(self, head, n):
        def remove(head):
            if not head:
                return 0, head
            i, head.next = remove(head.next)
            return i+1, (head, head.next)[i+1 == n]
        return remove(head)[1]
    
# Your solution but a bit diff

class Solution(object):
    def removeNthFromEnd(self, head, n):
        if n == 0 or head is None:
            return head
        dummy = ListNode(-1)
        dummy.next = head
        fast = slow = dummy
        k = n
        while fast and k > 0:
            fast = fast.next
            k -= 1
        
        while fast and fast.next:
            fast = fast.next
            slow = slow.next
            
        if slow.next:
            slow.next = slow.next.next
            
        return dummy.next
    


