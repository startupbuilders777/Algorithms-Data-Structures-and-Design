'''
86. Partition List
Medium

1157

285

Add to List

Share
Given a linked list and a value x, partition 
it such that all nodes less than x come before 
nodes greater than or equal to x.

You should preserve the original relative order 
of the nodes in each of the two partitions.

Example:

Input: head = 1->4->3->2->5->2, x = 3
Output: 1->2->2->4->3->5
'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        
        # ok so 2 root nodes. 
        # traverse, if val is less than x, append to right root node
        # otherwise append to left root node.
        # then join 2 lists and return. 
        dummy1 = ListNode(0)
        dummy2 = ListNode(0)
        
        A = dummy1
        B = dummy2
        
        node = head
        
        while node:
        
            if node.val < x:
                A.next = node
                A = A.next
                print("A val", A.val)
            else:
                B.next = node
                B = B.next
                print("B val", B.val)
                
            node = node.next
        
        B.next = None
        A.next = dummy2.next
        return dummy1.next
    
