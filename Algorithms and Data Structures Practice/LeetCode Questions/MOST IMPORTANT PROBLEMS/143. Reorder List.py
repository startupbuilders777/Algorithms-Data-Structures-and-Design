'''
143. Reorder List
Medium

1706

109

Add to List

Share
Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example 1:

Given 1->2->3->4, reorder it to 1->4->2->3.
Example 2:

Given 1->2->3->4->5, reorder it to 1->5->2->4->3.


'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        
        '''
        get length of list, 
        
        then go halfway in the list. 
        reverse the second part of the list. 
        
        Then have a pointer at start and end. 
        
        Then accumulate. 
        
        '''
        
        l =  0
        
        node = head
        while node:
            node = node.next
            l += 1
            
        '''
        Ok want to go to middle of list. 
        
        If list is even: 
        1 2 3 4
        1 4 2 3
        Start reversing at node 3 idx=2, you can even split it off.
        
        (4 + 1) // 2 ->  2
        
        odd list
        1 2 3 4 5 
        1 5 2 4 3
        start reversing at node 4 idx == 3. (5 + 1) // 2 -> 3         
        '''
        
        mid = (l + 1) // 2
        r = head
        for _ in range(mid):
            r = r.next
        
        # start reversing list. 
        prev = None
        while r:
            nxt = r.next
            r.next = prev
            prev = r
            r = nxt
            
        # build result!
        
        node = head
        reverse_node = prev
        
        while node:
            nxt = node.next
            node.next = reverse_node
            node, reverse_node = reverse_node, nxt
            
        return head
            
# You can find middle of list much more easily like so:

      
def reorderList(self, head):
    if not head:
        return
    # ensure the first part has the same or one more node
    fast, slow = head.next, head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    # reverse the second half
    p = slow.next
    slow.next = None
    node = None
    while p:
        nxt = p.next
        p.next = node
        node = p
        p = nxt
    # combine head part and node part
    p = head
    while node:
        tmp = node.next
        node.next = p.next
        p.next = node
        p = p.next.next #p = node.next
        node = tmp