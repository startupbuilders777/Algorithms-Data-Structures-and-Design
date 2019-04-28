# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
            
    def removeElementsIterative(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        
        if( head is None):
            return None
        
        prev = None
        
        
        # Find first element that isnt val:
        
        while(head and head.val == val):
            head = head.next
        
        if(head is None):
            return None
        
        start = head
        prev = head
        head = head.next

        while(head):
            if(head.val == val):
                # set prev to head.next
                prev.next = head.next
                head = head.next         
            else:
                prev = head
                head = head.next
        
        return start
    
    
    # RECURSIVE
    def removeElements(self, head, val):
        
        def recur(head):
            if(head is None):
                return None

            if(head.val == val):
                return recur(head.next)
            else:
                head.next = recur(head.next)
                return head
            
        return recur(head)
                    
            
            