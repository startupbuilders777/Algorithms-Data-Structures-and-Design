# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    # Solution 1: Do we set.
    def getIntersectionNodeSlow(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        
        
        seen = set()
        while headA is not None:
            seen.add(headA)
            headA = headA.next
            
        # Mark all the nodes for one list, 
        # then traverse other and see if we intersect
        # not smart.
        while headB is not None:
            if(headB in seen):
                return headB
            else:
                headB = headB.next

    # Solution 2. PLEASE DO IN O(1) SPACE. DONT USE A SET. 
    # USE 2 POINTERS. 
    
    # Basically traverse each list one node at a time. 
    # Maintain two pointers pApA and pBpB initialized at the head of A and B, 
    # respectively. Then let        
    # them both traverse through the lists, one node at a time.
    # When pApA reaches the end of a list, then redirect it to the head of B (yes, B, that's right.);   
    # similarly when pBpB reaches the end of a list, redirect it the head of A.
    #If at any point pApA meets pBpB, then pApA/pBpB is the intersection node.
    
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        la = self.length(headA)
        lb = self.length(headB)
        curA = headA
        curB = headB
        while(la>lb):
            curA = curA.next
            la -= 1
        while(lb>la):
            curB = curB.next
            lb -= 1
        while(curA!=curB):
            curA = curA.next
            curB = curB.next
        return curA
    def length(self,head):
        l = 0
        while head:
            l += 1
            head = head.next
        return l