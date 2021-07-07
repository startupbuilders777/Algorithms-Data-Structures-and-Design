
# EASY: 
# Write a program to find the node at which the intersection of two singly linked lists begins.


'''
Find the different in 2 lists, then traverse longer one 
shifted by difference, and other, one node at a time.
When nodes are equal that is the intersection node. 
'''

class Solution:
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
    
    
# Other solution

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

    # SET SOLUTION IS TRIVIAL
    
    # Solution 2. PLEASE DO IN O(1) SPACE. DONT USE A SET. 
    # USE 2 POINTERS. 
    
    # Basically traverse each list one node at a time. 
    # Maintain two pointers pApA and pBpB initialized at the head of A and B, 
    # respectively. Then let        
    # them both traverse through the lists, one node at a time.
    # When pApA reaches the end of a list, then redirect it to the head of B (yes, B, that's right.);   
    # similarly when pBpB reaches the end of a list, redirect it the head of A.
    #If at any point pApA meets pBpB, then pApA/pBpB is the intersection node.
    

class Solution:
    # @param two ListNodes
    # @return the intersected ListNode
    def getIntersectionNode(self, headA, headB):
        if headA is None or headB is None:
            return None

        pa = headA # 2 pointers
        pb = headB

        while pa is not pb:
            # if either pointer hits the end, switch head and continue the second traversal, 
            # if not hit the end, just move on to next
            pa = headB if pa is None else pa.next
            pb = headA if pb is None else pb.next

        return pa # only 2 ways to get out of the loop, they meet or the both hit the end=None

# the idea is if you switch head, the possible difference between length would be countered. 
# On the second traversal, they either hit or miss. 
# if they meet, pa or pb would be the node we are looking for, 
# if they didn't meet, they will hit the end at the same iteration, pa == pb == None, return either one of them is the same,None




