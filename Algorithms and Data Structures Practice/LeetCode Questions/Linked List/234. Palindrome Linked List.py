'''

234. Palindrome Linked List
Easy

2887

343

Add to List

Share
Given a singly linked list, determine if it is a palindrome.

Example 1:

Input: 1->2
Output: false
Example 2:

Input: 1->2->2->1
Output: true
Follow up:
Could you do it in O(n) time and O(1) space?

Accepted
400,344
Submissions
1,034,565
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        fast = slow = head
        # find the mid node
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        # reverse the second half
        node = None
        while slow:
            nxt = slow.next
            slow.next = node
            node = slow
            slow = nxt
        # compare the first and second half nodes
        while node: # while node and head:
            if node.val != head.val:
                return False
            node = node.next
            head = head.next
        return True


# FASTEST SOLUTION:
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        backStack = []
        
        beginNode = head
        while beginNode != None:
            backStack.append(beginNode)
            beginNode = beginNode.next
        
        nodeA = head
        nodeB = None
        while (nodeA != None):
            nodeB = backStack.pop()
            if (nodeA.val != nodeB.val):
                return False
            nodeA = nodeA.next
        
        return True
    
    




# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        
        if(head == None):
            return True
        
        n = head
        l = 0
        
        while n:
            n = n.next
            l += 1
        
        lp = head
        rp = head    
        
        rpCounter = (l+1)//2
        lpCounter = (l//2 -1)
        left_counter = 0
        
        for i in range(rpCounter):
            rp = rp.next
            
        def check_palin(lp): 
            # We only need these 2 as nonlocals. Why?
            # Also we cant use thse as arguments to 
            # function call. Why?
            nonlocal rp 
            nonlocal left_counter
            # nonlocal lpCounter
                        
            if (left_counter < lpCounter):
                left_counter += 1
                result = check_palin(lp.next)
                if result == False:
                    return False
            
            if(rp == None):
                return True
            
            if(rp.val == lp.val):
                rp = rp.next # check next rp. 
                return True # needed when there are only 2 nodes in linked list. 
            else:
                return False
        return check_palin(lp)
            