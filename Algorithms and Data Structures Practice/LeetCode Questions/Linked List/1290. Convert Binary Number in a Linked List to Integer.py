'''
1290. Convert Binary Number in a Linked List to Integer
Easy

346

32

Add to List

Share
Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

 

Example 1:


Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10
Example 2:

Input: head = [0]
Output: 0
Example 3:

Input: head = [1]
Output: 1
Example 4:

Input: head = [1,0,0,1,0,0,1,1,1,0,0,0,0,0,0]
Output: 18880
Example 5:

Input: head = [0,0]
Output: 0
 

Constraints:

The Linked List is not empty.
Number of nodes will not exceed 30.
Each node's value is either 0 or 1.

'''


# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def getDecimalValue(self, head):
        """
        :type head: ListNode
        :rtype: int
        """
        
        l = 0
        n = head
        while n:
            l += 1
            n = n.next
           
        v = 0
        
        n = head
        counter = 0
        
        ## Actually u have to go backwards!!
        
        while n:
            
            v += (n.val)*(2**(l-counter - 1))
            n = n.next
            counter += 1    
            
        return v

# Faster?

class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        i=head
        s=""
        while i:
            s+=str(i.val)
            i=i.next
        return int(s,2)
    
# Python generator:

class Solution(object):
    def yield_content(self, head):
        current = head
        yield current.val
        while current.next != None:
            current = current.next
            yield current.val

    def getDecimalValue(self, head):
        bin_number = ''
        generator = self.yield_content(head)
        while True:
            try:
                bin_number += str(next(generator))
            except StopIteration:
                break
        return int(bin_number, 2)
