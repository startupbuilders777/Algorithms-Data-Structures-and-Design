# Done

'''
You are given two non-empty linked lists representing two non-negative integers. The digits are stored 
in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
'''


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        
        def nodeSum(l1, l2, carriedTheOne):
            leftVal = 0 if l1 is None else l1.val
            rightVal = 0 if l2 is None else l2.val
            sum = leftVal + rightVal 
            
            
            sum += carriedTheOne
            
            digit = sum % 10
            carryOne = 1 if sum > 9 else 0
            
            return (ListNode(digit), carryOne)
        
        # ok we have to recurse to the end of each linked list, grab the value, 
        # add, and return. we also need to carry the one if we need to!
        def addTwoNumbersRecur(l1, l2, carriedTheOne):
            if(l1 is None and l2 is None and  not carriedTheOne):
                return None
            elif(l1 is None and l2 is None and carriedTheOne):
                return ListNode(1)

            (sumDigit, carryTheOne) =  nodeSum(l1, l2, carriedTheOne)

            if(l1 is not None and l2 is not None): 
                sumDigit.next = addTwoNumbersRecur(l1.next, l2.next, carryTheOne)
            elif(l1 is not None):
                sumDigit.next = addTwoNumbersRecur(l1.next, l2, carryTheOne)
            elif(l2 is not None):    
                sumDigit.next = addTwoNumbersRecur(l1, l2.next, carryTheOne)
                
            return sumDigit
            
        
        return addTwoNumbersRecur(l1, l2, False)



##

