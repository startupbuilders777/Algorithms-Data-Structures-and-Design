# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

a = ListNode(3)
b = ListNode(2)
c = ListNode(1)
d = ListNode(6)
e = ListNode(7)

a.next = b
b.next = c
c.next = d
d.next = e


class Solution:
    def reverseListIter(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        curr = head
        prev = None
        next = None
        
        while True:
            if(curr == None):
                break
                
            next = curr.next
          
            curr.next = prev
            if(prev is not None):
                prev.next = curr
            
            curr = next
            prev = curr
        
        
                
        printList(prev)
        return prev

    def reverseListRecursive(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def recurse(head):
            if(head.next == None):
                # this head will be the head of the reversed list. 
                return (head, head)


            (reversedRest, newHead) = recurse(head.next)
        
            reversedRest.next = head
            return (reversedRest, newHead)

        return newHead


def printList(node):
    while(node is not None):
        print(node.val)
        node = node.next

solution = Solution()

printList(solution.reverseListRecursive(a))
printList(solution.reverseListIter(a)[1])

