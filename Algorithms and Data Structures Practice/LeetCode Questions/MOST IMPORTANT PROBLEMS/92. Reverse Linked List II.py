'''
92. Reverse Linked List II
Medium

2137

138

Add to List

Share
Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.

Example:

Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL
'''

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        
        node = head
        startHandle = None
        # We need the node rite before the start of reversal
        # so we can attach the reversed list. 
        # also need an end handle. 
        
        for i in range(m-1):
            startHandle = node
            node = node.next
            
        '''
        you have to reverse it, 
        then connect it back to where it should be. 
        '''
        
        # We connect the reverse
        prev = None # endHandle
        connectToEndHandle = node 
        # saving this location so we can connect it to the end handle. 
        
        for i in range(n - m):
            print("node value", node.val)
            nxt = node.next
            node.next = prev 
            prev = node
            node = nxt
         
        nxt = node.next
        node.next = prev
        connectToEndHandle.next = nxt
        
        if startHandle: 
            startHandle.next = node
        else: 
            # the new head is node?
            head = node
            
        return head

# UNDERSTAND LEETCODE OFFICIAL 2 POINTER RECURSIVE SOLUTION:
class Solution2:
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """

        if not head:
            return None

        left, right = head, head
        stop = False
        def recurseAndReverse(right, m, n):
            nonlocal left, stop

            # base case. Don't proceed any further
            if n == 1:
                return

            # Keep moving the right pointer one step forward until (n == 1)
            right = right.next

            # Keep moving left pointer to the right until we reach the proper node
            # from where the reversal is to start.
            if m > 1:
                left = left.next

            # Recurse with m and n reduced.
            recurseAndReverse(right, m - 1, n - 1)

            # In case both the pointers cross each other or become equal, we
            # stop i.e. don't swap data any further. We are done reversing at this
            # point.
            if left == right or right.next == left:
                stop = True

            # Until the boolean stop is false, swap data between the two pointers     
            if not stop:
                left.val, right.val = right.val, left.val

                # Move left one step to the right.
                # The right pointer moves one step back via backtracking.
                left = left.next           

        recurseAndReverse(right, m, n)
        return head



class Solution:
    '''
    When we are at the line pre.next.next = cur 
    the LL looks like this for [1,2,3,4,5] m = 2, n = 4

    1 -> 2 <- 3 <- 4 5

    Note that there is no connection between 4 and 5, 
    here pre is node 1, reverse is node 4, cur is node 5; 
    So pre.next.next = cur is basically linking 2 with 5; 
    pre.next = reverse links node 1 with node 4.
    '''
    
    def reverseBetween(self, head, m, n):
        if m == n:
            return head
        p = dummy = ListNode(0)
        dummy.next = head
        for _ in range(m - 1):
            p = p.next
        cur = p.next
        pre = None
        for _ in range(n - m + 1):
            cur.next, pre, cur = pre, cur, cur.next
        p.next.next = cur
        p.next = pre
        return dummy.next


#  Another way:
#  This requires a drawing because there are double jumps, and buckling 
#  when connecting the linked lists

class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head or m == n: return head
        p = dummy = ListNode(None)
        dummy.next = head
        for i in range(m-1): p = p.next
        tail = p.next

        for i in range(n-m):
            tmp = p.next                  # a)
            p.next = tail.next            # b)
            tail.next = tail.next.next    # c)
            p.next.next = tmp             # d)
        return dummy.next



## FASTEST SOLUTION:

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

#Recursive Implemetation
class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        
        if not head:
            return head
        L=head
        R=head
        
        stop=False
        
        def recursive(m,n,L,R,stop):
            
            
            if (n==1):
                return L,stop
            R=R.next
            if m>1:
                L=L.next
                
            L, stop = recursive(m-1,n-1,L,R,stop)
           
            if L==R or R.next ==L:
                stop = True
            
            if not stop:
                L.val,R.val = R.val,L.val
                
                L=L.next
            return L, stop 
           
        recursive(m,n,L,R,stop)
        return head


## ALSO FASTER THAN YOURS:

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution(object):
    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if not head:
            return None
        
        dummy_head = ListNode(None)
        dummy_head.next = head
        
        prev = dummy_head
        
        for _ in range(m - 1):
            prev = prev.next
        
        curr = prev.next
        for _ in range(n - m):
            next = curr.next
            curr.next = next.next
            next.next = prev.next
            prev.next = next
            
        
        return dummy_head.next
    

