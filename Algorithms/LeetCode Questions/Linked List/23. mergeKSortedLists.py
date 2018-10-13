'''

Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

Example:

Input:
[
  1->4->5,
  1->3->4,
  2->6
]
Output: 1->1->2->3->4->4->5->6



ALGO: do something similar to merge sort's merge function,
but instead the merge function is mergeK and merges the heads of all the lists. 

We compare the heads, then create a listNode from the smallest head/
Do the comparision quickly with a priority queue, instead of in linear time becomes => logk

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        
        from queue import PriorityQueue
        q = PriorityQueue()
        
        i = 0 # to break ties
        for l in lists:
            if(l is not None):
                q.put((l.val, i, l))
            i += 1
            
        point = ListNode(0)
        head = point
        while(not q.empty()):
            val, _, l = q.get()
            
            nextNode = l.next
            if(nextNode is not None):
                nextNodeVal = l.next.val
                q.put((nextNodeVal, i, nextNode))
                i += 1

            point.next = ListNode(val)       
            
            point = point.next
         
        return head.next

