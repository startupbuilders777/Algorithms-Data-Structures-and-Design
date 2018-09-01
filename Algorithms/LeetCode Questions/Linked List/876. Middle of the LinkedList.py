# DONE

'''
Given a non-empty, singly linked list with head node head, return a middle node of linked list.

If there are two middle nodes, return the second middle node.

 

Example 1:

Input: [1,2,3,4,5]
Output: Node 3 from this list (Serialization: [3,4,5])
The returned node has value 3.  (The judge's serialization of this node is [3,4,5]).
Note that we returned a ListNode object ans, such that:
ans.val = 3, ans.next.val = 4, ans.next.next.val = 5, and ans.next.next.next = NULL.
Example 2:

Input: [1,2,3,4,5,6]
Output: Node 4 from this list (Serialization: [4,5,6])
Since the list has two middle nodes with values 3 and 4, we return the second one.
 

Note:

The number of nodes in the given list will be between 1 and 100.


'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def middleNode(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # you could get length of the list, then traverse the list again to grab it
        # or you can store elements in map, and then index for it. (so O(n) vs O(2n))
        # 
        
        node = head
        len = 0
        dict = {}
        while True:
            if(node is None):
                break
            else:
                dict[len] = node.val
                node = node.next
                len += 1
        
        if(len % 2 == 0):
            return [dict[i] for i in range(int(len/2), len)]
        else: 
            return [dict[i] for i in range(int((len-1)/ 2), len)]



# Approach 2: Fast and Slow Pointer

# When traversing the list with a pointer slow, make another pointer fast 
# that traverses twice as fast. When fast reaches the end of the list, slow must be in the middle.

class Solution(object):
    def middleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

