'''
148. Sort List
Medium

2591

127

Add to List

Share
Sort a linked list in O(n log n) time using constant space complexity.

Example 1:

Input: 4->2->1->3
Output: 1->2->3->4
Example 2:

Input: -1->5->3->4->0
Output: -1->0->3->4->5

'''

# MY QUICK SORT. IT DOES TLE BUT IT WORKS FOR ALL INPUTS!

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        '''
        random.randint(a, b)¶
        Return a random integer N such that a <= N <= b. Alias for randrange(a, b+1).
        '''
        
        node = head
        l = 0
        while node:
            l += 1
            node = node.next
            
        def partition(head, start, end, pivot):
            if start == end:
                return head
            if head is None:
                return None
            
            pivotVal = pivot.val
            before = ListNode(0)
            after = ListNode(0)
            afterCopy = after
            beforeCopy = before
            
            temp = head
            left_len = 0
            
            while temp:       
                # print("processing temp", temp.val)
                if temp == pivot:
                    temp = temp.next
                    continue
                    
                if temp.val < pivotVal: 
                    left_len += 1
                    before.next = temp
                    before = before.next
                    temp = temp.next
                else:
                    after.next = temp
                    after = after.next
                    temp = temp.next
                    
            before.next = None
            after.next = None
            return beforeCopy.next, left_len, afterCopy.next
 
        def quicksort(head, start, end):
            if head is None:
                return None
            
            if end-start <= 1:
                return head 
            
            pivotLoc = random.randint(start, end-1)            
            pivot = head
            i = 0
            while i < pivotLoc:
                pivot = pivot.next
                i += 1
                
            if pivot is None:
                return None
               
            left, left_len, right = partition(head, start, end, pivot) 
            sorted_left = quicksort(left, 0, left_len)
            sorted_right = quicksort(right, 0, end - left_len - 1)
            
            if sorted_left:
                temp = sorted_left
                while temp and temp.next:
                    temp = temp.next
                temp.next = pivot
            else:
                sorted_left = pivot
            
            pivot.next = sorted_right
            return sorted_left
        
        return quicksort(head, 0, l)