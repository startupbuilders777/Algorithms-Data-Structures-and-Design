'''
189. Rotate Array
Easy

1731

672

Favorite

Share
Given an array, rotate the array to the right by k steps, where k is non-negative.

Example 1:

Input: [1,2,3,4,5,6,7] and k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]
Example 2:

Input: [-1,-100,3,99] and k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]
Note:

Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
Could you do it in-place with O(1) extra space?
'''

# DONE!
class Solution(object):
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        '''
        
        O(1) space.
        
        Swap technique!
        
        Circular swap rite.
        
        Input: [1,2,3,4,5,6,7] and k = 3
        Output: [5,6,7,1,2,3,4]
        
        Swap 5 -> 1
        1-> 4,
        4->7, 
        7-> 3
        3->6
        6-> 2
        2-> 5 (and 5 is 0 so we are done right)
        
        We only need one circular swap not multiple.
        We can even start with index 0.
        
        
        '''
        
        # THIS IS CIRCULAR SORT BRO!!!
        
        # We have to count how many swaps we did.
        # If we fall short, we up the index of start, and try again.
        # We cant do in one circle sort!
        
        start = 0
        val = nums[start]
        
        i = start
        N = len(nums)
        swaps = 0
        
        while True:
            pivot = i + k
            
            '''
            Can do if here. Have to do while for the following case
            when k > N
            
            '''
            # Dont do this. just do mod operation.
            
            #while( pivot >= N):
            # k = pivot/N
            pivot %= N
                # so if i is 2, and k is 3, and N is 5, 
                # then i + k should be at index 0
            
            temp = nums[pivot]
            nums[pivot] = val
            val = temp
            i = pivot
            
            # we stop when we do a certain number of swaps, 
            # or when we return to our start index.
            # then we have to continue again from a new start.
            swaps += 1
            if(swaps == N):
                return 
            if pivot == start:
                # print("WE RETURNED TO START")
                # ok we are allowd to kill val because its just the start val
                # and we were suppoed to throw it away anyway
                
                # have to find a new cycle to circle sort!!
                # but we need to be able to detect when an item 
                # has not been sorted yet! find that item, and start 
                # circle sort!!!
                # OK SO the idea is we have to start circle sorting from 
                # the next element to the starter. 
                i = start + 1
                
                val = nums[start + 1]
                start += 1
                
            # print("NUMS NOW IS", nums)
            '''
            ('NUMS NOW IS', [1, 2, 3, 1, 5, 6])
            WE RETURNED TO START
            ('NUMS NOW IS', [4, 2, 3, 1, 5, 6])
            ('NUMS NOW IS', [4, 2, 3, 1, 2, 6])
            ('NUMS NOW IS', [4, 5, 3, 1, 2, 6])
            ('NUMS NOW IS', [4, 5, 3, 1, 2, 6])
            '''    
            
            
        
        return nums
        
        
        