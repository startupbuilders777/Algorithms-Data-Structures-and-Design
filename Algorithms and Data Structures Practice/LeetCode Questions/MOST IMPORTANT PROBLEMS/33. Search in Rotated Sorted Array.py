'''
33. Search in Rotated Sorted Array
Medium

5033

461

Add to List

Share
Suppose an array sorted in ascending order is 
rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
Example 2:

Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

'''

# MY ACCEPTED SOLUTION

class Solution:
    def search(self, nums: List[int], target: int) -> int:
        
        '''
        Rotated array 
        
        [4, 5, 6, 7, 0, 1, 2]
        
        target bigger than left, number. 
        it should be between [left, pivot]
        target smaller than left number. 
        should be betwe [pivot, right]
        
        if left number is smaller than right number 
            -> the array sequence is sorted -> do normal bin search in it. 
        
        if left number bigger than right    
            we have following situation:
            -> left ____ pivot ____ right. 
        '''
        i = 0
        j = len(nums) - 1

        while i <= j:
            # print("i and j", (i, j))
            left = nums[i]
            mid_idx = i + (j-i)//2
            
            # print("MID IDX IS", mid_idx)
            
            middle = nums[mid_idx]
            
            if middle == target:
                return mid_idx
            elif left <= middle :
                # SORTED ARRAY!
                # elements after middle can still be larger than left!
                # print("LEFT IS LESS THAN MIDDLE")
                if target > middle:
                    # search right side. 
                    i = mid_idx + 1
                elif target >= left and target < middle:
                    j = mid_idx - 1
                else: # target smaller than left implies its smaller than middle
                    # search right side for the magical pivot that contains small elemetns
                    i = mid_idx + 1
                    
            elif left > middle:
                # print("LEFT IS GREATER THAN MIDDLE")
                # NOT SORTED ARRAY
                # pivot is between left and middle.
                # but everything after middle is sorted!!
                if target >= left: # IMPLYING bigger than middle!
                    # search left side. 
                    # only place it could be  
                    j = mid_idx - 1
                elif target < left and target < middle:
                    # we need to access the pivot because it contains the smalllest elements
                    # we need!
                    j = mid_idx - 1
                else: 
                    # target < left and target > middle
                    # target wont match elements after leeft because the bigger, 
                    # and target wont match pivot up to middle because they are small
                    # move to the other side!
                    i = mid_idx + 1
                    
                    # search rigth side. 
            # print("i and j after", (i, j))
            
        return -1

