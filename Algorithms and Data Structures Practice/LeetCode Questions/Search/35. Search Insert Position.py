'''
35. Search Insert Position

Given a sorted array and a target value, return the index if the target is found. 
If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Example 1:

Input: [1,3,5,6], 5
Output: 2
Example 2:

Input: [1,3,5,6], 2
Output: 1
Example 3:

Input: [1,3,5,6], 7
Output: 4
Example 4:

Input: [1,3,5,6], 0
Output: 0

'''


class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        
        l = 0
        r = len(nums)
        mid = None
        
        while l != r:            
            mid = l + (r-l)//2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid
        # DO NOT RETURN MID, RETURN L
        # REMEMBEBR THAT!!! 
        # THE DIFFERENCE IS RETURNING MID WORKS
        # UNTIL IT DOESNT, BECAUSE WE WANT THE 
        # INSERTION POINT TO BE IN THE LEFT
        return l


# THE OTHER WAY:

def searchInsert(self, nums, target):
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                low = mid + 1
            else:
                high = mid - 1
        return low