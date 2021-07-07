'''
75. Sort Colors
Medium

3352

223

Add to List

Share
Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:

Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
Follow up:

A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed by 2's.
Could you come up with a one-pass algorithm using only constant space?
'''

# MY CONSTANT SPACE 3 POINTER ALGORITHM

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        '''
        Use 3 pointers,
        the first and last one delimit the correct sequence of 0s and 2s on the left and right side.
        ALWAYS TRAVERSE THE MIDDLE POINTER ACROSS, AND SWAP INTO CORRRECT POSITIONS. WHEN YOU SEE A 1, CONTINUE. 
        
        '''
        i = 0
        l = 0         
        j = len(nums) - 1
        
        while l != len(nums):
            if nums[l] == 1:
                l += 1
            elif nums[l] == 0 and i == l:
                l += 1
            elif nums[l] == 0:
                nums[l], nums[i] = nums[i], nums[l]
                i += 1 
            elif nums[l] == 2 and l >= j:
                l += 1
            elif nums[l] == 2:
                nums[l], nums[j] = nums[j], nums[l]
                j -= 1
        return nums
    

# CLEANER SOLUTION, DUTCH PARTITION PROBLEM, COLORED POINTERS, AKA 3 POINTERS

'''
Then why you can increase white by 1 in the first case, 
how can you be sure the one you get by swapping is 1?

WHY CAN WE SIMPLIFY IT THAT MUCH. 
SPELL OUT THE INVARIANTS AKA ABUSE: 

nums[0:red] = 0, nums[red:white] = 1, nums[white:blue + 1] = unclassified, nums[blue + 1:] = 2.
The code is written so that either (red < white and nums[red] == 1) or (red == white) at all times.
Think about the first time when white separates from red. 
That only happens when nums[white] == 1, so after the white += 1, 
we have nums[red] == 1, and notice that nums[red] will continue to be 1 as long as 
red != white. This is because red only gets incremented in the first case 
(nums[white] == 0), so we know that we are swapping nums[red] == 1 with nums[white] == 0.
'''

def sortColors(self, nums):
    red, white, blue = 0, 0, len(nums)-1
    
    while white <= blue:
        if nums[white] == 0:
            nums[red], nums[white] = nums[white], nums[red]
            white += 1
            red += 1
        elif nums[white] == 1:
            white += 1
        else:
            nums[white], nums[blue] = nums[blue], nums[white]
            blue -= 1