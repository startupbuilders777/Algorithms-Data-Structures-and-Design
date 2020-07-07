'''
88. Merge Sorted Array
Easy

2197

4153

Add to List

Share
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:

The number of elements initialized in nums1 and nums2 are m and n respectively.
You may assume that nums1 has enough space (size that is equal to m + n) to hold additional elements from nums2.
Example:

Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]
 

Constraints:

-10^9 <= nums1[i], nums2[i] <= 10^9
nums1.length == m + n
nums2.length == n
Accepted
592,599
Submissions
1,514,076
'''

'''
MEMORIZE THE BEAUTIFUL WAY:
'''

def merge(self, nums1, m, nums2, n):
        while m > 0 and n > 0:
            if nums1[m-1] >= nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        if n > 0:
            nums1[:n] = nums2[:n]



class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """

        # SWAP ALL THE ELEMENTS IN NUMS1 TO THE END FIRST!!!
        end = len(nums1) - 1
        
        sizeNums1 = len(nums1) - len(nums2)
        swapPtr = sizeNums1 - 1
        
        while swapPtr != -1:
            nums1[swapPtr], nums1[end] = nums1[end], nums1[swapPtr]
            swapPtr -= 1
            end -= 1
            
        print(nums1)
        
        inPtr = 0
        l = end + 1
        r = 0
        
        if len(nums2) == 0:
            return nums1
        
        while inPtr != len(nums1):
            if r == len(nums2) and l == len(nums1):
                return nums1
            elif l == len(nums1):
                nums2[r], nums1[inPtr] = nums1[inPtr], nums2[r]                
                r += 1
            elif r == len(nums2):
                nums1[l], nums1[inPtr] = nums1[inPtr], nums1[l]
                l += 1
            elif nums2[r] < nums1[l]:
                nums2[r], nums1[inPtr] = nums1[inPtr], nums2[r]                
                r += 1
            else:
                nums1[l], nums1[inPtr] = nums1[inPtr], nums1[l]
                l += 1
            inPtr += 1
            
