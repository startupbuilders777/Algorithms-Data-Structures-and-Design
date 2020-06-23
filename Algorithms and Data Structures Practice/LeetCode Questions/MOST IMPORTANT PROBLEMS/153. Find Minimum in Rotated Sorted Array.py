'''
153. Find Minimum in Rotated Sorted Array
Medium

1979

227

Add to List

Share
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.

Example 1:

Input: [3,4,5,1,2] 
Output: 1
Example 2:

Input: [4,5,6,7,0,1,2]
Output: 0
'''

# MY SOLUTION:
class Solution:
    def findMin(self, nums: List[int]) -> int:
        
        '''
        check mid and mid + 1 for descending order. 
        if not, compare against reference, 
        go in direction based on reference comparison 
        for bin search.
        '''
        if(len(nums) == 1):
            return nums[0]
        
        # base cases:
        # how to detect if it was never shifted?
        # last element shifted?
        toFind = nums[0]
        l = 0
        r = len(nums) 
        
        while l != r:
            mid = l + (r-l)//2
            ele = nums[mid]
            nextEle = None
            if( mid + 1 < len(nums)):
                nextEle = nums[mid+1]
            else:
                # use prev element.
                nextEle = ele
                ele = nums[mid-1]
            
            # find pivot rite?
            if(ele > nextEle):
                return nextEle
            
            # pivot on left side. 
            if toFind > ele: 
                r = mid
            else:
                l = mid + 1
        return toFind

# Clearer Solution:
def findMin(self, nums: List[int]) -> int:
    left, right = 0, len(nums) - 1
    while nums[left] > nums[right]:
        middle  = (left + right) // 2
        if nums[middle] < nums[right]:
            right = middle
        else:
            left = middle + 1
    return nums[left]


'''
class Solution {
public:
    int findMin(vector<int> &num) {
        int low = 0, high = num.size() - 1;
        // loop invariant: 1. low < high
        //                 2. mid != high and thus A[mid] != A[high] (no duplicate exists)
        //                 3. minimum is between [low, high]
        // The proof that the loop will exit: after each iteration either the 'high' decreases
        // or the 'low' increases, so the interval [low, high] will always shrink.
        while (low < high) {
            auto mid = low + (high - low) / 2;
            if (num[mid] < num[high])
                // the mininum is in the left part
                high = mid;
            else if (num[mid] > num[high])
                // the mininum is in the right part
                low = mid + 1;
        }

        return num[low];
    }
};

'''

