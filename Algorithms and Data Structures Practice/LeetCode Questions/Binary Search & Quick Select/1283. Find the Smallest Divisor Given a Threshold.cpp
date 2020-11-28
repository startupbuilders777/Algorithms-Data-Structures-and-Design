/*
1283. Find the Smallest Divisor Given a Threshold
Medium

598

109

Add to List

Share
Given an array of integers nums and an integer threshold, we will choose a positive integer divisor and divide all the array by it and sum the result of the division. Find the smallest divisor such that the result mentioned above is less than or equal to threshold.

Each result of division is rounded to the nearest integer greater than or equal to that element. (For example: 7/3 = 3 and 10/2 = 5).

It is guaranteed that there will be an answer.

 

Example 1:

Input: nums = [1,2,5,9], threshold = 6
Output: 5
Explanation: We can get a sum to 17 (1+2+5+9) if the divisor is 1. 
If the divisor is 4 we can get a sum to 7 (1+1+2+3) and if the divisor is 5 the sum will be 5 (1+1+1+2). 
Example 2:

Input: nums = [2,3,5,7,11], threshold = 11
Output: 3
Example 3:

Input: nums = [19], threshold = 5
Output: 4
 

Constraints:

1 <= nums.length <= 5 * 10^4
1 <= nums[i] <= 10^6
nums.length <= threshold <= 10^6
Accepted
45.8K
Submissions
93.9K
*/



class Solution {
public:
    bool enough(vector<int>& nums, const int & threshold, const int & divisor) {
        int res = 0;
        for(auto & i : nums) {
            // cieling
            res += (i/divisor) + (i % divisor != 0);
            if(res  > threshold){
                return false;
            }
        } 
        return true;
    }
    
    int smallestDivisor(vector<int>& nums, int threshold) {
        int low = 1; 
        int high = *max_element(nums.begin(), nums.end());
        while(low < high) {   
            int mid = low + (high - low)/2;
            if(enough(nums, threshold, mid)) {
                // divisor worked go smaller. 
                high = mid;
            } else {
                //divisior too small, need bigger. 
                low = mid + 1;
            }
        }
        return low;
    }
};

/*
PYTHON SOLN: 


class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        compute_sum = lambda x : sum([ceil(n / x) for n in nums])
        
        # binary search
        left, right = 1, max(nums)
        while left <= right:
            pivot = (right + left) >> 1
            num = compute_sum(pivot)

            if num > threshold:
                left = pivot + 1
            else:
                right = pivot - 1
        
        # at the end of loop, left > right,
        # compute_sum(right) > threshold
        # compute_sum(left) <= threshold
        # --> return left
        return left

*/