/*
410. Split Array Largest Sum
Hard

2185

81

Add to List

Share
Given an array nums which consists of non-negative integers and an integer m, 
you can split the array into m non-empty continuous subarrays.

Write an algorithm to minimize the largest sum among these m subarrays.

 

Example 1:

Input: nums = [7,2,5,10,8], m = 2
Output: 18
Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.
Example 2:

Input: nums = [1,2,3,4,5], m = 2
Output: 9
Example 3:

Input: nums = [1,4,4], m = 3
Output: 4
 

Constraints:

1 <= nums.length <= 1000
0 <= nums[i] <= 106
1 <= m <= min(50, nums.length)

*/


class Solution {
public:
    
    int enough(vector<int>& nums, int m, int k) {
        int groups = 0;
        int curr = 0;
        for(auto & i : nums) {
            if(curr + i > k)  {
                groups += 1;
                curr = 0;
            }
            curr += i;
        }
        groups += 1; // last group
        if(groups > m) {
            return false;
        } 
        return true; 
    }
    
    int splitArray(vector<int>& nums, int m) {
        // binary search because we only want the minimized value as answer 
        int high = std::accumulate(nums.begin(), nums.end(), 0);
        // low is actually the largest element in the array? 
        int low = *max_element(nums.begin(), nums.end());
        
        while(low < high) {
            
            int mid = low + (high - low)/2;
            // cout << "testing value " << mid << endl; 
            if(enough(nums, m, mid)) {
                // ok that worked. can we go smaller?
                high = mid;
            } else {
                // we need it bigger. 
                low = mid+1; 
            }   
        }
        
        return low; 
    }
};