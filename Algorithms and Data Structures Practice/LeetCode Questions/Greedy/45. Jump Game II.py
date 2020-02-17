'''
45. Jump Game II
Hard

1844

107

Add to List

Share
Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Your goal is to reach the last index in the minimum number of jumps.

Example:

Input: [2,3,1,1,4]
Output: 2
Explanation: The minimum number of jumps to reach the last index is 2.
    Jump 1 step from index 0 to 1, then 3 steps to the last index.

'''

from collections import deque

class Solution(object):
    def jump(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        '''
        Try greedy solution
        
        process each node going forward-> 
        
        if it extends coverage, add it to soln. 
        if not do not add it. `
        always add nodes that add the most to the coverage!
        
        then add the next node, within that coverage that extends the 
        coverage the most.
        
        GREEDY COVERAGE SOLUTION
        '''
        if(len(nums) == 1):
            return 0
        
        
        curr_coverage = nums[0]
        
        i = 0
        steps = 1
        
        while True:
            # when looking at elements to jump to
            # jump to the one that maximizes coverage
            
            print("STEP WE CHOSE IS ")
            if curr_coverage >= len(nums) -1:
                # we are done!!
                return steps
            
            
            
            max_coverage = 0
            next_i = i
            
            for i in range(i+1, curr_coverage+1):
                this_coverage = i + nums[i]
                
                if( i < len(nums) and this_coverage > max_coverage):
                    max_coverage = this_coverage
                    next_i = i
                
            # Ok so we found the biggest one in our current coverage. 
            # set it and update params.
            
            steps += 1
            i = next_i
            curr_coverage = max_coverage
            