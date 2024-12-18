'''
435. Non-overlapping Intervals
Medium

Given a collection of intervals, 
find the minimum number of intervals 
you need to remove to make the rest of 
the intervals non-overlapping.

 

Example 1:

Input: [[1,2],[2,3],[3,4],[1,3]]
Output: 1
Explanation: [1,3] can be removed and 
the rest of intervals are non-overlapping.

Example 2:

Input: [[1,2],[1,2],[1,2]]
Output: 2
Explanation: You need to remove two [1,2] to 
make the rest of intervals non-overlapping.

Example 3:

Input: [[1,2],[2,3]]
Output: 0
Explanation: You don't need to remove any of 
the intervals since they're already non-overlapping.

 

Note:

You may assume the interval's end point 
is always bigger than its start point.
Intervals like [1,2] and [2,3] have borders 
"touching" but they don't overlap each other.
'''

class Solution(object):
    def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: int
        """
        
        '''
        This problem is basically
        SCHEDULE AS MANY INTERVALS AS POSSIBLE,
        than remove the ones outside of that schedule!
        
        add the first earliest, one, then add the next earliest one that doesnt interfere 
        with what you have, keep going, done. 
        
        Intervals not scheduled are ones to remove. 
        '''
        
        intervals.sort(key=lambda x: x[1])
        
        scheduled = 0
        last_end_time = float("-inf")
        
        for i in intervals:
            if i[0] >= last_end_time: 
                scheduled += 1
                last_end_time = i[1]
        
        return len(intervals) - scheduled

## THE DIRECT SOLUTION:


# Sort the intervals by their start time. 
# If two intervals overlap, the interval 
# with larger end time will be removed 
# so as to have as little impact on 
# subsequent intervals as possible.

def eraseOverlapIntervals(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        if not intervals: return 0
        intervals.sort(key=lambda x: x.start)  # sort on start time
        currEnd, cnt = intervals[0].end, 0
        for x in intervals[1:]:
            if x.start < currEnd:  # find overlapping interval
                cnt += 1
                currEnd = min(currEnd, x.end)  # erase the one with larger end time
            else:
                currEnd = x.end   # update end time
        return cnt