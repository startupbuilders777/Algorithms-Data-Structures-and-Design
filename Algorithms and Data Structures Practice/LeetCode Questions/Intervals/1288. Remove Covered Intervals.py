'''
1288. Remove Covered Intervals
Medium

507

23

Add to List

Share
Given a list of intervals, remove all intervals that are covered by another interval in the list.

Interval [a,b) is covered by interval [c,d) if and only if c <= a and b <= d.

After doing so, return the number of remaining intervals.

 

Example 1:

Input: intervals = [[1,4],[3,6],[2,8]]
Output: 2
Explanation: Interval [3,6] is covered by [2,8], therefore it is removed.
Example 2:

Input: intervals = [[1,4],[2,3]]
Output: 1
Example 3:

Input: intervals = [[0,10],[5,12]]
Output: 2
Example 4:

Input: intervals = [[3,10],[4,10],[5,11]]
Output: 2
Example 5:

Input: intervals = [[1,2],[1,4],[3,4]]
Output: 1
 

Constraints:

1 <= intervals.length <= 1000
intervals[i].length == 2
0 <= intervals[i][0] < intervals[i][1] <= 10^5
All the intervals are unique.


'''


class Solution:
    def removeCoveredIntervals(self, intervals: List[List[int]]) -> int:
        '''
        Sort by start time. 
        
        Then check if interval after me I cover. If i do remove, 
        otherwise keep. 
        
        sort intervals by start time.          
        
        when checking next interval, make sure its start time is after 
        the max finish time, otherwise covered. 
        
        Sort by start time.
        
        Compare end time of first one with endtime of second one, if its within its covered.
        OTHERWISE!!,
        
        if you get an interval with further endtime -> update the FOCUS interval to that. 
        because it will be able to cover more on the right. 
        So keep the one with largest end time asthe main focus. 
        
        Requires sorting on start time  -> NLOGN.
        
        need to do 2 sorts, 
        earliest start time,
        then for those with same start time -> latest finish time comes first. 
        So then the latest finish times can consume the earlier finish times and be used to consume intervals
        without same start time. 
        '''
        
        # DO A DOUBLE SORT -> MAJOR SORT ON START -> MINOR SORT ON FINISH TIME. 
        intervals.sort(key=lambda x: (x[0], -x[1]))
        
        curr_fin = intervals[0][1]
        
        covered_count = 0
        for i in range(1, len(intervals)):
            nxt_fin = intervals[i][1]

            if nxt_fin <= curr_fin:
                covered_count += 1
            else:
                curr_fin = nxt_fin
                
        return len(intervals) - covered_count