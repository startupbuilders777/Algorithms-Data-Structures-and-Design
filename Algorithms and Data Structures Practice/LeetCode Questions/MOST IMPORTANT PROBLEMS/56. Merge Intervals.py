'''
56. Merge Intervals
Medium

3184

245

Add to List

Share
Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
Example 2:

Input: [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considered overlapping.
NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.
'''


class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[List[int]]
        :rtype: List[List[int]]
        """
        
        # Sort by start time. 
        '''
        check if end of first is greater than start of second,
        then merge,
        
        keep going. 
        do it in place. Set items to None,
        then
        '''
        
        if(len(intervals) <= 1):
            return intervals
        
        
        intervals.sort(key=lambda x: x[0])
            
        i = 1
        
        prev = intervals[0] 
        result = []
        
        
    
        while i < len(intervals):
            nxt = intervals[i]
            if prev[1] >= nxt[0]:
                # merge
                prev = [min(prev[0], nxt[0]), max(prev[1], nxt[1])]
            
            else:
                result.append(prev)
                prev = nxt
            
            i += 1
            
            
        if prev:
            result.append(prev)
            
        return result
                
                
            