'''
57. Insert Interval
Hard

1247

144

Add to List

Share
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
Example 2:

Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
Output: [[1,2],[3,10],[12,16]]
Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].
NOTE: input types have been changed on April 15, 2019. Please reset to default code definition to get new method signature.
'''


# BEST OVERALL SOLUTION:

class Solution:
    # @param intervals, a list of Intervals
    # @param newInterval, a Interval
    # @return a list of Interval
    def insert(self, intervals, newInterval):
        start = newInterval.start
        end = newInterval.end
        result = []
        i = 0
        while i < len(intervals):
            if start <= intervals[i].end:
                if end < intervals[i].start:
                    break
                start = min(start, intervals[i].start)
                end = max(end, intervals[i].end)
            else:
                result.append(intervals[i])
            i += 1
        result.append(Interval(start, end))
        result += intervals[i:]
        return result

'''
fastest solution 36ms


'''
# I extract all the start time points from 
# intervals to a list, all the end time 
# points to another list. Then make use of 
# binary search to check the position of 
# the newInterval. Once I found out the 
# position ( the overlapping duration), 
# then I replace them with the new 
# interval.

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """
        left, right = newInterval.start, newInterval.end
        start = [interval.start for interval in intervals]
        end = [interval.end for interval in intervals]
        i = bisect.bisect_left(start, left)
        j = bisect.bisect(end, right)
        if i > 0 and left <= intervals[i-1].end:
            left = intervals[i-1].start
            i = i - 1
        if j < len(intervals) and right >= intervals[j].start:
            right = intervals[j].end
            j = j + 1
        intervals[i:j] = [Interval(left, right)]
        return intervals


# BETTER SOLUTIONS SPEED WITH BISECT:

class Solution:
    def insert(self, intervals, newInterval):
        i = bisect.bisect_left([i.end for i in intervals], newInterval.start)
        j = bisect.bisect_right([i.start for i in intervals], newInterval.end)-1
        if i <= j:
            newInterval.start = min(newInterval.start, intervals[i].start)
            newInterval.end = max(newInterval.end, intervals[j].end)
        intervals[i: j+1] = [newInterval]
        return intervals

# custom binary search, without extra space, 56ms
class Solution:
    def insert(self, intervals, newInterval):
        n = len(intervals)

        def bis(op):          # return the index of the first op(intervals[i])==true
            l, r = 0, n-1
            while l <= r:
                m = (l+r) >> 1
                if op(intervals[m]):
                    r = m - 1
                else:
                    l = m + 1
            return l
        i = bis(lambda inv: inv.end >= newInterval.start)
        j = bis(lambda inv: inv.start > newInterval.end) - 1
        if i <= j:
            newInterval.start = min(newInterval.start, intervals[i].start)
            newInterval.end = max(newInterval.end, intervals[j].end)
        intervals[i: j+1] = [newInterval]
        return intervals


# HARMANS BAD BUT 100% CORRECT SOLUTION:

class Solution(object):
    def insert(self, intervals, newInterval):
        # Just assume you are inserting to end
        startIdx = len(intervals)
        endIdx =  len(intervals)
        
        startInterval = None 
        endInterval = None
        
    
        look_for_start = True
        look_for_end = False
        
        for idx, i in enumerate(intervals):
            
            if look_for_start and  newInterval[0] < i[0]:
                # Completely disjoint start. 
                startIdx = idx
                look_for_end = True
                look_for_start = False
                
                # end can still either be in it or not
                
            elif look_for_start and newInterval[0] >= i[0] and newInterval[0] <= i[1]:
                
                startInterval = i
                startIdx = idx
                look_for_end = True
                look_for_start = False
            
            if look_for_end and newInterval[1] < i[0]:
                # Disjoint end. 
                endInterval = None
                endIdx = idx
                break
                
            elif look_for_end and newInterval[1] >= i[0] and newInterval[1] <= i[1]:
                endInterval = i # newInterval[]
                endIdx = idx
                break
        
        print("START INT, END INT", (startInterval, endInterval))
        
        if startInterval is None:
            # put interval before startIdx.
            mergedStart = newInterval[0]
            
        elif startInterval:
            mergedStart = min(startInterval[0], newInterval[0])
        
        if endInterval is None:
            # put interval before endIdx. 
            # but what if we reach the END!!!!
            mergedEnd = newInterval[1]
        elif endInterval: 
            mergedEnd = max(endInterval[1], newInterval[1])
        
        computed = [mergedStart, mergedEnd]
        
        print(computed)
        
        # could probably do it inplace
        # with list insert, and swap popping!
        # no you cant messes up sort
        
        # What if startInterval is endInterval???
        # Only add once, not twice,
        # and sometimes dont add!! if it was merged on either end.
        # so add once or never[if its been merged on the start or the end]

        '''
        completely 
        
        '''        
        
        result = []
        
        for idx, i in enumerate(intervals):
            if idx == startIdx:
                if startInterval is None and newInterval[1] < i[0]:                
                    result.append(computed)
                    # the end interval can still be in it causing a merge!
                    result.append(i)
                else:
                    result.append(computed)
            elif idx > startIdx and idx < endIdx:
                continue
            elif idx == endIdx:
                if endInterval is None and newInterval[1] < i[0]:
                    result.append(i)
                else: 
                    continue
            else:
                result.append(i)
            
        if startIdx == len(intervals):
            result.append(computed)
        
        return result
    
        
        
        
        
        