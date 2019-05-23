'''
Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
Example 2:

Input: [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considerred overlapping.
Seen this question in a real interview before?  YesNo


'''


# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

# greedy solution
# sort by start time
# check if finish time for interval 1 intersects with start time for interval 2.
# Then take the left intervals start time, and the finish 
# time is the longer of those 2 intervals

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if(len(intervals) == 0):
            return []
        merged = []
        intervals.sort(key=lambda x: x.start)        
        
        a = intervals[0]
        for k in range(1, len(intervals)):
            j = intervals[k]
            
            if(a.end >= j.start):
                a = Interval(a.start, max(a.end, j.end))
                continue
            else:
                merged.append(a)
                a = j
        
        if( a is not None):
            merged.append(a)
            
        return merged


# Faster solution;

# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort(key = lambda x: x.start)
        ans = []
        
        for interval in intervals:
            if not ans or interval.start > ans[-1].end:
                ans.append(interval)
            else:
                ans[-1].end = max(ans[-1].end, interval.end)
        return ans
    
# Even faster

# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort(key=lambda x: x.start)
        result = []
        for i in intervals:
            if len(result) == 0:
                result.append(i)
            
            elif i.start <= result[-1].end:
                if i.end >= result[-1].end:
                    result[-1].end = i.end
                continue
            else:
                result.append(i)
        
        return result

# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        if not intervals:
            return []
        intervals=sorted(intervals, key=lambda x:x.start)
        res=[intervals[0]]
        for i in range(1,len(intervals)):
            if intervals[i].start>res[-1].end:
                res.append(intervals[i])
            else:
                res[-1].end=max(intervals[i].end,res[-1].end)
        return res

'''
Approach 1: Connected Components

Intuition

If we draw a graph (with intervals as nodes) that contains 
undirected edges between all pairs of 
intervals that overlap, then all intervals in each connected 
component of the graph can be merged into a single interval.

Algorithm

With the above intuition in mind, we can represent the graph as an adjacency list, 
inserting directed edges in both directions to simulate undirected edges. Then, 
to determine which connected component each node is it, we perform graph traversals 
from arbitrary unvisited nodes until all nodes have been visited. To do this 
efficiently, we store visited nodes in a Set, allowing for constant time containment 
checks and insertion. Finally, we consider each connected component, merging all of 
its intervals by constructing a new Interval with start equal to the minimum start 
among them and end equal to the maximum end.

This algorithm is correct simply because it is basically the brute force solution. We 
compare every interval to every other interval, so we know exactly which intervals overlap. 
The reason for the connected component search is that two intervals may not directly overlap, 
but might overlap indirectly via a third interval. See the example below to see this more clearly.

Components Example

Although (1, 5) and (6, 10) do not directly overlap, either would overlap with the other if first 
merged with (4, 7). There are two connected components, so if we 
merge their nodes, we expect to get the following two merged intervals:

(1, 10), (15, 20)
'''

