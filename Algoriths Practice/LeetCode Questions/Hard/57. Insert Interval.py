'''
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:
Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

Example 2:
Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].

This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].

'''


# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]

        Optimized with binary search.
        This is the O(n) solution doe, O(logn) binary solution shouldnt be too hard doe
        """

        if (len(intervals) == 0):
            intervals.append(newInterval)
            return intervals

        i = 0
        while (i != len(intervals) and intervals[i].start <= newInterval.start):
            i += 1

        # At the end of loop -> intervals[i].start > newInterval.start

        prevIntervalIndexI = i - 1
        j = prevIntervalIndexI

        while (j != len(intervals) and intervals[j].end <= newInterval.end):
            j += 1

        # At the end of loop -> intervals[j].end > newInterval.end

        # i is an the interval that has a start time larger than newInterval's start time
        # j is an the interval that has an end time that is larger than newInterval's end time
        prevIntervalIndexJ = j

        insertedInterval = None

        if (i == len(intervals)):
            intervals.append(Interval(newInterval.start, newInterval.end))
            return intervals

        if (newInterval.start <= intervals[prevIntervalIndexI].end):
            if (j == len(intervals) or newInterval.end < intervals[prevIntervalIndexJ].start):
                print("A")
                insertedInterval = Interval(intervals[prevIntervalIndexI].start, newInterval.end)
                return intervals[:prevIntervalIndexI] + [insertedInterval] + intervals[prevIntervalIndexJ:]
            elif (newInterval.end >= intervals[prevIntervalIndexJ].start):
                print("B")
                insertedInterval = Interval(intervals[prevIntervalIndexI].start, intervals[prevIntervalIndexJ].end)
                return intervals[:prevIntervalIndexI] + [insertedInterval] + intervals[prevIntervalIndexJ + 1:]
        elif (newInterval.start > intervals[prevIntervalIndexI].end):
            if (j == len(intervals) or newInterval.end < intervals[prevIntervalIndexJ].start):
                print("C")
                insertedInterval = Interval(newInterval.start, newInterval.end)
                return intervals[:prevIntervalIndexI] + [insertedInterval] + intervals[prevIntervalIndexJ:]
            elif (newInterval.end >= intervals[prevIntervalIndexJ].start):
                print("D")
                insertedInterval = Interval(newInterval.start, intervals[prevIntervalIndexJ].end)
                return intervals[:prevIntervalIndexI] + [insertedInterval] + intervals[prevIntervalIndexJ + 1:]





