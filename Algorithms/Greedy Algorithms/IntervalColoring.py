'''
Input n intevals: [s1, f1], [s2, f2], ... , [sn, fn]
Output: Use the minimum number of colors to color the intervals so that each interval gets one color and two
        overlapping intervals get diff colors

Lets think of a greedy algo:
First thought of a greedy algo: Choose a maximum subset of disjoint intervals and use only one color for them and repeat
                (Use earliest finishing time algo)

Lets see if there are counter exaples
_
____
   _
It wont work for this case because you only need 2 colors but the algo will suggest you need 3.

Well we would find a place where the maximum intersections occur at the same instance of time
between intervals and use that many colors rite?
 -> How to do that:
 Sort by starting time
Color the first interval.
additional intervals intersecting this interval shall be colored a different color,
(use the minimum number to color the interval i so that it doesnt conflict with the colors of the intervals
that are already colored)

pseudo algo
def colorIntervals(intervals):
    intervals = sort intervals by starting time
    color first interval <- 1

    activeInterval1 = [firstIntervalStart, firstIntervalEnd]
    map -> {activeInterval1 -> val}

    for(i in 1 to end):
        (startI, endI) = intervals[i]
        checkStartIsInAnActiveInterval(startI, endI, map)

def checkStartIsInActiveInterval:
    result = binaryRangeSearch(elementsInMap, startI) <- O(log n)
    if(it is in active interval)
        colorings = map[activeInterval]
        NewActiveInterval1 -> [startI, min{activeIntervalEnd, endI}]
        NewActiveInterval2 -> [min{activeIntervalEnd, endI}, all the other ends it rea]

def binaryRangeSearch():
    iterate thru keys
    return the range-element In The Map that is most clost to the start time (the active range start time is most close to the startElement)


THIS ALGO MAY NEED RANGE TREES
Exercise: Find a counter example showing that using an arbitrary ordering of intervals will not works
'''

intervals = {
    (1,2), (1, 7), (2, 6), (4, 5), (5, 9), (9,11),(6,9), (7,9), (4,7),(3,4),(4, 12), (1, 3), (3, 5), (2, 6), (2, 7)
}

