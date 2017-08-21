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

    map[] <- Takes a starting time when you get something and returns number of intervals intersecting at that time, k,
             then color your interval k+1
    after that
    map[(start, end)] <- add a range, and it increments corresponding intervals by 1 depending

    for(1 to end):
        if()


THIS ALGO MAY NEED RANGE TREES
Exercise: Find a counter example showing that using an arbitrary ordering of intervals will not works
'''

intervals = {
    (1,2), (1, 7), (2, 6), (4, 5), (5, 9), (9,11),(6,9), (7,9), (4,7),(3,4),(4, 12), (1, 3), (3, 5), (2, 6), (2, 7)
}

