'''
986. Interval List Intersections
Medium

1687

53

Add to List

Share
Given two lists of closed intervals, each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

(Formally, a closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.  The intersection of two closed intervals is a set of real numbers that is either empty, or can be represented as a closed interval.  For example, the intersection of [1, 3] and [2, 4] is [2, 3].)

 

Example 1:



Input: A = [[0,2],[5,10],[13,23],[24,25]], B = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
 

Note:

0 <= A.length < 1000
0 <= B.length < 1000
0 <= A[i].start, A[i].end, B[i].start, B[i].end < 10^9


HARMAN SOLUTION:
'''

class Solution:
    def intervalIntersection(self, A: List[List[int]], B: List[List[int]]) -> List[List[int]]:
        
        '''
        FOR INTERVAL QUESTIONS RMBR:
        SORTS FOR INTERSECTIONS USUALLY BEST TO SORT BY START TIME.
        
        BUT WHENEVER WE ARE CHECKING TO REMOVE INTERVALS FROM THE ACTIVE REGION!
        REMOVE BASED ON EARLIEST FINISH TIME, RATHER THAN OTHER METRICS:
        
        SOLUTION IS 2 POINTERS:
        
        ANOTHER INTERSECTION HINT: 
        INTERSECTIONS HAVE THE FORM OF THE FOLLOWING:
        [max(Astart, Bstart), min(Aend, Bend)]
        -> intersection exists if above computation is a real interval!
             (aka has positive length)        
        '''
        
        i = 0
        j = 0
        
        res = []
        
        if len(A) == 0 or len(B) == 0:
            return []
        
        '''
        You dont move pointers based on what the next earlier one was
        but the one that finished earlier, 
        because that one can no longer intersect with anything!
        '''
        while i < len(A) and j < len(B):
            
            a_start, a_end = A[i]
            b_start, b_end = B[j]
            
            pos_int_s = max(a_start, b_start)
            pos_int_e = min(a_end, b_end)
            if pos_int_s <= pos_int_e:
                res.append([pos_int_s, pos_int_e])
            
            if a_end < b_end:
                i += 1
            else:
                j += 1 
                
        return res
    