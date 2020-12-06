/*
436. Find Right Interval
Medium

635

183

Add to List

Share
You are given an array of intervals, where 
intervals[i] = [starti, endi] and each starti is unique.

The right interval for an interval i is an interval j 
such that startj >= endi and startj is minimized.

Return an array of right interval indices for each interval i. 
If no right interval exists for interval i, then put -1 at index i.

 

Example 1:

Input: intervals = [[1,2]]
Output: [-1]
Explanation: There is only one interval in the collection, so it outputs -1.
Example 2:

Input: intervals = [[3,4],[2,3],[1,2]]
Output: [-1,0,1]
Explanation: There is no right interval for [3,4].
The right interval for [2,3] is [3,4] since start0 = 3 is the smallest start that is >= end1 = 3.
The right interval for [1,2] is [2,3] since start1 = 2 is the smallest start that is >= end2 = 2.
Example 3:

Input: intervals = [[1,4],[2,3],[3,4]]
Output: [-1,2,-1]
Explanation: There is no right interval for [1,4] and [3,4].
The right interval for [2,3] is [3,4] since start2 = 3 is the smallest start that is >= end1 = 3.
 

Constraints:

1 <= intervals.length <= 2 * 104
intervals[i].length == 2
-106 <= starti <= endi <= 106
The start point of each interval is unique.



*/
using namespace std; 

class Solution {
public:
    vector<int> findRightInterval(vector<vector<int>>& intervals) {
        // sort by start time. 
        // then just choose the right interval as you go through? 
        /*
        
        // sort the intervals you insert into PQ by finish time. 
        
        then check next interval start time, if its after an intervals finish time. 
        pop interval, assign its index to us. 
        
        ---   ---  -----
                      ----
                     --
                           -
        sorting + binary search?
        sorting + heap?
        sorting + treemap? 
        sorting + 2 arrays [ayyy the best soln ]
        
        The reason we shoud seek something faster than heap/treemap is because
        we are dealing with static data that doesnt change, and those structures are used 
        for dynamic data, hence 2 array soln. 
        */    
        
        //loop through intervals and save the index as part of tuple!
        for(int i = 0; i != intervals.size(); ++i) {
            intervals[i].push_back(i);
        }
        
        sort(intervals.begin(), intervals.end(), [](auto x, auto y) { return x[0] < y[0];});
        
        // sort by largest finish time at top!
        // IF YOU LOOK AT CMP, TO DO LEAST TO GREATEST, YOU ACTUALLY HAVE TO INVERSE
        // so its not a[1] < b[1] like in sort function above but a[1] > b[1]
        auto cmp = [](vector<int> a, vector<int>  b) {return a[1] > b[1];};
        priority_queue< vector<int>, vector< vector<int> >, decltype(cmp)> pq(cmp);
        
        vector<int> res;
        res.resize(intervals.size());
        
        for(int i = 0; i!= intervals.size(); ++i) {
            vector<int> inte = intervals[i];
            
            while(pq.size() > 0 && pq.top()[1] <= inte[0]) {
                res[pq.top()[2]] = inte[2];
                pq.pop();
            }
            pq.push(inte);
        }
        
        while(pq.size() > 0) {
            res[pq.top()[2]] = -1;
            pq.pop();
        }
        return res; 
    }
};
