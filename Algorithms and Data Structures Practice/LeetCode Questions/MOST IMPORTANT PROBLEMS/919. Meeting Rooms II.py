"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

from heapq import *
import itertools

class Solution:
    """
    @param intervals: an array of meeting time intervals
    @return: the minimum number of conference rooms required
    """
    
    
    def minMeetingRooms(self, intervals):
        # Write your code here
        
        '''
        Sort by start time.
        
        runningCounter = 0
        process intervals:
            
        
            process, keep a current end.
            when we process interval outside current interval,
            
            while startTimeOFNewInterval > endTime:
            
                we pop interval 
                runningCounter -= 1
                look at next interval with starting Time, check its end time
                if its also not in there, pop
                (In this way we only look at an interval twice, when we add it, 
                and when we remove it. )

        '''
        sorted_i = sorted(intervals, key=lambda x: x.start)
        
        pq = []
        counter = itertools.count()
        active_colors = 0
        max_colors = 0
        
        for i in sorted_i:
            iStart = i.start
            iEnd = i.end
            
            while len(pq) != 0:
                
                min_end_time, _, interval_to_be_popped = pq[0]                
                if(iStart <= min_end_time):
                    break                
                active_colors -= 1
                _ = heappop(pq)
                            
            c = next(counter)
            item = [iEnd, c, i]
            heappush(pq, item)
            print("increment active colors")
            active_colors += 1
            max_colors = max(active_colors, max_colors)
        return max_colors
'''
C++ SOLUTIONS


Solution

# The idea is to group those non-overlapping meetings in the same 
# room and then count how many rooms we need. You may refer to this link.

bool myComp(const Interval &a, const Interval &b){
    return (a.start<b.start);
}
class Solution {
public:
    int minMeetingRooms(vector<Interval>& intervals) {
        int rooms = 0;
        priority_queue<int> pq;//prioritize earlier ending time
        sort(intervals.begin(), intervals.end(), myComp);
        for(int i=0; i<intervals.size(); ++i){
            while(!pq.empty() && -pq.top()<intervals[i].start) pq.pop();
            pq.push(-intervals[i].end);
            rooms = max(rooms, (int)pq.size());
        }
        return rooms;
    }
};
another solution: for each group of non-overlapping intervals, we just need to store the last added one instead of the full list. So we could use a vector < Interval > instead of vector < vector < Interval >> in C++. The code is now as follows.

class Solution {
public:
    int minMeetingRooms(vector<Interval>& intervals) {
        sort(intervals.begin(), intervals.end(), compare);
        vector<Interval> rooms;
        int n = intervals.size();
        for (int i = 0; i < n; i++) {
            int idx = findNonOverlapping(rooms, intervals[i]);
            if (rooms.empty() || idx == -1)
                rooms.push_back(intervals[i]);
            else rooms[idx] = intervals[i];
        }
        return (int)rooms.size();
    }
private:
    static bool compare(Interval& interval1, Interval& interval2) {
        return interval1.start < interval2.start;
    }
    int findNonOverlapping(vector<Interval>& rooms, Interval& interval) {
        int n = rooms.size();
        for (int i = 0; i < n; i++)
            if (interval.start >= rooms[i].end)
                return i;
        return -1;
    }
};
another smart solution: here

class Solution {
public:
    int minMeetingRooms(vector<Interval>& intervals){
        // Time O(NlogN)
        map<int, int> mp; // key:time, val:+1 start, -1 end
        for (int i = 0; i < intervals.size(); ++i) {
            mp[intervals[i].start]++;
            mp[intervals[i].end]--;
        }
        int cnt = 0; maxcnt = 0;
        for (auto it = mp.begin(); it != mp.end(); ++it) {
            cnt += it->second;
            maxcnt = max(maxcnt, cnt);
        }
        return maxcnt;
    }
};

'''