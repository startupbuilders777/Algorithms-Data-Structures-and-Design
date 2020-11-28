/*
Hanging Banners
Question 212 of 858
You are given a list of list of integers intervals of the form [start, end] representing the starts and end points of banners you want to hang. Each banner needs at least one pin to stay up, and one pin can hang multiple banners. Return the smallest number of pins required to hang all the banners.

Note: The endpoints are inclusive, so if two banners are touching, e.g. [1, 3] and [3, 5], you can put a pin at 3 to hang both of them.

Constraints

n ≤ 100,000 where n is the length of intervals
Example 1
Input

intervals = [
    [1, 4],
    [4, 5],
    [7, 9],
    [9, 12]
]
Output

2
Explanation

You can put two pins at 4 and 9 to hang all the banners..

Example 2
Input

intervals = [
    [1, 10],
    [5, 10],
    [6, 10],
    [9, 10]
]
Output

1
Explanation

You can put one pin at 10.

*/

// TWO WAYS TO SOLVE WOWOWO
// YOU CAN EITHER SORT BY START TIME LIKE BELOW
int solve1(vector<vector<int>>& intervals) {
    /*
    sort by start time, 

    keep set of end times. 
    update with smallest end time so far seen. 

    if next interval is past the current smallest end time, pop all intervals and add a pin,
    then restart algo. 
    */
    sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b)-> bool {return a[0] < b[0];} );
    
    int pins = 0;
    int nearestEnd = -1;
    
    for(int i = 0; i != intervals.size(); ++i) {
        
        auto intv = intervals[i];
        if(intv[0] > nearestEnd) {
            pins += 1;
            nearestEnd = intv[1];
        } else {
            // keep in set of intervals!
            nearestEnd = min(nearestEnd, intv[1]);
        }
    }
    return pins;
}

// YOU CAN SORT BY END TIME TOO LIKE BELOW: 

class Solution:
    def solve(self, intervals):
        intervals.sort(key=lambda i: i[1])
        last = float("-inf")
        ans = 0
        for s, e in intervals:
            if s <= last:
                continue
            last = e
            ans += 1
        return ans



using namespace std;
bool comp(vector<int>& a, vector<int>& b) {
    return a[1] < b[1];
}
class Solution {
    public:
    int solve(vector<vector<int>>& intervals) {
        int n = intervals.size();
        if (n <= 1) return n;
        sort(intervals.begin(), intervals.end(), comp);
        int l = intervals[0][1];
        int ans = 1;
        for (int i = 1; i < n; i++) {
            if (intervals[i][0] > l) {
                ans++;
                l = max(l, intervals[i][1]);
            }
        }
        return ans;
    }
};

int solve(vector<vector<int>>& p) {
    return (new Solution)->solve(p);
}
