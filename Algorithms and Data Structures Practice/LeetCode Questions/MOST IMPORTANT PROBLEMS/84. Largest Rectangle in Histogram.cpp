/*
COMPLETED

84. Largest Rectangle in Histogram
Hard

4572

94

Add to List

Share
Given n non-negative integers representing the histogram's bar 
height where the width of each bar is 1, 
find the area of largest rectangle in the histogram.

 


Above is a histogram where width of each bar is 1, given height = [2,1,5,6,2,3].

 


The largest rectangle is shown in the shaded area, which has area = 10 unit.

 

Example:

Input: [2,1,5,6,2,3]
Output: 10
*/


class Solution {
public:
    /*    
        process element.
        larger, append and do nothing.
        
        smaller -> start popping toward left direction
        until you see element smaller than you. 
        see area you get. 
        save that in stack as tuple 
        -> (SMALLHEIGHT, LENGTH)
        process next element append. 
        
        if there is 1 element return it. 
    
        you also have to process left to right
        and right to left.     
        
        IDEA REFINEMENT.
        
        after processing with a stack in one direction.
        we dont have to do other way, because whats in stack is 
        monotonically increasing. 
        
        then you can process monotic stack by going backwards and accumulating
        total width and from right side and current height. 
        
        what we have leftover in stack 
        
        2 1 2 
        
        
        ABOVE IS TOTALLY WRONG DOES NOT WORK FOR WIERD EXAMPLES
        BECAUSE THE 4 FOR LOOPS u get from doing above does not work on 
        [3,6,5,7,4,8,1,0] where you get answer 18
        when actual answer is 20. 
        
        META STRATEGY -> when algo becomes complicated/
        has disjoint parts and you have to coordinate how disjoint parts work
        you are PROBABLY DOING IT WRONG. ITS A SMELL. wierd algo constructions 
        are usually wrong.         
    */
        
    struct Pair {
        int h;
        int len;
        Pair(int h, int len): h(h), len(len) {}
    
    };
    
    int largestRectangleArea(vector<int>& heights) {
        
        stack<Pair> st; 
        int widthFromRight;
        int currArea = 0;
        
        
        /*
        2, 1, 2 was failed in previous solution
        THE MONOTONIC STACK SHOULD BE USED COMPLETELY.
        ALL OF ITS IDEAS SHOULD BE SQUEEZED AND USED FOR SOLVING,
        LIKE FOR THIS PROBLEM. 
        -> EXTRA trick is: 
            when popping from stack -> do things concurrently,
            accumulate width from the current height considered,
            + DO MAX SLICE with the increasing sequence you are popping out. 
        */
        for(auto it = heights.begin(); it != heights.end(); it++) {    
            int h = *it;            
            int widthFromRight = 0;
            while(st.size() > 0 && st.top().h >= h) {
                // start popping. 
                // the elements we pop actually form an increasing sequence
                // and you can get max area of increasing rectangles 
                // easily by accumulating!
                auto [h2, w] = st.top();
                widthFromRight += w;
                currArea = max(widthFromRight*h2, currArea);
                st.pop();
            } 
            
            // multiply width and 
            // height and compare with curr max. 
            // the below line can be erased, and algo would still work
            // because this shit gets computed anyway in the while loop below
            currArea = max(h*(1+widthFromRight), currArea);
            st.push(Pair(h, (1+widthFromRight)));
        }
        // elements left in stack are what?
        // -> elements that form an increasing sequence 1, 2, 3
        widthFromRight = 0;
        while(st.size() > 0) { 
            auto [h, w] = st.top();
            widthFromRight += w;
            currArea = max(widthFromRight*h, currArea);
            st.pop();
        }
        return currArea;  
    }
};


// More improved solutions:

/*
Explanation: As we check each height, we see if it is less than 
any height we've seen so far. If we've seen a larger height in 
our stack, we check and see what area that rectangle could have 
given us and pop it from the stack. We can continue this approach 
for each for each rectangle: finding the max area of any larger 
rectangle previously seen before adding this rectangle to the stack 
and thus limiting the height of any possible rectangle after.
*/

int largestRectangleArea(vector<int>& heights) {
        if(heights.size() == 0) return 0;
        
        stack<int> s;
        int area = 0;
        
        for(int i = 0; i <= heights.size(); i++){
            while(!s.empty() && (i == heights.size() || heights[s.top()] > heights[i])){
                int height = heights[s.top()];
                s.pop();
                int width = (!s.empty()) ? i - s.top() -1 : i;
                area = max(area, height * width);
            }
            s.push(i);
        }
        return area;
    }

/*
PYTHON DP:

1dp
    res = 0
    for j in range(len(heights)):
        h = heights[j]
        for i in range(j,len(heights)):
            h = min(heights[i],h)
            res = max(res,h*(i-j+1))
    return res
    n = len(heights)
    for mid in range(n):
        h,l,r = heights[mid],mid,mid
        while l-1>=0 and heights[l-1]>=h:
            l-=1
        while r+1<n and  heights[r+1]>=h:
            r+=1
        res = max(res,(r-l+1)*h)
    return res 
2
    n = len(heights)
    l,r = [0]*n,[0]*n
    stack = list()
    for i in range(n):
        while stack and heights[stack[-1]]>=heights[i]:
            stack.pop()
        l[i]=stack[-1] if stack else -1
        stack.append(i)
    stack = list()
    for i in range(n-1,-1,-1):
        while stack and heights[stack[-1]]>=heights[i]:
            stack.pop()
        r[i]=stack[-1] if stack else n
        stack.append(i)
    return max((r[i]-l[i]-1)*heights[i] for i in range(n)) if n>0 else 0
3
	n = len(heights)
    l,r = [0]*n,[n]*n
    stack = list()
    for i in range(n):
        while stack and heights[stack[-1]]>=heights[i]:
            r[stack[-1]]=i
            stack.pop()
        l[i]=stack[-1] if stack else -1
        stack.append(i)
    return max((r[i]-l[i]-1)*heights[i] for i in range(n)) if n>0 else 0




The idea is simple: for a given range of bars, the maximum area 
can either from left or right half of the bars, or from the area 
containing the middle two bars. For the last condition, expanding 
from the middle two bars to find a maximum area is O(n), which 
makes a typical Divide and Conquer solution with T(n) = 2T(n/2) + O(n). 
Thus the overall complexity is O(nlgn) for time and O(1) for space (or O(lgn) considering stack usage).

Following is the code accepted with 44ms. I posted this because 
I didn't find a similar solution, but only the RMQ 
idea which seemed less straightforward to me.

class Solution {
    int maxCombineArea(const vector<int> &height, int s, int m, int e) {
        // Expand from the middle to find the max area containing height[m] and height[m+1]
        int i = m, j = m+1;
        int area = 0, h = min(height[i], height[j]);
        while(i >= s && j <= e) {
            h = min(h, min(height[i], height[j]));
            area = max(area, (j-i+1) * h);
            if (i == s) {
                ++j;
            }
            else if (j == e) {
                --i;
            }
            else {
                // if both sides have not reached the boundary,
                // compare the outer bars and expand towards the bigger side
                if (height[i-1] > height[j+1]) {
                    --i;
                }
                else {
                    ++j;
                }
            }
        }
        return area;
    }
    int maxArea(const vector<int> &height, int s, int e) {
        // if the range only contains one bar, return its height as area
        if (s == e) {
            return height[s];
        }
        // otherwise, divide & conquer, the max area must be among the following 3 values
        int m = s + (e-s)/2;
        // 1 - max area from left half
        int area = maxArea(height, s, m);
        // 2 - max area from right half
        area = max(area, maxArea(height, m+1, e));
        // 3 - max area across the middle
        area = max(area, maxCombineArea(height, s, m, e));
        return area;
    }
public:
    int largestRectangleArea(vector<int> &height) {
        if (height.empty()) {
            return 0;
        }
        return maxArea(height, 0, height.size()-1);
    }
};
*/


// DIVID AND CONQUER
/*
The idea is simple: for a given range of bars, the maximum area can either 
from left or right half of the bars, or from the area containing the middle two bars. 
For the last condition, expanding from the middle two bars to find a maximum 
area is O(n), which makes a typical Divide and Conquer solution with T(n) = 2T(n/2) + O(n). 
Thus the overall complexity is O(nlgn) for time and O(1) 
for space (or O(lgn) considering stack usage).

Following is the code accepted with 44ms. I posted this because I didn't find a similar solution, 
but only the RMQ idea which seemed less straightforward to me.

class Solution {
    int maxCombineArea(const vector<int> &height, int s, int m, int e) {
        // Expand from the middle to find the max area containing height[m] and height[m+1]
        int i = m, j = m+1;
        int area = 0, h = min(height[i], height[j]);
        while(i >= s && j <= e) {
            h = min(h, min(height[i], height[j]));
            area = max(area, (j-i+1) * h);
            if (i == s) {
                ++j;
            }
            else if (j == e) {
                --i;
            }
            else {
                // if both sides have not reached the boundary,
                // compare the outer bars and expand towards the bigger side
                if (height[i-1] > height[j+1]) {
                    --i;
                }
                else {
                    ++j;
                }
            }
        }
        return area;
    }
    int maxArea(const vector<int> &height, int s, int e) {
        // if the range only contains one bar, return its height as area
        if (s == e) {
            return height[s];
        }
        // otherwise, divide & conquer, the max area must be among the following 3 values
        int m = s + (e-s)/2;
        // 1 - max area from left half
        int area = maxArea(height, s, m);
        // 2 - max area from right half
        area = max(area, maxArea(height, m+1, e));
        // 3 - max area across the middle
        area = max(area, maxCombineArea(height, s, m, e));
        return area;
    }
public:
    int largestRectangleArea(vector<int> &height) {
        if (height.empty()) {
            return 0;
        }
        return maxArea(height, 0, height.size()-1);
    }
};
*/
