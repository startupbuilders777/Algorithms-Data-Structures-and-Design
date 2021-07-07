/*
480. Sliding Window Median
Hard

1131

91

Add to List

Share
Median is the middle value in an ordered integer list. 
If the size of the list is even, there is no middle 
value. So the median is the mean of the two middle value.

Examples:
[2,3,4] , the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Your job is to output the median array for each window in the original array.

For example,
Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

Window position                Median
---------------               -----
[1  3  -1] -3  5  3  6  7       1
 1 [3  -1  -3] 5  3  6  7       -1
 1  3 [-1  -3  5] 3  6  7       -1
 1  3  -1 [-3  5  3] 6  7       3
 1  3  -1  -3 [5  3  6] 7       5
 1  3  -1  -3  5 [3  6  7]      6
Therefore, return the median sliding window as [1,-1,-1,3,5,6].

Note:
You may assume k is always valid, ie: k is always smaller than input array's size for non-empty array.
Answers within 10^-5 of the actual value will be accepted as correct.

*/

// HARMAN SINGH TREESET SOLUTION:


class Solution {
public:
    void insertTree(int element, multiset<double> & l, multiset<double> & r) {
        if(l.size() == r.size()) {
            // insert into left. 
            
            if(r.size() > 0 && *(r.begin()) < element) { 
                double temp = *(r.begin());
                
                // ERASING BY VALUE IS BUG FOR MULTISET BECAUSE IT REMOVES ALL COPIES
                // ONLY ERASE THE ITERATOR!! TO ERASE ONE. 
                r.erase(r.begin());
                r.insert(element);
                element = temp;
            }
            l.insert(element);
        } else {
            // l is bigger, insert into right. 
            
            if( *(--l.end()) > element ) {
                double temp = *(--l.end()) ;
                l.erase(--l.end()); //COOL TIP, YOU CAN ERASE WITH EITHER VALUE OR ITERATOR
                l.insert(element);
                element = temp; 
            }
            
            r.insert(element);
        }
    }
    
    void deleteTree(int element, multiset<double> & l, multiset<double> & r ) {
        // Find tree that contains element, remove, then rebalance. 
        bool leftBigger = l.size() > r.size();
        
        auto leftSearch =l.find(element);  
        if( leftSearch != l.end()) {
            l.erase(leftSearch);
            // if left is greater than right by 1 dont do anything    
            // if left is same size as right, move right element to left.  
            if(!leftBigger) {
                // move right to left. 
                auto rightEle = *(r.begin());
                r.erase(r.begin());
                l.insert(rightEle);
            }            
        } else {
            // search right, has to contain it.  
            auto rightSearch = r.find(element);
            r.erase(rightSearch);
            
            // if left is same size as right do nothing
            // otherwise, move left to right. 
            
            if(leftBigger) {
                auto leftEle = *(--l.end());
                l.erase(--l.end());
                r.insert(leftEle);
            }
        }
    }
    
    
    double calcMedian(const multiset<double> & l, const multiset<double> & r) {
       // always ensure left has 1 more element than right. 
       // then always return *(left.end() - 1)
          
      if(l.size() == r.size()) {
          
          return ( *(--l.end()) + *(r.begin()) ) / 2.0;  
      }  else {
          return *(--l.end());
      }
    } 
    
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {    
        // keep 2 multsets. 
        multiset<double> l;
        multiset<double> r;
        
        int i = 0;
        int j = 0;

        while(j < k) {            
            insertTree(nums[j], l, r);
            j += 1;
        }
        
        vector<double> res;
        double med = calcMedian(l, r);
        res.push_back(med);
        
        while(j != nums.size()) {            
            insertTree(nums[j], l, r);
            deleteTree(nums[i], l, r);
           
            med = calcMedian(l, r);
            res.push_back(med);
            i += 1;
            j += 1;
        }
        return res;    
    }
};
