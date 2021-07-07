/*

1200. Minimum Absolute Difference
Easy

57

7

Favorite

Share
Given an array of distinct integers arr, find all pairs of elements with the minimum absolute difference of any two elements. 

Return a list of pairs in ascending order(with respect to pairs), each pair [a, b] follows

a, b are from arr
a < b
b - a equals to the minimum absolute difference of any two elements in arr
 

Example 1:

Input: arr = [4,2,1,3]
Output: [[1,2],[2,3],[3,4]]
Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.
Example 2:

Input: arr = [1,3,6,10,15]
Output: [[1,3]]
Example 3:

Input: arr = [3,8,-10,23,19,-4,-14,27]
Output: [[-14,-10],[19,23],[23,27]]


*/

// YOU CAN USE CURLY BRACES AND NOT ADD TYPE!!!


class Solution {
public:
    vector<vector<int>> minimumAbsDifference(vector<int>& arr) {
        sort(arr.begin(), arr.end());
        vector<vector<int>> res;
        int n = arr.size();
        int cur = INT_MAX;
        for (int i = 1; i < n; i++) {
            if (arr[i] - arr[i - 1] == cur) {
                res.push_back({arr[i - 1], arr[i]});
            }
            else if (arr[i] - arr[i - 1] < cur) {
                res = {{arr[i - 1], arr[i]}};
                cur = arr[i] - arr[i - 1];
            }
        }
        return res;
    }
};


class Solution {
public:
    vector<vector<int>> minimumAbsDifference(vector<int>& arr) {
    
        
        vector<vector<int>> result; 
        
        
        // Just sort it. 
        // Then take running min -> if new min is found, dump our old result, and start again with that. return result.
        
        
        if(arr.size() == 0){
            return result;
        }
        
        sort(arr.begin(), arr.end());
        
        auto currElement = arr[0];
        
        auto currMin = INT_MAX;
        
        for(int i = 1; i != arr.size(); ++i) {
            
            
            if(abs(currElement - arr[i])  < currMin) {
                result.clear();
                currMin =  abs(currElement - arr[i]); 
            } 
            
            if(abs(currElement - arr[i]) == currMin) {
                result.push_back( vector<int>{currElement, arr[i]} );
            }
            currElement = arr[i];
            
        }
        
        return result;
        
    }
};