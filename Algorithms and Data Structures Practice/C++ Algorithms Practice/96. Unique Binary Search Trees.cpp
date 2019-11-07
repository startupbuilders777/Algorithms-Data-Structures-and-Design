
/*

96. Unique Binary Search Trees
Medium

2107

82

Favorite

Share
Given n, how many structurally unique BST's (binary search trees) 
that store values 1 ... n?

Example:

Input: 3
Output: 5
Explanation:
Given n = 3, there are a total of 5 unique BST's:

   1         3     3      2      1
    \       /     /      / \      \
     3     2     1      1   3      2
    /     /       \                 \
   2     1         2                 3

*/

class Solution {
public:
    int numTrees(int n) {
        if(n <= 1) return 1;
        
        vector<int> dp(n + 1, 0);
        dp[0] = 1;
        dp[1] = 1;        
        
        for(int i = 2; i <= n; i++) {
            for(int j = 0; j < i; j++) {
                dp[i] += dp[j] * dp[ i - j -1];
            }
        }
        return dp[n];
        
    }
};



#include <vector>

class Solution {
    

public:
    int numTrees(int n) {
        
        
        
        /*
        
        
        well we are given n:
        
        pick root node. 
        
        all elements less go to left, 
        all elemetns greater go to right. 
        
        
        Problem has DP!
        
        
        
        if we know the solution for n = 3
        
        Solution for n=4.
        Reuses solution for n=2 on its left side. 
        
        reuse solution n = 1 on right side sorta! because 3 is there (push it back to 1),
        
        Multiply left, and right side -> get count for that node. 
        Sum up all pivots!
        
        
        
        There is probably a math formula too:
        
        1 ->1
        
        2 -> 2
        
        3 -> 5
        
        4 -> 10 
        
        */
        
        vector<int> arr = vector<int>(n+1);
        
        
        if(n == 1){
            return 1;
        } else  if (n == 2){ 
            return 2;
        }
        
        arr[0] = 1;
        arr[1] = 1;
        
        for(int i = 2; i < n+1; i ++) {
            
           arr[i] = 0;   
        }
        
        /*
        int[3] -> 2 on left side, 2 on right side, 1 on both sides. 
        
        
        int[4] -> 3 on left side, 2L 1R, 1L, 2R, 3R, -> sum up all and set it 
        int[5] -> same shit
        
        
        */    
        
        for(int i = 2; i < n+1; ++i){
            
            //Root node cant be used so subtract 1 
            int nodes = i-1;
            
            for(int rightNodes = 0; rightNodes < nodes +1 ; ++rightNodes){
                
                int leftNodes = nodes - rightNodes;
                arr[i] += (arr[rightNodes] * arr[leftNodes]);
            }
        }
        
        return arr[n];
    }
};