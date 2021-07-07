/*
1497. Check If Array Pairs Are Divisible by k
Medium

330

34

Add to List

Share
Given an array of integers arr of even length n and an integer k.

We want to divide the array into exactly n / 2 pairs such that the sum of each pair is divisible by k.

Return True If you can find a way to do that or False otherwise.

 

Example 1:

Input: arr = [1,2,3,4,5,10,6,7,8,9], k = 5
Output: true
Explanation: Pairs are (1,9),(2,8),(3,7),(4,6) and (5,10).
Example 2:

Input: arr = [1,2,3,4,5,6], k = 7
Output: true
Explanation: Pairs are (1,6),(2,5) and(3,4).
Example 3:

Input: arr = [1,2,3,4,5,6], k = 10
Output: false
Explanation: You can try all possible pairs to see that there is no way to divide arr into 3 pairs each with sum divisible by 10.
Example 4:

Input: arr = [-10,10], k = 2
Output: true
Example 5:

Input: arr = [-1,1,-2,2,-3,3,-4,4], k = 3
Output: true
 

Constraints:

arr.length == n
1 <= n <= 10^5
n is even.
-10^9 <= arr[i] <= 10^9
1 <= k <= 10^5


*/

#include <bits/stdc++.h> 

class Solution {
public:
    /*
        k=3
        2+7 
        2%3 -> 2
        7%3 -> 1   -> search for 3 - 1 -> find 2, remove from map.     
        if the number itself is divisible by  k then what?
        if the number is negative then what? 
    */
    bool canArrange(vector<int>& arr, int k) {

        unordered_multiset<int> s;
        
        for(auto & i : arr) {
            if(i < 0){ 
                // HOW TO MAKE NUMBER POSITIVE
                i += (abs(i)/k + (i%k != 0))*k;
            }
            // ANOTHER WAY  TO KEEP MODS BETWEEN [0, K-1 is just do following]
            // i = (i%k + k)%k
            if(s.find(k - i%k) != s.end()) {
                s.erase(s.find(k - i%k));
            } else {
                if(i%k == 0) {
                    s.insert(k);
                }else {                
                    s.insert(i%k);
                }
            }
        }

        if(s.size() == 0) {
            return true;
        }
        return false;
    }
};


// VERY CLEAN SOLUTION

class Solution {
public:
    bool canArrange(vector<int>& arr, int k) {
        vector<int> freq(k);
        
        for (int x : arr)
            freq[((x % k) + k) % k]++;
        
        if (freq[0] % 2)
            return false;
        
        for (int i=1, j=k-1; i<j; i++, j--)
            if (freq[i] != freq[j])
                return false;
        
        return true;
    }
};

/*
FASTER CPP SOLUTIONS

*/

class Solution {
public:
    bool canArrange(vector<int>& arr, int k) {
        vector<int> freq(k,0);
        int n = arr.size();
        for(int i = 0; i < n; ++i) {
            if(arr[i] >= 0) {
                freq[arr[i] % k] = ((freq[arr[i] % k] + 1) % k);
            }
            else {
                int temp = k - abs(arr[i] % k);
                if(temp == k)
                    temp = 0;
                freq[temp] = ((freq[temp] + 1) % k);
            }
        }

        if(freq[0] % 2 != 0) 
            return false;
        for(int i = 1; i <= freq.size() / 2; i++){
            if(freq[i] != freq[k - i]) return false;
        }
        return true;
    }
    
};

static const auto speedup = []() {
		std::ios::sync_with_stdio(false); std::cin.tie(nullptr); cout.tie(nullptr); return 0;
}();



// FASTEST SOLUTION:

class Solution {
public:
    bool canArrange(vector<int>& arr, int k) {
        ios::sync_with_stdio(false), cin.tie(nullptr), cout.tie(nullptr);
        
        vector<int> modulo(k, 0);
        for(int e : arr){
            e %= k;
            if(e < 0) modulo[e + k]++;
            else modulo[e]++;
        }
        
        if(modulo[0] & 1) return false;
        for(int i = k / 2; i >= 1 ; i--)
            if(modulo[i] != modulo[k - i])
                return false;
        return true;
    }
};