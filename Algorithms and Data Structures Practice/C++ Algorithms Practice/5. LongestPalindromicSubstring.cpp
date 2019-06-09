/*
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

Example 1:

Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.
Example 2:

Input: "cbbd"
Output: "bb"

*/



// THIS WAS MY ACCEPTED SOLUTION. IT DID NOT USE DYNAMIC PROGRAMMING. 
using namespace std; 

typedef pair<int, int> palindrome_key;


class Solution {
public:
    
    bool isPalindrome(int start, int end, const string & str, map< pair<int, int>, bool> & cache) {
        
        //compare first half of string with second half!
        for(int t=0; t != (end-start)/2; ++t){
            if(str[start+t] != str[end-1-t]){
                return false;
            }
        }
        
        return true;
        
    }
    
    string longestPalindrome(string s) {
        // check all substrings, check if its palindromic, if it is, greater than max, then save
        // save partial palindroms in a 2d map, keep index of start and end node.
        // if you are checking somethign is a palindrom m[start-1][end+1] is a palindrom
        // just check if s[start-1] == s[end+1] is because we already know the inside is!
        // if not make the check and save the result in the map!
        
        // cant use unordered_map because it needs a suitable hashing function. hmmm
        map< pair<int, int>, bool> cache;
        
        int len = s.size();
        
        int longest = 0;
        int maxI = 0;
        int maxJ = 0;
        
        for(int i = 0; i!= len; ++i) {
            for(int j = i+1; j != len+1; ++j) { 
                //why do we make j's end len+1. because we want it to go up to len. 
                
                // cout << "WE ARE TESTING SUBSTRING " << s.substr(i, j-i) <<  " " ;
                // cout << " and it has length " << j-i << endl;
                
                if(j-i > longest && isPalindrome(i, j, s, cache)){
                    longest = j-i;
                    maxI = i;
                    maxJ = j;
                    
                }
            }
        }
        
        return s.substr(maxI, maxJ-maxI);
    }
};

// redo this, and fix DP please!
// ADDING DYNAMIC PROGRAMMING ACTUALLY MADE IT WORSE BECAUSE I WAS NOT CALCULATING THE OVERLAPPING SOLUTIONS PROPERLY!!!!
// VERY BAD. even when i fixed it so it did overlapping solutions still TLE. 

// ANOTHER SOLUTIONS IS THE EXAPAND AROUND CENTER SOLUTION! CHECK 2N-1 centers. return largest palindrome.



using namespace std; 

typedef pair<int, int> palindrome_key;


class Solution {
public:
    
    bool isPalindrome(int start, int end, const string & str, map< pair<int, int>, bool> & m){
      /*  
        // BTW: the following are the same thing: v.push_back({x, y}) and v.push_back(make_pair(x, y));
        cout << " start was " << start << " and end was  " << end << endl;
        
        cout << " LETS CHECK CACHE " << endl;
            for(auto kv: m) {
                
                cout << "For key " << kv.first.first << ", " << kv.first.second << " we have val " << kv.second << endl; 
           }
        
        */
        if(m.count({start+1, end-1}) == 1 ) {
          //  cout << " ive seen this before : " << 
          //      str.substr(start+1, (start+1)-(end-1)) << 
          //      " and result was: " 
          //      << m[{start+1, end-1}] 
          //      << endl;
            
            
            if(m.at({start+1, end-1}) && 
               str[start] == str[end-1]){
                
                m[{start, end}] = true;
            } else{
                m[{start, end}] = false;
            }  
            return m[{start, end}];
        }
        
        //compare first half of string with second half!
        // maybe we can fill lots of entries through one check actually!
        for(int t=0; t != (end-start)/2; ++t){
            if(str[start+t] != str[end-1-t]){
                
                m[{start, end}] = false;
                return false;
            }
        }
        
        m[{start, end}] = true;
        return true;
        
    }
    
    string longestPalindrome(string s) {
        // check all substrings, check if its palindromic, if it is, greater than max, then save
        // save partial palindroms in a 2d map, keep index of start and end node.
        // if you are checking somethign is a palindrom m[start-1][end+1] is a palindrom
        // just check if s[start-1] == s[end+1] is because we already know the inside is!
        // if not make the check and save the result in the map!
        
        // cant use unordered_map because it needs a suitable hashing function. hmmm
        map< pair<int, int>, bool> cache;
        
        int len = s.size();
        
        int longest = 0;
        int maxI = 0;
        int maxJ = 0;
        
        //You need to make it overlap. it isnt overlapping to take advantage of the map!
        // Let i go from start to end, and j from end to start. 
        
        for(int i = len-1; i!= -1; --i) {
            for(int j = len ; j != i; --j) { 
                //why do we make j's end len+1. because we want it to go up to len. 
                
                // cout << "WE ARE TESTING SUBSTRING " << s.substr(i, j-i) <<  " " ;
                // cout << " and it has length " << j-i << endl;
                
                if( j-i > longest && isPalindrome(i, j, s, cache)){
                    
                    longest = j-i;
                    maxI = i;
                    maxJ = j;
                    
                }
            }
            
            // cout << " LETS CHECK CACHE " << endl;
          //  for(auto kv: cache) {
                
            //    cout << "For key " << kv.first.first << ", " << kv.first.second << " we have val " << kv.second << endl; 
           // }
        }
        
        return s.substr(maxI, maxJ-maxI);
    }
};