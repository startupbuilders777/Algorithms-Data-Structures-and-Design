/*


Given an array of integers arr, write 
a function that returns true if and 
only if the number of occurrences of 
each value in the array is unique.

 

Example 1:

Input: arr = [1,2,2,1,1,3]
Output: true
Explanation: The value 1 has 3 occurrences, 2 has 2 and 3 has 1. No two values have the same number of occurrences.
Example 2:

Input: arr = [1,2]
Output: false
Example 3:

Input: arr = [-3,0,1,-3,1,1,1,-3,10,0]
Output: true

*/




#import <unordered_map>
#import <unordered_set>

class Solution {

    
    public:
    bool uniqueOccurrences(vector<int>& arr) {
        // count the occurences of every element in a map.
        // itereate through map check if any 2 elements have same value
        // by putting values in a set. 
        // if set ever "has" a value, return false, otherwise true
        /*
        
        Contrary to most existing answers here, note that there are actually 
        4 methods related to finding an element in a map (ignoring lower_bound, 
        upper_bound and equal_range, which are less precise):

    operator[] only exist in non-const version, as noted it will 
                create the element if it does not exist
    at(), introduced in C++11, returns a reference to 
            the element if it exists and throws an exception otherwise
    find() returns an iterator to the element if 
            it exists or an iterator to map::end() if it does not
    count() returns the number of such elements,
            in a map, this is 0 or 1
    Now that the semantics are clear, 
        let us review when to use which:

    if you only wish to know whether an element is
    present in the map (or not), then use count().
    if you wish to access the element, and it shall 
    be in the map, then use at().
    if you wish to access the element, and do not know 
    whether it is in the map or not, then use find(); do 
    not forget to check that the resulting iterator is 
    not equal to the result of end().
    
    finally, if you wish to access the element if it exists 
    or create it (and access it) if it does not, use operator[];
    if you do not wish to call the type default constructor to create it, 
    then use either insert or emplace appropriately
        
        */
        
        unordered_map<int,int> m;
        unordered_set<int> s;
        
        for(auto i : arr){
        
            if(m.find(i) != m.end()){
            m[i] += 1;
            } else{
            m[i] = 1;
            }
        
        }
        
        for(pair<int, int> e: m) {
            auto val = e.second;
            if(s.count(val)){
            return false;
            }
            s.insert(val);
        
        }
        
        return true;
        
    }
};