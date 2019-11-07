/*

1122. Relative Sort Array
Easy

245

18

Favorite

Share
Given two arrays arr1 and arr2, the elements of arr2 are distinct, 
and all elements in arr2 are also in arr1.

Sort the elements of arr1 such that the relative ordering of items 
in arr1 are the same as in arr2.  Elements that don't appear in arr2 
should be placed at the end of arr1 in ascending order.

 

Example 1:

Input: arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
Output: [2,2,2,1,4,3,3,9,6,7,19]


*/


#import <unordered_map>


class Solution {
public:
    vector<int> relativeSortArray(vector<int>& arr1, vector<int>& arr2) {
        // Collect elements in map
        // Do insertion sort for elements that are not keys of map!
        
        // Do Radix sort or counting sort or insertion sort on other elements!
        
        
        unordered_map<int, int> o;
        vector<int> otherElements; 
        
        for(auto i: arr2) {
            // Setup map
            o[i] = 0;
        }
        
        for(auto i: arr1){
            if(o.find(i) != o.end()) {
                o[i] += 1;
            } else {
                otherElements.push_back(i);
            } 
        }
        
        vector<int> solution = vector<int>(arr1.size());
        
        cout << "ARRY 1 LENGTH " << arr1.size() <<endl;;
        
        int i = 0;
        for(auto key: arr2) {
            
           
            int count = o[key];
            for(int j = 0; j!= count; ++j) {
                
                solution[i] = key;
                i += 1;
            }
            
        }       
        
        sort(otherElements.begin(), otherElements.end(),  [](int a, int b) {return a < b; });
        
        for(auto i : o) {
            
            cout << i.first << " "<< i.second << endl;
        
        }
        
        ///solution.insert(solution.begin() + i, otherElements.begin(), otherElements.end());
        for(auto oe: otherElements) {
            solution[i] = oe;
            i +=1;
            
        }
        return solution;
        
        
    }
};



