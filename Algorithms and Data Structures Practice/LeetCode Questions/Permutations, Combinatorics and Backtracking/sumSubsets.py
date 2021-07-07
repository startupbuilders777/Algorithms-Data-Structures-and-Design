'''

Given a sorted array of integers arr and an integer num, find all possible unique subsets of arr that add up to num. Both the array of subsets and the subsets themselves should be sorted in lexicographical order.

Example

For arr = [1, 2, 3, 4, 5] and num = 5, the output should be
sumSubsets(arr, num) = [[1, 4], [2, 3], [5]].

Input/Output

[execution time limit] 4 seconds (py3)

[input] array.integer arr

A sorted array of integers.

Guaranteed constraints:
0 ≤ arr.length ≤ 50,
1 ≤ arr[i] ≤ num.

[input] integer num

A non-negative integer.

Guaranteed constraints:
0 ≤ num ≤ 1000.

[output] array.array.integer

A sorted array containing sorted subsets composed of elements from arr that have a sum of num. It is guaranteed that there are no more than 1000 subsets in the answer.

[Python 3] Syntax Tips



DO RECURSION WITHOUT SET AND SORTING RESULTS. 

'''


def sumSubsets(arr, num):
    
    # create subset then check against num?
    # maybe. 
    # dp sums we can form?
    
    # enumerate all subsets -> if sum goes over target throw away subset. 
    
    result = []
   
    N = len(arr)
    
    # limit parameter is for deduplicating. 
    # if you dont take curr element, you have to limit all after you. 
    
    def collect(i, subset, val, limit):
        nonlocal result
        if val < 0:
            return 
              
        if val == 0:
            result.append(subset)
            return 
        
        if i == N:
            return         
        
        if arr[i] <= limit:
            collect(i+1, subset, val, limit)
            return 
        
        # check if we are seeing repeat values and if so skip
        added = subset[::]
        not_added = subset[::]
        added.append(arr[i])
        
        collect(i+1, added, val - arr[i], limit)
        collect(i+1, not_added, val, arr[i])
        
    collect(0, [], num, float("-inf"))
    return result
    
        
        
        
        
        
    
    