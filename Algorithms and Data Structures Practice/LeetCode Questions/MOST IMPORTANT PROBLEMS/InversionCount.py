'''
Count inversions:

The inversion count for an array indicates how far the array is from being sorted. If the array is already sorted, then the inversion count is 0. If the array is sorted in reverse order, then the inversion count is the maximum possible value.

Given an array a, find its inversion count. Since this number could be quite large, take it modulo 109 + 7.

Example

For a = [3, 1, 5, 6, 4], the output should be countInversions(a) = 3.

The three inversions in this case are: (3, 1), (5, 4), (6, 4).

THIS PROBLEM IS THE SAME AS 
countSmallerToTheRight


Use binary index tree/segment tree/merge sort algos


'''

'''
NOT DONE
So elements smaller and to the right
'''

def countInversions(a):
    # you can use merge sort to count ?
    # compare sorted version with unsorted. 
    
    '''
    3 1 5 6 4
    1 3 4 5 6
    
    Circle sort the first one based on the second one, 
    and the length of the circle sort is the answer???
    
    NLOGN
    
    count all elements out of index!
    divide difference by 2. 
    
    You cant use the 
    Sort array, then start from largest element in 
    sorted array go to smallest, 
    and use index to denote #number of elements to right of
    something because then you have to split array 
    each time on each element. 
    
    
    Merge Sort. 
    
    
    Split elements recursively. 
    
    
    
    '''
    
    
    
    
    b = sorted(a)
    
    sorted_m = {}
    
    for idx, i in enumerate(b):
        sorted_m[i] = idx
    
    diff = 0
    for idx, i in enumerate(a):
        out = sorted_m[i]  
        diff += abs(idx - out)
    
    
    
    return diff//2