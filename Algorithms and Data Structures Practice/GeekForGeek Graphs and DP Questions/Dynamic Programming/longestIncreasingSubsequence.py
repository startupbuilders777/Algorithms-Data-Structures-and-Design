
'''

The Longest Increasing Subsequence (LIS) problem is to find the length of the longest subsequence 
of a given sequence such that all elements of the subsequence are sorted in increasing order. 
For example, the length of LIS for {10, 22, 9, 33, 21, 50, 41, 60, 80} is 6 and LIS is {10, 22, 33, 50, 60, 80}.

Input  : arr[] = {3, 10, 2, 1, 20}
Output : Length of LIS = 3
The longest increasing subsequence is 3, 10, 20

Input  : arr[] = {3, 2}
Output : Length of LIS = 1
The longest increasing subsequences are {3} and {2}

Input : arr[] = {50, 3, 10, 7, 40, 80}
Output : Length of LIS = 4
The longest increasing subsequence is {3, 7, 40, 80}
'''

'''
Go through each value in the array.
Get a map for DP


start with 0. 
start with index 0. 

map will store for index 0, its LIS.

map will store tuples of this (currMax, currLongestSeq)


subproblem i is max( <newCurrMax, 1 + longestSequence(i-1)> , <oldCurrMax, longestSequence(i-1)> )
where oldCurrMax < newCurrMax



then index 1, 2, 3

for index 4 => check if its value is greater than the current maxvalue from [0, 3]. 

Optimal Substructure:
Let arr[0..n-1] be the input array and L(i) be the length of the LIS ending at index i such that arr[i] is the last element of the LIS.
Then, L(i) can be recursively written as:
L(i) = 1 + max( L(j) ) where 0 < j < i and arr[j] < arr[i]; or
L(i) = 1, if no such j exists.
To find the LIS for a given array, we need to return max(L(i)) where 0 < i < n.
Thus, we see the LIS problem satisfies the optimal substructure property as the main problem can be solved using solutions to subproblems.


'''


def longestIncreasingSubsequence(arr):
    m = {};

    # solve returns the value.
    def solve(arr, index): 
        if (m[index]):
            return m[index][1] #get the val for that


        oldMax = m[index-1]; 
        
        val = arr[index];
        
        if val > oldMax:
            m[index] = max(solve(arr, index+1),  )




    return solve(arr, len(arr) - 1);