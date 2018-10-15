'''

[10 marks] Maximum omit-one subrange sum. We consider a variation on the maximum
subrange sum problem studied in class. Let A[1::n] be an integer array, and A[i::j] be
a non-empty subrange. The omit-one sum is the sum of all but the least element in
A[i::j],  The problem asks for the maximum
value of the omit-one sum, Design an  divide-and-
conquer algorithm to solve this problem. You only need to compute the maximum
value, not the subrange it comes from.

Need a justication of correctness and a time complexity analysis.

'''
arr = [1, 4, 3, -1, 7, 2, 3]

def maxOmitOneSubrangeSum(arr):
    # compute max on left and right side, and return 
    # the min from left and right side, and current maximum
    # also need to return if we are touching a boundary
    # if the 2 recursive results are touching the boundary
    # then compare their mins, take the smaller min
    # and combine left and right side
    # return the max of left, max of right, and max of middle as 
    # recursive result along with its min
    if(len(arr) == 1):
        return (0, arr[0])
    left = arr[:len/2]
    
    
    return 2




maxOmitOneSubrangeSum()