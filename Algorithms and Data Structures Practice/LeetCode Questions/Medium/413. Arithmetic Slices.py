'''
DONE

413. Arithmetic Slices
Medium

A sequence of number is called arithmetic if it consists 
of at least three elements and if the difference between 
any two consecutive elements is the same.

For example, these are arithmetic sequence:

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9

The following sequence is not arithmetic.

1, 1, 2, 5, 7

A zero-indexed array A consisting of N numbers is given. 
A slice of that array is any pair of integers (P, Q) such that 0 <= P < Q < N.

A slice (P, Q) of array A is called arithmetic if the sequence:
A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. 

In particular, this means that P + 1 < Q.

The function should return the number of arithmetic slices in the array A.


Example:

A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.


'''

class Solution(object):
    def numberOfArithmeticSlices(self, A):
        """
        :type A: List[int]
        :rtype: int
        """
        
        
        '''
        Count all the arithmetic sequences in the start of the array.
        Grow it to the right. 
        (Capture all subarrays! of an arithmetic sequence longer than 3)
        
        When you cant. stop and try again from that element. 
        
        ACTUALLY WE just count it too!
        
        so if there are 4 elements. there are 3 sequences.
        if there are 5 elements there are 4 sequences.
        
        <-> keep expandign arithmetic sequences from the left. capture longest, 
        than start again. 
        
        
        4 elements -> 1 + 2 = 3
        
        5 elements -> 1 + 2 + 3 = 6
        6 elements = 1 + 2 + 3 + 4 = 10
        
        7 -> 15 
        
        n(n+1) / 2 -> (n-1)*(n-2)/2
        4 -> 3*2/2 = 3
        7 -> 6*5/2 = 15
        
        '''
            
        if len(A) == 0:
            return 0
        
        i = 1
        curr_len = 1
        
        diff = None
        count = 0
        prev = A[0]
        
        while i < len(A):
            nxt = A[i]
            
            if(diff == None):
                diff = nxt - prev
                curr_len += 1
            elif(prev + diff == nxt):
                curr_len += 1
            else:
                diff = nxt - prev
                
                if(curr_len >= 3):
                    count += (curr_len - 1)*(curr_len - 2)/2
                
                curr_len = 2
            
            print("next", nxt)
            
            i += 1
            prev = nxt
        
        if(curr_len >= 3):
            print(curr_len)
            count += (curr_len - 1)*(curr_len - 2)/2
            
        return count
        