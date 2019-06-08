'''

922. Sort Array By Parity II

Share
Given an array A of non-negative integers, half of the integers 
in A are odd, and half of the integers are even.

Sort the array so that whenever A[i] is odd, i is odd; and 
whenever A[i] is even, i is even.

You may return any answer array that satisfies this condition.

 

Example 1:

Input: [4,2,5,7]
Output: [4,5,2,7]
Explanation: [4,7,2,5], [2,5,4,7], [2,7,4,5] would also have been accepted.
 

Note:

2 <= A.length <= 20000
A.length % 2 == 0
0 <= A[i] <= 1000
'''

class Solution:
    def sortArrayByParityII(self, A: List[int]) -> List[int]:
        # Need even -> odd -> even  -> odd
        
        # Find a misplaced element, then find the opposite misplaced element, swap, keep going. 
        # what if we find an even in an odd position, then see another even in an odd position?
        # then we keep seeing evens in odd position, then near the end see odd in even positon????
        
        # WELLL!!!!
        # OKAY FIGURED IT OUT!!!
        
        
        i = 0 
        j = 1
        
        while i < len(A) and j < len(A):
            
            if(A[i] % 2 == 0 and A[j] % 2 == 1):
                i += 2
                j += 2
            elif(A[i] % 2 == 0):
                i += 2
            elif(A[j] % 2 == 1):
                j += 2 
            elif(A[i] % 2 == 1 and A[j] % 2 ==0): # Value at i is odd!!!
                A[i], A[j] = A[j], A[i]
            else:
                print("error?")
        
        return A

