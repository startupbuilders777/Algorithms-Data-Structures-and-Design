'''
191. Number of 1 Bits
Easy


Share
Write a function that takes an unsigned integer and 
return the number of '1' bits it has (also known as the Hamming weight).

Example 1:
Input: 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.

Example 2:
Input: 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.
Example 3:

Input: 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.

Follow up:

If this function is called many times, how would you optimize it?

'''


class Solution:
    #Slow way
    def hammingWeightSlow(self, n: int) -> int:
        # count it up!
        
        count = 0
        while n != 0: 
            count += (n & 1)
            n = n >> 1
        return count
    
    '''
    We can make the previous algorithm simpler and a little faster. 
    Instead of checking every bit of the number, we repeatedly flip 
    the least-significant 1-bit of the number to 0, and add 1 to the sum. 
    As soon as the number becomes 00, we know that it does 
    not have any more 11-bits, and we return the sum.

    The key idea here is to realize that for any number n, 
    doing a bit-wise AND of nn and n - 1 flips the least-significant 1-bit in n to 0.
    
    '''
    def hammingWeight(self, n: int) -> int:
        # count it up!
        
        count = 0
        while n != 0: 
            count += 1
            n = n & (n-1)
        return count

# To optimizie you can cache?


