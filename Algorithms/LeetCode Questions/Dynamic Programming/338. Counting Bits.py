'''
Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate 
the number of 1's in their binary representation and return them as an array.

Example 1:

Input: 2
Output: [0,1,1]
Example 2:

Input: 5
Output: [0,1,1,2,1,2]
Follow up:

It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
Space complexity should be O(n).
Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.

'''

class Solution:
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        # You should make use of what you have produced already.
        # Divide the numbers in ranges like [2-3], [4-7], [8-15] and so on. And try to generate new range from previous.
        # Or does the odd/even status of the number help you in calculating the number of 1s?

        # even means even number of ones
        # odd means odd number of ones
        
        # NUMBER OF ONES IN BINARY REPRESENTATION.
        # take number, 5 => count 1's that is 101
        # maybe build numbers up to that number using single pass process
        # 0 1 10 11 100 101 110 111 1000 1001 1010 1011 1100 1101 1110 1111    10000 10001 10010 10011 10100 11000 11001 11010 11011 11100 11101 11110 11111
        # 0 1 1  2      1 2 2 3       1 2  2  3  2  3   3 4    1 2 2  2 2   1 2 2 3 2 3 3 4 3 4 4 5
        # 0 1 2    3      4 5 6 7       8 9 10  11 12 13 14 15
        # i see repetition of solutions in solutions
        # 2^0 2^1 2^2 2^3 2^4
        # 1   2    4   8   16 => patterns repeat between these ranges
        
        # [0, 2^0) 0 
        # [2^0, 2^1) 1 
        # [2^1, 2^2) 2 3 {2 maps to 0 (add one for next range so, bits == (bits of 0 + 1 which is ) 1), 3 maps to 1, since 3 is odd +1 so bits == 2}
        # [2^2, 2^3) 4 5 6 7 => {4 maps to 0 (add one to it, ), 5 maps to 1(add 1 since odd, ),} 
        
        # OK I FIGURED OUT THE PATERN, YOU JUST ADD 1 TO EVERYTHING IN THE PREVIOUS RANGE: examine bits: 
         # 0 
         # 1 
         # 1  2
         # 1 2 2 3
        #   1 2  2  3  2  3   3 4 (this row was constructed by adding 1 to each element in the array before this)
        
        oneBits = [0] # BASE CASE
        startExponent = 0
        i = 0
        
        k = 0 # k will cycle between 0 and 2^x and increment x each time, grabbing old values from the oneBits array
        
        while(i != num):
                # so you can append 2^startExponent-1 elements
                # if odd add 1, if even dont add 1.
            if(k != 2 ** startExponent):
                i += 1 # adding the bits for this value
                print(2^startExponent)
                print(k)
                oneBits.append(oneBits[k] + 1)
                
                k += 1
            else:
                k = 0
                startExponent += 1
                
        
        return oneBits

# FASTER

class Solution:
    """
    Given a non negative integer number num. 
    For every numbers i in the range 0 ≤ i ≤ num 
    calculate the number of 1's in their binary representation 
    and return them as an array.

    Example 1:
    Input: 2
    Output: [0,1,1]

    Example 2:
    Input: 5
    Output: [0,1,1,2,1,2]


    1.  It is very easy to come up with a solution with run time O(n*sizeof(integer)). 
         But can you do it in linear time O(n) /possibly in a single pass?
    
    2.  Space complexity should be O(n).

    3.  Can you do it like a boss? Do it without using any builtin function 
        like __builtin_popcount in c++ or in any other language.
    """
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        return self.firstTry(num)

    def firstTry(self, num):
        """ val: 0  1   2   3   4    5    6    7
            bin: 0  1   10  11  100  101  110  111
                 ----
                 ------------
                 ---------------------------------
            just add 1 to previous result

            space: O(n)
            time: O(n)
        """
        dp = [0, 1]
        if num <= 1:
            return dp[:num+1]

        curr = 1
        while True:
            size = len(dp)
            for i in range(size):
                dp.append(dp[i] + 1)
                curr += 1
                if curr == num:
                    return dp

# FASTEST SOLUTION

class Solution:
    def countBits(self, num):
        """
        :type num: int
        :rtype: List[int]
        """
        ret = [0]
        k = 1
        while k*2 <= num:
            ret += [r + 1 for r in ret]
            k *= 2
        ret += [r+1 for r in ret[:num-k+1]]
        return ret