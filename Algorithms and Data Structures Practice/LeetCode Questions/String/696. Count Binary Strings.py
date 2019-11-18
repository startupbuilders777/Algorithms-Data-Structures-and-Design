'''
696. Count Binary Substrings
Easy

735

132

Favorite

Share
Give a string s, count the number of non-empty (contiguous) substrings 
that have the same number of 0's and 1's, and all the 0's and all the 1's 
in these substrings are grouped consecutively.

Substrings that occur multiple times are counted the number of times they occur.

Example 1:
Input: "00110011"
Output: 6
Explanation: There are 6 substrings that have equal number of consecutive 
1's and 0's: "0011", "01", "1100", "10", "0011", and "01".

Notice that some of these substrings repeat and are counted the number of times they occur.

Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.
Example 2:
Input: "10101"
Output: 4
Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number 
of consecutive 1's and 0's.
Note:

s.length will be between 1 and 50,000.
s will only consist of "0" or "1" characters.

'''

class Solution(object):
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        
        '''
        Process left to right.
        
        
        If you see zeros, start counting zeros.
        Then if you see ones, start counting 1's 
        until u see zero again, or you count more zeros
        
        
        so if you see 
        
        000111 -> count this as 3 -> 000111, 0011, 01, 
        
        Then after that swap, count 0's again.
        
        
        Do the same matching.
        
        Done
        
        
        '''
        
        zeros = 0
        ones = 0
        
        start = False
        counting_zeros = False
        
        ans = 0
        start_idx = 0
        
        # Ok first start by counting whats at the start of s:
        
        for idx, k in enumerate(s):
            start_idx = idx
            k = int(k)
            
            if not start and k == 0:
                counting_zeros = True
                zeros += 1
                start = True       
            elif not start and k == 1:
                counting_zeros = False
                start = True
                ones += 1
            elif start and k == 0 and counting_zeros: 
                zeros += 1
            elif start and k == 0 and not counting_zeros:
                break
            elif start and k == 1 and counting_zeros:
                break
            elif start and k == 1 and not counting_zeros:
                ones += 1
            else:
                print("ERROR WITH INITIALIZATION")
                print(start, k, counting_zeros)
            
        
                
        
        print("start idx", start_idx)
        include_end = False
        
        for idx in range(start_idx, len(s)):
            i = int(s[idx])
            # print(i)
            if counting_zeros and i == 0:
                zeros += 1
                include_end = True        
                    # if zeros == ones and zeros > 1:
                    # ok so we reached the max, now pair em up.
            if not counting_zeros and i == 1:
                ones += 1
                include_end = True
                    
            elif not counting_zeros and i == 0:
                # Okay so check how many 1's we saw, 
                # then pair em up with previous zeros we saw!!
                #print("ONES AND ZEROS SAW IS ", (ones, zeros))
                ans += min(ones, zeros)
                zeros = 1
                counting_zeros = True
                include_end = False
            elif counting_zeros and i == 1:
                ans += min(ones, zeros)
                # print("ONES AND ZEROS SAW IS ", (ones, zeros))
                ones = 1
                counting_zeros = False
                include_end = False
        
        # OK now we have to add the end piece as well!!!
        # BUT WE CANT DOUBLE COUNT THE END PIECE!!!
        #if include_end:
        
        # print("ONES AND ZEROS SAW IS ", (ones, zeros))
        ans += min(ones, zeros)
        
        return ans

                    
                    
                    
                    
                    
