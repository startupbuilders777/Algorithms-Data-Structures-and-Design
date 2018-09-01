# NOT DONE, NEEDS TO BE FIXES BECAUSE SECOND TEST FAILS
# Longest common subsequence:

'''
Examples:
LCS for input Sequences ABCDGH and AEDFHR is ADH of length 3.
LCS for input Sequences AGGTAB and GXTXAYB is GTAB of length 4.

'''



'''
Harmans analysis: 

brute force solution:






go through string1, store each character. 

ok so for a dp type esk solution 


table[i] = longest subsequence to the ith character in both str1 and str2

so we will return table[N] where N is the length of shorter of str1 and str2 (thats the longest possible substring)

ok so we store these partial solutions in the map


so :

go through smaller string.
process first character. (then go through larger array to find that character)  
    -> if not found, process next character in smallerstr
    -> if found, make decision and get the max length of either decision and store in dict -> include in LCS, or dont include in LCS. 
                            -> if included in LCS, the j which traverses the larger array must start from where it found that char.
                                                        otherwise j starts in whatever position it was before 



ok either include it in the LCS or dont include it in the LCS. 2 decisions. 









'''

def lcs(str1, str2):
    dict = {}

    if(len(str1) <= len(str2)):
        smallerStr = str1
        longerStr = str2
    else:
        smallerStr = str2
        longerStr = str1
    
    def lcsRecur(short, long, smallIdx, dict):
        if(smallIdx == len(short)): # base case 1
            return 0

        if(len(long) == 0): # base case 2
            return 0

        if(dict.get(smallIdx) is not None):
            return dict.get(smallIdx)

        c = short[smallIdx] # process a char
        
        foundIdxLong = -1
        for i in range(len(long)):
            if long[i] == c:
                # found,
                foundIdxLong = i
                break

        if(foundIdxLong == -1):
            # DIDNT find that character, go to next character in short for processing
            return lcsRecur(short, long, smallIdx + 1, dict)
        
        # found!

        dict[smallIdx] = max(
                   lcsRecur(short, long, smallIdx + 1, dict) , 
                   lcsRecur(short, long[foundIdxLong : ], smallIdx  + 1, dict) + 1
                  )
         
        
        return dict[smallIdx]


    solution =  lcsRecur(smallerStr, longerStr, 0, dict)
    print("lcs dict: ", dict)
    return solution



print("lcs of ABCDGH and ADH", lcs("ABCDGDH", "ADH") ) # should be 3

print("lcs of AGGTAB and GXTXAYB ", lcs("AGGTAB", "GXTXAYB") ) # should be 4

