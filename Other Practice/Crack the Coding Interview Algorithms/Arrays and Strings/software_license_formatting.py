'''
1
Task description
Task
Python
C
C++
C#
Go
Java
JavaScript
Python
We are given a string S consisting of N characters and an integer K. The string represents a software license key which we would like to format. The string is composed of alphanumerical characters and/or dashes. The dashes split the alphanumerical characters within the string into groups (i.e. if there are M dashes, the string is split into M+1 groups). The dashes in the given string are possibly misplaced.

We want each group of characters to be of length K (except for possibly the first group, which could be shorter, but still must contain at least one character). To satisfy this requirement, we will reinsert the dashes. Additionally, all the lower case letters in the string must be converted to upper case.

For example, in the license key "2-4A0r7-4k" there are two dashes which split it into three groups of lengths 1, 5 and 2, respectively. If we want each group to be of length 4, then we would have to reinsert the dashes. Thus, for K = 4, the correctly formatted string is "24A0-R74K".

Write a function:

def solution(S, K)

that, given a non-empty string S consisting of N characters, representing a license key to format, and an integer K, returns the license key formatted according to the description above.

For example, given S = "2-4A0r7-4k" and K = 4, the function should return "24A0-R74K", and for K = 3, the function should return "24-A0R-74K" as the first group could be shorter. Given S = "r" and K = 1, the function should return "R".

Write an efficient algorithm for the following assumptions:

N and K are integers within the range [1..300,000];
string S consists only of alphanumerical characters (a−z and/or A−Z and/or 0−9) and/or dashes (-);
string S contains at least one alphanumerical character.

'''

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")


from collections import deque 


def solution(S, K):
    # write your code in Python 3.6
    split = S.split("-")    
    joined = "".join(split)
    grouped = deque()
    i = 1
    
    j = 0
       
    ye = deque()
    
    for i in reversed(range(len(joined))):
     
        ye.appendleft(joined[i])
        j += 1
        
        if(j == K):
            grouped.appendleft(ye)
            ye = deque()
            j = 0
    
    if(len(ye) > 0):
        grouped.appendleft(ye)
    
    result = "-".join("".join(group).upper() for group in grouped)
        
    return result