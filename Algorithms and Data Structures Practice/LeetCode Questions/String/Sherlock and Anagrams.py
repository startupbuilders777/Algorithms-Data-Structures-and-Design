'''
Two strings are anagrams of each other if the letters of one string can be rearranged 
to form the other string. Given a string, find the number of pairs of substrings 
of the string that are anagrams of each other.

For example , the list of all anagrammatic pairs is  at positions  respectively.

Function Description

Complete the function sherlockAndAnagrams in the editor below. It must return an integer that represents the number of anagrammatic pairs of substrings in .

sherlockAndAnagrams has the following parameter(s):

s: a string .
Input Format

The first line contains an integer , the number of queries.
Each of the next  lines contains a string  to analyze.

Constraints



String  contains only lowercase letters  ascii[a-z].

Output Format

For each query, return the number of unordered anagrammatic pairs.

Sample Input 0

2
abba
abcd
Sample Output 0

4
0
Explanation 0

The list of all anagrammatic pairs is  and  at positions  and  respectively.

No anagrammatic pairs exist in the second query as no character repeats.

Sample Input 1

2
ifailuhkqq
kkkk
Sample Output 1

3
10
Explanation 1

For the first query, we have anagram pairs  and  at positions  and  respectively.

For the second query:
There are 6 anagrams of the form  at positions  and .
There are 3 anagrams of the form  at positions  and .
There is 1 anagram of the form  at position .

'''

# SOLUTION IS TO COUNT CHARACTERS THROUGH PREPROCESSING AND TAKING DIFFERENCES, AND USING A COUNTER MAP

def sherlockAndAnagrams(s):
    '''
    Need to iterate through all substrings, 
    and have counts of number of characters. 

    Can keep running count of total characters seen, 
    then find number of chracters in a substring by doing count difference. 
    '''

    tot = [ ]
    cs = [0 for i in range(26)] 
    tot.append(cs[::])

    for idx, i in enumerate(s): 
        cs[ord(i) - ord('a')] += 1
        tot.append(cs[::])

    def substr(a, b):
        res = []
        for i, j in zip(a,b):
            res.append(i-j)
        return res 
    N = len(s)

    # cum arr has size N + 1. [0,1] -> A
    m = defaultdict(int)
    c = 0
    for i in range(N):
        for j in range(i+1, N+1):
            chars = str(substr(tot[j], tot[i]))

            if m.get(chars) is not None:
                c += m.get(chars)
                m[chars] += 1  
            else:
                m[chars] = 1
    return c
