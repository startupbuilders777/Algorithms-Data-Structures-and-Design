# COMPLETED
'''

[10 marks] Weighted Longest Common Subsequence. Let S = s1s2 sm and T =
t1t2 tn be two strings. Additionally, assume each letter a in the alphabet E is
associated with a weight w(a) that is given to you as part of the input. The weight
of a string is the sum of the weights of all its letters. So if vowels are weighted 2 and
consonants 5, the weight of example is 26.
Give an efficient algorithm to compute a common subsequence of the two input strings
having the largest weight. If there are multiple subsequences with the same weight,
you only need to return one.


'''

'''

Algo:



'''

wfnMatchings = {'a': 1, 'b': 2, 'c': 3, 'd': 4}


def wfn(c):
    if(wfnMatchings.get(c) is None):
        return 0 
    else:
        return wfnMatchings[c]


# DO TOP DOWN FIRST THEN BOTTOM UP.
def weightedLCS(str1, str2, wfn): 
    m = {}

    def weightedLCSTD(str1, i1, str2, i2, match, m, wfn):
        if(m.get((i1, i2))):
            return m[(i1, i2)]      

        if(len(str1) == i1):
            return (0, match)
        if(len(str2) == i2) :
            return (0, match)
        
        charA = str1[i1]
        charB = str2[i2]

        if(charA == charB):
            res = weightedLCSTD(str1, i1 + 1, str2, i2 + 1, match + charA, m, wfn)   
            matchingResult = (res[0] + wfn(charA), res[1]) 
            m[(i1, i2)] = matchingResult
            return matchingResult

        result = max(weightedLCSTD(str1, i1 + 1, str2, i2, match, m, wfn),
                     weightedLCSTD(str1, i1, str2, i2 + 1, match, m, wfn),
                     key=lambda i: i[0]) 
        # print(result)

        m[(i1, i2)] = result
        return result 

    result = weightedLCSTD(str1, 0, str2, 0, "", m, wfn)
    # print(len(m))

    return result


a = "abcbaaad"
b = "fffcdaaffd"

e = "ddd"
f = "ddd"

g = "baad"
h = "dbaa"

m = "bcadcbadcbadcbacdabcdbcabda"
n = "babbdabdbcbdcabcbacdbabcb"

'''
   So the string returned by result works when you dont memoize. Works when you do memoize. 
'''
print(weightedLCS(g, h, wfn))

# print(weightedLCS(e, f, wfn))
# print(weightedLCS(g, h, wfn))
# print(weightedLCS(m, n, wfn))

# TOP DOWN COMPLETED.

# DO BOTTOM UP
# need to build table of solutions

    # so we have the following
    
'''
        we have to work backwards from the end of the two strings to the start and use the following:
        For strings x and y, let X and Y represent the length of the strings.
        
        a(i, j) represents the max weighted LCS for characters from x[1..i] and y[1..j] where 0 <= i <= X   
                                                                                              0 <= j <= Y
        We want to return the solution:  a(X,Y)

        Therefore:

                      / 0 if i = 0, or j = 0       
          a(i,j) =   | a(i-1, j-1) + w(fn) if x[i] = y[j]
                      \ max(a(i-1, j), a(i, j-1))
'''


def weightedLCSBP(str1, str2, wfn):
    
    # add base cases
    A = [[] for i in range(len(str1)) ]

    for i in range(len(str1)):
        # print(A[i])
        for j in range(len(str2)):
            if str1[i] == str2[j]:
                A[i].append([wfn(str1[i]),str2[j]])
            else:
                A[i].append([0, ""])

    # print(A)

    for i in range(1, len(str1)):
        for j in range(1, len(str2)):
            if(str1[i] == str2[j]):
                A[i][j][0] = A[i - 1][j - 1][0] + wfn(str1[i])
                A[i][j][1] = A[i - 1][j - 1][1] + str1[i]
            else:
                if A[i-1][j][0] >= A[i][j-1][0]:
                    A[i][j][0] = A[i - 1][j][0]
                    A[i][j][1] = A[i - 1][j][1]
                else:
                    A[i][j][0] = A[i][j-1][0]
                    A[i][j][1] = A[i][j-1][1]

                # A[i][j][0] = max(A[i - 1][j][0], A[i][j - 1][0])
    # print(A)
    return A[len(str1) - 1][ len(str2) - 1][1]

print(weightedLCSBP(g, h, wfn))

'''

[[0, 2, 2, 2], 
 [0, 2, 3, 3], 
 [0, 2, 3, 4], 
 [8, 8, 8, 8]]

'''