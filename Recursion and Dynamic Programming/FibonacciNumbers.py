
'''Normal way, slow'''
'''Runtime: T(n) = T(n-1) + T(n-2) + O(1) approx O(1.6^n)'''
def fibExpTime(i):
    if i == 0:
        return 0
    if i == 1:
        return 1
    else:
        return fibExpTime(i-1) + fibExpTime(i-2)

'''This is top down dynamic programming'''
'''Use memoization and cache the results'''
'''If you use an accumulator, you can make this even faster'''
def fibLinearTime(i):
    dict = {}
    return fibMemRecur(i, dict)

def printDictionary(dictArg):
    for key, value in dictArg.items():
        print("The key is " + str(key) + " and the value is " + str(value))
    for key in dictArg.keys():
        print("The key is " + str(key))
    for value in dictArg.values():
        print("the value is " + str(value))

def fibMemRecur(i, memDict):
    if(i == 0 or i == 1):
        #print("ayy")
        return i
    elif(memDict.get(i) == None):
        memDict[i] = fibMemRecur(i-1, memDict) + fibMemRecur(i-2, memDict)
        print(memDict) # <- Prints the whole dictionary automatically
        #printDictionary(memDict)
        #print("saving " +  str(i))
        return memDict[i]
    else:
        return memDict.get(i)

#print(fibLinearTime(120))
#print(fibExpTime(120))

'''bottom up dynamic programming'''

def fibBottomUp(i):
    if i == 0 :
        return 0
    a = 0
    b = 1
    for k in range(2, i):
        c = a + b
        a = b
        b = c
    return a+b

#print(fibBottomUp(1200)) <- Kind of like an accumulator var

