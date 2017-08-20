'''
Implement an algorithm to determine if a string has all unique characters.

What if you cannot use additional data structures.
'''

'''
Should i implement a set class first?

O(N) time and O(N) space
'''
def uniqueCharacters(a):
    data = {}
    for i in a:
        if data.get(i) is not None:
            return False
        else:
            data[i] = 1
    return True

print(uniqueCharacters("fook"))
print(uniqueCharacters("foa"))

'''

What if you cannot use any additional data structures 
Slow way -> O(n^2)
Fast way -> 
    Use a sorting algo that sorts it but it cannot use any additional space DOEEEE space has to be O(1) 
        -> You can only use this way if you are allowed to modify the string


'''
str  = "fook"

def sortString(strA):



'''String doesnt have a sorting method need one in O(1)'''
def uniqueCharactersNoDatastructure(strA):
    strA.sort()
    prevChar = ""
    for i in strA:
        if(i == prevChar):
            return False
        else:
            prevChar = strA[i]

    return True

print(uniqueCharactersNoDatastructure("fook"))
print(uniqueCharactersNoDatastructure("foa"))

