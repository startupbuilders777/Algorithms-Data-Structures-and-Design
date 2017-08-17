'''Write a method to return all subsets of a set'''

#This is a set
A = {"a", "b", "c"}

#Result -> {{}, {a}, {b}, {c}, {a,b} {a,c} {b,c} {a,b,c}}

def all_subsets(setS):
    size = len(setS)
    #Turn the set into a map that maps an index to a element


def chooseRecur(setS, amtLeft, newSet):
    allChoices = {}
    if(amtLeft == 0):
        return newSet
    for i in setS:
        setA = {}
        setA.add(i)
        setS.remove(i)
        sets = buildSet(setA, amtLeft-1, setElements=setS)
        allChoices = allChoices.union(sets)

def buildSet(setA, amt, setElements):

