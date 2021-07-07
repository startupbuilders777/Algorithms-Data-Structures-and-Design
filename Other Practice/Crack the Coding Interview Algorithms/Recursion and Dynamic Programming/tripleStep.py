# PROBLEM DONE

'''unoptimized'''
def tripleStep(n):
    if(n == 0):
        return 1
    ways1, ways2, ways3 = 0, 0, 0
    if(n >= 3):
        ways3 = tripleStep(n-3)
    if(n >= 2):
        ways2 = tripleStep(n-2)
    if(n >= 1):
        ways1 = tripleStep(n-1)
    return ways1 + ways2 + ways3

#print(tripleStep(200))

'''Top Down dynamic programming'''
def tripleStepTopDown(n):
    dict = {}
    return tripleStepTopDownRecur(n, dict)

def tripleStepTopDownRecur(n, dict):
    if n == 0:
        return 1
    if dict.get(n) is not None:
        return dict[n]
    else:
        ways1, ways2, ways3 = 0, 0, 0
        if n >= 3:
            ways3 = tripleStepTopDownRecur(n - 3, dict)
        if n >= 2:
            ways2 = tripleStepTopDownRecur(n - 2, dict)
        if n >= 1:
            ways1 = tripleStepTopDownRecur(n - 1, dict)
        tot = ways1 + ways2 + ways3
        dict[n] = tot
        return tot

print(tripleStepTopDown(200))