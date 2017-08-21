'''

Input: n intervals [s1, f1], [s2, f2], ... , [sn, fn]
Output: a maximum set of disjoint intervals


Performance O(nlogn)
'''

'''
Greedy Solution Decision Criteria: Use the earliest finishing time
'''

intervals = {
    (1,2), (1, 7), (2, 6), (4, 5), (5, 9), (9,11),(6,9), (7,9), (4,7),(3,4),(4, 12), (1, 3), (3, 5), (2, 6), (2, 7)
}
a = (1,2)

def intervalKey(interval):
    return interval[1]

def intervalScheduling(setOfIntervals):
    lst = []
    for i in setOfIntervals:
        lst.append(i)

    lst.sort(None,intervalKey)
    print(lst)
    selectedIntervals = [lst[0]]
    currentInterval = lst[0]
    #currentInterval[1] is the finishing time and currentInterval[0] is the starting time
    for i in range(1, len(lst)):
        if(lst[i][0] >= currentInterval[1]):
            selectedIntervals.append(lst[i])
            currentInterval = lst[i]

    print(selectedIntervals)

intervalScheduling(setOfIntervals=intervals)