

'''

Joe works for a bakery and needs to make $n$ cakes for customers to pick up. Each cake $i$ must be completed by a 
deadline $d_i$, and requires a length $L$ (independent of $i$) that is the amount 
of time Joe needs to make it {\em continuously} before finishing it. The work of 
a cake cannot be interrupted in the middle, and Joe cannot work on more than one 
cake simultaneously. Moreover, to ensure the cake is fresh when the customer 
picks it up, Joe can only start working on a cake no earlier than $d_i-L'$ for 
some $L'>L$ (again, $L'$ is independent of $i$).  During busy seasons Joe cannot 
satisfy all of the customer orders, and will have to select a subset of the 
orders to work on.  Design a greedy algorithm to maximize the number 
of cakes Joe can finish before the deadlines.
'''


'''
PART 2:

IMAGINE THERE ARE 2 BAKERS. RE WRITE ALGO.

'''


a = 4 #L
b = 7 #L2

deadlines = [1,3,2,6,2,5,8,9,11,12,13,15]



import random

def bakeTimesForTwo(deadlines, L, L2):
    # sort times
    sortedDeadlines = sorted(deadlines)
    print("sorted Deadlines is", sortedDeadlines)

    # start as early as possible which is deadline - L2 <= t <=  deadline - L
    assert(L2 > L)

    currTimeJean = 0
    currTimeJoe = 0

    cakesToMake = []
    cakesJoeDoes = []
    cakesJeanDoes = []

    for id, i in enumerate(sortedDeadlines):
        
        latestStartTime = i - L
        earliestStartTime = i - L2
        # calculate a start time!!!!

        if(latestStartTime >= currTimeJean): # i is a viable choice!
            # viable choice!
           
            startTime = max(currTimeJean, earliestStartTime)
            print("start time for ", i,  " is ", startTime)
            cakesToMake.append((id, i) )
            cakesJeanDoes.append((id, i))
            # when do we start baking?
            #earliestStartTime
            currTimeJean = startTime + L
            print("new curr Time JEAN", currTimeJean)
        
        elif(latestStartTime >= currTimeJoe): # i is a viable choice!
            # viable choice!
            startTime = max(currTimeJoe, earliestStartTime)
            print("start time for JOE ",  i,  " is ", startTime)
            cakesToMake.append((id, i))
            cakesJoeDoes.append((id, i))
            # when do we start baking?
            #earliestStartTime
            currTimeJoe = startTime + L
            print("new curr Time JOE", currTimeJoe)

    print("cakes joe does", cakesJoeDoes)
    print("cakes jean does", cakesJeanDoes)
    return len(cakesToMake)

#print(bakeTimesForTwo(deadlines, a, b))





'''
PART 3. THE SECOND BAKER TAKES L' TIME SO HE IS SLOWER. WRITE DP ALGO TOP DOWN, AND BOTTOM DOWN
'''

# doesnt work when one baker is slower than the other


# baker that is faster should be assigned on the sooner orders, and baker that is slower should be assigned on some
# later orders

# d = [5, 6, 10, 12, 16, 20, 28, 20]

ahard = 2
bhard = 3
 


#jean works on 4 :   0->5, 
# joe works on 5 : 0->4, 

def bakeTimesForTwoTopDownWithSlowDownGreedy(deadlines, L, L2):
    # sort times

    sortedDeadlines = sorted(deadlines)
    print("sorted Deadlines is", sortedDeadlines)

    # start as early as possible which is deadline - L2 <= t <=  deadline - L
    assert(L2 > L)

    currTimeJean = 0
    currTimeJoe = 0

    cakesToMake = []
    cakesJoeDoes = []
    cakesJeanDoes = []

    for id, i in enumerate(sortedDeadlines):
        
        latestStartTimeJoe = i - L
        latestStartTimeJean = i - L2

        theMinOfStartTimes = min(currTimeJean, currTimeJoe)

        # calculate a start time!!!!

        # joe goes first then jean
        if(theMinOfStartTimes == currTimeJoe and latestStartTimeJoe >= currTimeJoe): # i is a viable choice!
            # viable choice!
            earliestStartTime = i - L2
            startTime = max(currTimeJoe, earliestStartTime)

            print("LATEST START TIME FOR JOE IS", i, " is ", latestStartTimeJoe)
            print("start time for JOE ",  i,  " is ", startTime)
            cakesToMake.append((id, i))
            cakesJoeDoes.append(([startTime, startTime + L], i))
            
            # when do we start baking?
            #earliestStartTime
            currTimeJoe = startTime + L
            print("new curr Time JOE", currTimeJoe)
            continue

        if(theMinOfStartTimes == currTimeJean and latestStartTimeJean >= currTimeJean): # i is a viable choice!
            # viable choice!
            # earliestStartTime = i - L2
            # startTime = max(i - currTimeJean, earliestStartTime)
            print("start time for jean ", i,  " is ", latestStartTimeJean)

            cakesToMake.append((id, i))
            cakesJeanDoes.append(([currTimeJean, currTimeJean + L2], i))
            # when do we start baking?
            #earliestStartTime

            currTimeJean = currTimeJean + L2 #startTime + L2
            print("new curr Time JEAN", currTimeJean)
            continue
    print("CAKES JOE DOES: ", cakesJoeDoes)
    print("CAKES JEAN DOES: ", cakesJeanDoes)
    print("CAKES TO MAKE ARE: ", cakesToMake)
    return len(cakesToMake)


def bakeTimesForTwoTopDownWithSlowDownGreedySlowOneFirst(deadlines, L, L2):
    # sort times

    sortedDeadlines = sorted(deadlines)
    print("sorted Deadlines is", sortedDeadlines)

    # start as early as possible which is deadline - L2 <= t <=  deadline - L
    assert(L2 > L)

    currTimeJean = 0
    currTimeJoe = 0

    cakesToMake = []
    cakesJoeDoes = []
    cakesJeanDoes = []

    for id, i in enumerate(sortedDeadlines):
        
        latestStartTimeJoe = i - L
        latestStartTimeJean = i - L2

        theMinOfStartTimes = min(currTimeJean, currTimeJoe)


        # calculate a start time!!!!

        if(theMinOfStartTimes == currTimeJean and latestStartTimeJean >= currTimeJean): # i is a viable choice!
            # viable choice!
            # earliestStartTime = i - L2
            # startTime = max(i - currTimeJean, earliestStartTime)
            
            print("start time for jean ", i,  " is ", latestStartTimeJean)

            cakesToMake.append((id, i))
            cakesJeanDoes.append(([currTimeJean, currTimeJean + L2], i))
            # when do we start baking?
            #earliestStartTime

            currTimeJean = currTimeJean + L2 #startTime + L2
            print("new curr Time JEAN", currTimeJean)
            continue


        # joe goes first then jean
        if(theMinOfStartTimes == currTimeJoe and latestStartTimeJoe >= currTimeJoe): # i is a viable choice!
            # viable choice!
            earliestStartTime = i - L2
            startTime = max(currTimeJoe, earliestStartTime)

            print("LATEST START TIME FOR JOE IS", i, " is ", latestStartTimeJoe)
            print("start time for JOE ",  i,  " is ", startTime)
            cakesToMake.append((id, i))
            cakesJoeDoes.append(([startTime, startTime + L], i))
            
            # when do we start baking?
            #earliestStartTime
            currTimeJoe = startTime + L
            print("new curr Time JOE", currTimeJoe)
            continue

  
    print("CAKES JOE DOES: ", cakesJoeDoes)
    print("CAKES JEAN DOES: ", cakesJeanDoes)
    print("CAKES TO MAKE ARE: ", cakesToMake)
    return len(cakesToMake)




print(random.sample([3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16], 8))

d2 = [ 3, 3, 5, 6, 9, 10, 12, 13, 14, 15 ]


listsThatFckedIt = []

for i in range(10000):
    d2 = random.sample([3,  6,  9, 11, 12, 13, 14, 15, 16], 5)

    result1 = bakeTimesForTwoTopDownWithSlowDownGreedy(d2, ahard, bhard)
    result2 = bakeTimesForTwoTopDownWithSlowDownGreedySlowOneFirst(d2, ahard, bhard)

    print(result1)
    print(result2)

    if(result1 != result2):
        listsThatFckedIt.append( (d2, result1, result2))


print(listsThatFckedIt)




# print(bakeTimesForTwoTopDownWithSlowDownGreedy(d2, ahard, bhard))

# print(bakeTimesForTwoTopDownWithSlowDownGreedySlowOneFirst(d2, ahard, bhard))
