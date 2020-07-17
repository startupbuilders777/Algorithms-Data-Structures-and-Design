


# DP PROBLEM:
'''

A[i] represents the max cakes that can be produced if we consker all cakes from deadline 0 to deadline i


A[0] = 0

A[1] = 1 # 1 cake we make it
A[2] = 2 # 2 cakes, joe makes one, and jean makes one.


A[k] = A[k-1] + 1 if Jean Bakes kth Cake when currTimeJean < currTimeJoe (Jean is available to make a cake) and currTimeJean = time after making cake
A[k] = A[k-1] + 1 if Joe Bakes kth Cake when currTimeJoe < currTimeJean (Joe is available to make a cake) and currTimeJoe = time after making cake
A[k] = max(A[k-1] + 1 and Jean Makes cake , A[k-1] + 1 and Joe Bakes Cake) when currTimeJean == currTimeJoe


A(deadlineIndex) = max()
'''





def bakeTimesDP(deadlines, L, L2):
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
    amountOfCakesWeCook = [0] * len(sortedDeadlines)

    amountOfCakesWeCook[len(sortedDeadlines) - 1] = 1
    
    # we have to process backwards
    for k, i in len(sortedDeadlines) to 0:
        
        if(currTimeJean == currTimeJoe): #if both can go we have to decide whose better
            amountOfCakesWeCook[k] = max(amountOfCakesWeCook[k-1], amountOfCakesWeCook[k-1])
        


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
            cakesToMake.append((k, i))
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

            cakesToMake.append((k, i))
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

    for k, i in enumerate(sortedDeadlines):
        
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
            cakesToMake.append((k, i))
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

            cakesToMake.append((k, i))
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
