

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
b = 6 #L2

deadlines = [1,3,2,6,2,5,8,9,11,12,13,15]

def bakeTimes(deadlines, L, L2):
    # sort times

    sortedDeadlines = sorted(deadlines)
    print("sorted Deadlines is", sortedDeadlines)

    # start as early as possible which is deadline - L2 <= t <=  deadline - L
    assert(L2 > L)

    currTime = 0

    cakesToMake = []

    for id, i in enumerate(sortedDeadlines):
        
        latestStartTime = i - L
        # calculate a start time!!!!

        if(latestStartTime >= currTime): # i is a viable choice!
            # viable choice!

            print("SO THE LATEST START TIME COULD have BEen: ", i,  " is ", latestStartTime)
            earliestStartTime = i - L2

            # we can start between from (i-L) to (i - L2) to start baking the cake
            
            startTime = max(currTime, earliestStartTime)
            # i is the deadline for the cake we are on
            # currTime is the time now
            # a + L <= d
            #                 a    b                                   a                                      c          d

            print("but the start time ended up being ", i,  " is ", startTime)
            cakesToMake.append((id, i))
            
            # when do we start baking?
            #earliestStartTime
            
            currTime = startTime + L
            print("new curr Time", currTime)


    return cakesToMake

print(bakeTimes(deadlines, a, b))




