'''

We just want to find a room we can book an interval in:
Input interval is [start, end)

The intervals below are closed bracket then open bracket.
Specify room that has space and if you cant find one, return -1

[
    [ [1,3), [4, 5), [8, 12) ],
    [ [1, 2), [5, 6)]
    [ [4, 12), [12, 15) ]
]

USE BINARY SEARCH
'''

from bisect import bisect_left, bisect_right

data = [ [[1,3], [4, 5], [8, 12] ],
    [ [1, 2], [5, 6]],
    [ [4, 12], [12, 15] ]
]


def soln(rooms, start, end):
    '''
    Intervals are sorted. 
    
    Binary
    search for start location, using the end times. 
    Because we can only start right after something started. 
    
    
    
    Binary search for end location using start times.
    we can only end before the other things have started. 
    
    if we get 2 different values from each binsearch, 
    we prolly cannont insert it,
    otherwise we can
    
    
    if its different values means there is an interval in the middle
    
    '''
    
    startTimes = [[room[0] for room in AroomList] for AroomList in rooms]
    endTimes = [[room[1] for room in AroomList] for AroomList in rooms]
    
    # print("start times, end times", startTimes, endTimes)
    
    for roomIdx, (startTime, endTime) in enumerate(zip(startTimes, endTimes)):
        
        # Bisect left locates insertion point in array to maintain sorted order. 
        # And left vs right depends on what happens if you see the same entries
        # in the list. 
        # bisect_right means find the insertion point to the right of existing entries
        # bisect_left means find the insertion point to the left of existing entries. 
        
        startLoc = bisect_right(endTime, start)
        endLoc = bisect_left(startTime, end)
        # print("START LOC, END LOC", startLoc, endLoc)
        
    
        if startLoc == endLoc:
            return roomIdx + 1
            
    return -1

        
    
# should return room 3
print(soln(data, 3,4 )) # prints 1 
print(soln(data, 3,5 )) # prints 2 
print(soln(data, 2,5 )) # prints 2 


print(soln(data, 5,9 )) # returns -1 
print(soln(data, 1,4 )) # returns 3

print(soln(data, 12,15 )) 

print(soln(data, 3,13 )) # prints -1
print(soln(data, 1,5 )) # should print -1
    
    
    



    

