'''
Given an array of integers, construct a tree. Each node of the tree has either 2 children or none,
in which case it is a leaf node. A leaf node costs 0 to construct. The cost to build  a parent node is the 
product of the maximum leaf values in its left and right sub trees . 
Parititon array to minimze the cost of building the entire tree




 
n = 3, arr = [4, 6, 2]
1 Possible choices to split the array: {4}, {6, 2} => cost 24 + 12 = 36


other possibl choice -> {4, 6} , {2}  => max({4,6}) * 2 = 12 to 
            create root node and parent node of 4, and 6 is 24, total cost 36.

Choose the minium of these 2 which is 36
'''



def treeCost(arr) :
    # gutta split the array into pieces
    
    # ok to get all possible partitions, either take 1 or take 2 elements, 2 recursive calls.   
    # then onces youve taken all the elements, need a function to calculate the cost of that partition.
    # then return cost. Pick the min of these costs. 

    def partitions(arr):
        if(len(arr) == 0):
            return []
        elif(len(arr) <= 2):
            return  [ [[*arr]] ]
        else: #arr has length of 2 or more
            copy = arr[::]
            leaf1 = copy.pop() 
            aresult = partitions(copy)
            singleLeafPlusPartitions = []
            doubleLeafPartitions = []
            singleLeafPlusPartitions = [ [[leaf1]] + i for i in aresult]

            leaf2 = copy.pop()
            bresult = partitions(copy)
            doubleLeafPartitions = [ [[leaf1, leaf2]] + i for i in bresult]
            
            return [*singleLeafPlusPartitions , *doubleLeafPartitions]


    # tree cost
    print(partitions(arr))
    # go through partitions and calculate cost for each, then pick the best one





treeCost([4, 6, 3, 2])




    

