

a = "CS34Alg A35orit m41Jeff BinMath junklin e2makei tsquare"
traversalListExample = [ list(i) for i in a.split()]
path = "CS341"
path2= "sadas"
print(traversalListExample)

# traverse from upper left corner to bottom right:

# TOP DOWN
def traverse(maze, path):
    m = {
        
    } 

    # the location in the board is a good key for the DP
    # 2D DP can be done
    # Some of the base cases are on the edges of the map

    leftToRightDistance = len(maze[0])
    topDownDistance = len(maze)
    # DONT NEED TO STORE PATH IN MAP, JUST STORE TRUE OR FALSE
    def move(currX, currY, path, pathIndex, m, moveSequence):
        if(pathIndex == len(path)):
            return (True, moveSequence)
        
        key = (currX, currY, pathIndex)

        if(m.get(key) is not None ):
            return m[key]

        theNode = path[pathIndex]
        if(theNode != maze[currY][currX]):
            m[key] = (False, [])     
            return (False, [])


        if(currX + 1 < leftToRightDistance):
            result1, path1 = move(currX + 1, currY, path, pathIndex+1, m, moveSequence + [(currY+1, currX+1)])
            if(result1):
                m[key] = (result1, path1)
                return (result1, path1) 
        
        if(currY + 1 < topDownDistance):
            result2, path2 = move(currX, currY+1, path, pathIndex+1, m, moveSequence + [(currY+1, currX+1)] )
            if(result2):
                m[key] = (result2, path2)
                return (result2, path2)
        
        m[key] = (False, [])
        return (False, [])

    
    result =  move(0, 0, path, 0, m,[])
    print(m)
    return result

# BOTTOM UP
print(traverse(traversalListExample, path))





# BOTTOM UP IMPLEMENTATION

def traverseBU(maze, path):
    '''
        Let maze be M and length of maze be L, and Height of maze M be H
        Let path be P, and length of Path be pathLen.
        Let i, j, k be integers such that 0 <= i < L and 0 <= j < H, and 0 <= k < pathLen
        A(i, j, k) = is a boolean indicating if a subpath starting at 0 to k (so the subpath is path[0:k]) 
        is contained within the maze up to the length bound i, and height bound j 
        (so contained within M[0:i][0:j]) and path[k] == M[i][j].

        Then:
        A(i, j, k) = A(i-1, j, k-1) if path[k] == M[i][j]
                     A(i, j-1, k-1) if path[k] == M[i][j] 
                     False if (A(i-1, j, k-1) || A(i, j-1, k-1)) and path[k] != M[i][j] 
                     True if path[pathLen-1] ==   
                     False if 



    '''

    
    leftToRightDistance = len(maze[0])
    topDownDistance = len(maze)

    A = [([] for i in range(leftToRightDistance)) for j in range(topDownDistance)]
    print(A)

    #for k in range(0, len(pathLen)):
    k = 0

    for i in range(0, len(leftToRightDistance))
        for j in range(0, len(topDownDistance)):
            kVal = None
            if(i == -1 and j == -1):
                kVal = 0
            else if(i == -1):
                kVal = 
            else if(j == -1)
            if(path[kVal+1] == maze[i][j]):
                A[i][j] = kVal + 1
            else:
                A[i][j] = -1 # -1 indicates path doesnt exist from this node


