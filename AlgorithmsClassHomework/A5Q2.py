import numpy as np

from sys import stdin

def traverseBU():#  maze, path):
    path = ""

    maze = []
    for id, line in enumerate(stdin):
        if(id == 0):
            path = line.strip()
        elif(id == 1):
            continue # empty line
        elif line != '': # If empty string is read then stop the loop
            maze.append(list(line.strip()))
        else:    
            break

    leftToRightDistance = len(maze[0])
    topDownDistance = len(maze)
    pathLen = len(path)
    print("maze", maze)
    print("path", path)

    A = [ [[0 for k in range(pathLen)] for j in range(topDownDistance)] for i in range(leftToRightDistance) ]
    
    #  for i in range(leftToRightDistance):
    #      for j in range(topDownDistance):
    #          A[i].append(0)
    # print(A)
    for k in range(0, pathLen):
        for i in range(0, leftToRightDistance):
            for j in range(0, topDownDistance):
          
                
                
                print("k-1", k-1)
                print("i-1", i-1)
                print("j-1", j-1)
                print() 
                if( k == 0 and maze[i][j] == path[0]): # Base Case
                    A[i][j][0] = 1
                    continue
                elif(k == 0):
                    A[i][j][0] = 0
                    continue
                
                
                if(k - 1 >= 0 and i - 1 >= 0):
                    print("A(i-1, j)", i-1, j, A[i-1][j])
                    topHadPath = A[i-1][j][k-1]
                    # Try from left
                    if(topHadPath == 1 and maze[i][j] == path[k]):
                        A[i][j][k] = 1 #kValFromLeft + 1
                        print( "went right, i,j,k worked ", i, j, k)
                        
                        if(k + 1 == len(path)):
                            print("THE ARRAY", np.array(A))
                            return True
                        continue

                if(k - 1 >= 0 and j-1 >= 0):
                    leftHadPath = A[i][j-1][k-1];
                    if(leftHadPath == 1 and maze[i][j] == path[k]):
                        A[i][j][k] = 1 
                        print( "went down, i,j,k worked ", i, j, k)
                        if(k + 1  == len(path)):
                            print("THE ARRAY", np.array(A))
                            return True
                    
    print("THE ARRAY", A)
    return False

result = traverseBU()
print("1" if  result else "0")
