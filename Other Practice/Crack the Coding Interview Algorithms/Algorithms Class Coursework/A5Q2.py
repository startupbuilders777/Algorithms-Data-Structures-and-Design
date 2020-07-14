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

    A = [ [[ (0, (-1,-1)) for k in range(pathLen)] for j in range(topDownDistance)] for i in range(leftToRightDistance) ]

    for k in range(0, pathLen):
        for i in range(0, leftToRightDistance):
            for j in range(0, topDownDistance):

                if( k == 0 and maze[i][j] == path[0]): # Base Case
                    A[i][j][0] = (1, (i,j)) #pass along start i, j 
                    continue
                elif(k == 0):
                    A[i][j][0] = (0, (-1,-1))
                    continue
                
                
                if(k - 1 >= 0 and i - 1 >= 0):
                    topHadPath, startIandJ = A[i-1][j][k-1]
                    if(topHadPath == 1 and maze[i][j] == path[k]):
                        A[i][j][k] = (1, startIandJ) 
                        if(k + 1 == len(path)):
                            print(str(1 + startIandJ[0]) + " " + str(1 + startIandJ[1]) + " " + str(1 + i) + " " + str(1 + j)) 
                            return True
                        continue
                if(k - 1 >= 0 and j-1 >= 0):
                    leftHadPath, startIandJ = A[i][j-1][k-1];
                    if(leftHadPath == 1 and maze[i][j] == path[k]):
                        A[i][j][k] = (1, startIandJ) 
                        if(k + 1  == len(path)):
                            print(str(1 + startIandJ[0]) + " " + str(1 +  startIandJ[1]) + " " + str(1 + i) + " " + str(1 + j)) 
                            return True
                    
    print("0")

traverseBU()
