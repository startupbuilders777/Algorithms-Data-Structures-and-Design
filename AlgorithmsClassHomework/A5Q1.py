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

    A = [[] for i in range(leftToRightDistance) ]

    for i in range(leftToRightDistance):
        for j in range(topDownDistance):
            A[i].append(0)
    
    for i in range(0, leftToRightDistance):
        for j in range(0, topDownDistance):
            if(i == 0 and j == 0 and maze[i][j] == path[0]): # Base Case
                A[0][0] = 0
                continue
            elif(i == 0 and j == 0):
           
                return False

            kValFromLeft = 0
            kValFromTop = 0
            if(i - 1 >= 0 and A[i-1][j] != -1):
                kValFromLeft = A[i-1][j]
                # Try from left
                if(maze[i][j] == path[kValFromLeft+1]):
                    A[i][j] = kValFromLeft + 1
                    if(A[i][j] + 1 == len(path)):
                        return True
                    continue

            if(j-1 >= 0 and A[i][j-1] != -1 ):
                kValFromTop = A[i][j-1];
                if(maze[i][j] == path[kValFromTop+1]) :
                    A[i][j] = kValFromTop + 1
                    if(A[i][j] + 1  == len(path)):
                        return True
                    continue

            A[i][j] = -1
    return False

result = traverseBU()
print("1" if  result else "0")
