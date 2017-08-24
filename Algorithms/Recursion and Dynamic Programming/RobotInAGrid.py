arr = [[1, 0, 0, 1, 0, 1, 1, 1],
       [1, 1, 1, 1, 0, 1, 1, 1],
       [1, 1, 0, 1, 1, 1, 1, 1],
       [1, 1, 1, 0, 1, 1, 1, 1],
       [0, 1, 1, 0, 1, 1, 1 , 1],
       [0, 1, 1, 0, 1, 1, 1, 1],
       [0, 1, 1, 0, 1, 1, 1, 1],
       [0, 1, 1, 0, 1, 1, 1, 1],]
'''you cant step on the 0s, but you can on the 1s'''
'''Find path from top left to bottom right'''

'''Unoptimized'''
def travelNormal(grid = arr):
    path = []
    pathExists = [False]; # <- Have to do this BS because i dont know how to pass by reference in Python
    travelRecurNormal(arr=grid, path=path, col=0, row=0, pathExists=pathExists)
    print(pathExists[0])
    if pathExists:
        return path
    else:
        print("PATH DOES NOT EXIST")
        return []

def travelRecurNormal(arr, path, row, col, pathExists):
    if row == (len(arr) - 1) and col == (len(arr[0]) - 1):
        print("Ayy")
        pathExists[0] = True
        return
    if row + 1 < len(arr) and arr[row+1][col] == 1:
        print("(" + str(row+1) + " , " + str(col) + ")")
        path.append("D")
        travelRecurNormal(arr, path, row+1, col, pathExists)
        if(pathExists[0] == False):
            path.pop()
        else:
            return
    if col + 1 < len(arr[0]) and arr[row][col+1] == 1:
        print("(" + str(row) + " , " + str(col+1) + ")")
        path.append("R")
        travelRecurNormal(arr, path, row, col+1, pathExists)
        if(pathExists[0] == False):
            path.pop()
        else:
            return

print(travelNormal(arr))



'''Optimized'''
def travel(grid = arr):
    path = []
    memDict = {}
    pathExists = [False]; # <- Have to do this BS because i dont know how to pass by reference in Python
    travelRecur(memDict=memDict, arr=grid, path=path, col=0, row=0, pathExists=pathExists)
    print(pathExists[0])
    if pathExists:
        return path
    else:
        print("PATH DOES NOT EXIST")
        return []

def travelRecur(arr, path, row, col, memDict, pathExists):
    if row == (len(arr) - 1) and col == (len(arr[0]) - 1):
        print("Ayy")
        pathExists[0] = True
        return
    if memDict.get(str(row) + "," + str(col)) == True: # <- If it is true then it is a bad point and return
        return
    else:
        if row + 1 < len(arr) and arr[row+1][col] == 1:
            print("(" + str(row+1) + " , " + str(col) + ")")
            path.append("D")
            travelRecur(arr, path, row+1, col, memDict, pathExists)
            if(pathExists[0] == False):
                memDict[str(row+1) + "," + str(col)] = True
                path.pop()
            else:
                return
        if col + 1 < len(arr[0]) and arr[row][col+1] == 1:
            print("(" + str(row) + " , " + str(col+1) + ")")
            path.append("R")
            travelRecur(arr, path, row, col+1, memDict, pathExists)
            if(pathExists[0] == False):
                memDict[str(row) + "," + str(col+1)] = True
                path.pop()
            else:
                return

print(travel(arr))
