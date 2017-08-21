import random

'''Randomized Pivot finding Quicksort'''

def append(*args):
    arr = []
    newSize = 0

    for i in args:
        newSize += len(i)

    k = 0
    j = 0

    for i in range(0, newSize):
        if(j < len(args[k])):
            arr.append(args[k][j])
            j += 1
        else:
            k += 1
            j = 0
            if(j < len(args[k])):
                arr.append(args[k][j])
                j += 1
    return arr

def quicksort(arr):
    if(len(arr) == 0):
        return []
    elif(len(arr) == 1):
        return [arr[0]]

    pivotIndex = random.randint(0, len(arr)-1)
    pivot = arr[pivotIndex]
    left = []
    right = []
    print(pivot)
    saw_pivot = False
    for i in range(0, len(arr)):
        if(arr[i] < pivot):
            left.append(arr[i])
        elif(arr[i] == pivot and saw_pivot == False):
            saw_pivot = True
        elif(arr[i] >= pivot):
            right.append(arr[i])
   # print("pivot: " + str(pivot))
   # print("left: " + str(left))
   # print("right: " + str(right))
    return append(quicksort(left), [pivot], quicksort(right))

arr = [1,5,3,2,1,4,5,7,4,8,9,4,2,35,2,1,12,4,5]
arr2 = [9,3,5,3,2,1]

sorted = quicksort(arr)
sorted2 = quicksort(arr2)

print(arr)
print(sorted)
print(arr2)
print(sorted2)

#print(append([1.3,5], [], [0,1,2,3], arr, [], [0,1,2,3], [2], [1,2]))
#print( for i in range(0,4))

#[1, 5, 3, 2, 1, 4, 5, 7, 4, 8, 9, 4, 2, 35, 2, 1, 12, 4, 5, 1, 5, 3, 2, 1, 4, 5, 7, 4, 8, 9, 4, 2, 35, 2, 1, 12, 4, 5, 0, 1, 2, 3, 1, 5, 3, 2, 1, 4, 5, 7, 4, 8, 9, 4, 2, 35, 2, 1, 12, 4, 5, 1, 5, 3, 2, 1, 4, 5, 7, 4, 8, 9, 4, 2, 35, 2, 1, 12, 4, 5, 1, 5, 3, 2, 1, 4, 5, 7, 4, 8, 9, 4, 2, 35, 2, 1, 12, 4, 5, 0, 1, 2, 3, 1, 5, 3, 2, 1, 4, 5, 7, 4, 8, 9, 4, 2, 35, 2, 1, 12, 4]
