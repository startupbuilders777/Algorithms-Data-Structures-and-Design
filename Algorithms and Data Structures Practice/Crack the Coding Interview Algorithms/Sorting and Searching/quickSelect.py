import random

def quickSelect(arr, k):
    '''Given an array A, we want to find te kth smallest (or kth largest) element'''
    def swap(arr, i, j):
        temp = arr[i]
        arr[i] = arr[j]
        arr[j] = temp

    def quickSelect(arr, k, start, end): #end is one element off from the final element in the range
        if(start == end):
            return None
        randomIndex = random.randint(start, end-1) #<- THIS RANGE IS INCLUSIVE
        pivot = arr[randomIndex]
        pivotLocation = start
        swap(arr, randomIndex, pivotLocation)
       # print(randomIndex)
       # print(pivot)

        '''Sorting elements to left side automatically sorts the right side elements'''
        for i in range(0, end):
            if i < pivotLocation and arr[i] < pivot:
                continue
            if i > pivotLocation and arr[i] <= pivot:
                if(i != pivotLocation + 1): #Dont do a double swap back
                    swap(arr, pivotLocation, pivotLocation + 1) # Move up the pivot by one if you can and i is far away
                swap(arr, pivotLocation, i) #i not far away, dont have to move up by one
                pivotLocation += 1
        print("START: " + str(start))
        print("END: "+ str(end))
        print("PIVOT LOCATION" + str(pivotLocation))
        print("PIVOT: " + str(pivot))
        print("K: " + str(k))
        print(arr)

        if(k == pivotLocation):
            return pivot
        elif(k < pivotLocation):
            return quickSelect(arr, k, start, pivotLocation)
        elif(k > pivotLocation):
            return quickSelect(arr, k-pivotLocation-1, pivotLocation+1, end)

    return quickSelect(arr, k-1, 0, len(arr))


#arr = [1,4,2,7,6,32,3,4,5,6,3,1,2,4,6,7,3,21,3]
#print(arr)
#print(quickSelect(arr, 4))

#arr.sort()
#print(arr)

arr = [1,2,9,10,12,3,4,6,14,15,7]
print(arr)
print("THE QUICK SELECTED ELEMENT, BIGGER THAN 5", quickSelect(arr, 5))
arr.sort()
print(arr)
