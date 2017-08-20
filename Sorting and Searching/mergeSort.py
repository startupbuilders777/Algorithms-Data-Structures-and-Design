'''


lets make this an inplace merge sort


'''

def mergeSort(arr):
    '''Arr end is the first indx not in the array'''
    return mergeSortRecur(arr, 0, len(arr))

'''
Sketch of algo
[1,2,3,4,5,6,7]

len = 7
7/2 = 3.5 -> 3
left1 ->0, right1 ->3
left2 ->3, right2 -> 7

'''


#merged parts are adjacent to each other?
def merge(arr, left1, right1, left2, right2):
    '''
    [->1,4,7,8,->2,3,5,6,7,9]
    merge (0,4) and (4,10)
    1 < 4 inc left
     [1,->4,7,8,->2,3,5,6,7,9]
    4 > 2 swap and inc left
    [1,->2,7,8,->4,3,5,6,7,9]
    [1,2,->7,8,->4,3,5,6,7,9]

    '''
    print("merging left1: " + str(left1) + " right1: " + str(right1) + " and " + "merging left2: " + str(left2) + " right2: " + str(right2))

    while True:
        if(left1 == right1):
            return arr
        elif(left2 == right2):
            return arr
        elif(arr[left1] < arr[left2]):
            left1 += 1
        elif(arr[left1] >= arr[left2]):
            while True:
                start = left1
                left2 += 1
                if(arr[left1] <= arr[left2]):
                    '''Reorganize the bounds'''













    return arr

arr = [1,4,7,8,2,3,5,6,7,9]
print(merge(arr, 0, 4, 4, 10))

def mergeSortRecur(arr, arrStart, arrEnd):
    print(arr)
    sizeA = arrEnd - arrStart
    if(sizeA == 1):
  #      print arr[arrStart]
        return arr[arrStart]

    half = int(sizeA/2)

    left1 = arrStart
    right1 = left1 + half #First element not sorted

    left2 = right1
    right2 = arrEnd

   #print("left1 " + str(left1))
   #print("right1 " + str(right1))
   #print("left2 " + str(left2))
   #print("right2 " + str(right2))
    mergeSortRecur(arr, left1, right1)
    mergeSortRecur(arr, left2, right2)

    return merge(arr,  left1, right1, left2, right2)

#arr = [7,4,3,1,7,5,4,2]
#print(mergeSort(arr))
