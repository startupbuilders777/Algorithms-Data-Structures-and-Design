'''
A magic index in an array A[0...n-1] is defined to be an index such that A[i] = i. Given a sorted array of distinct
integers, write a method to find a magic index if one exists, in array A

Follow up
What if the values are not distinct
'''

arr1 = [-6,-3,-2,1,4,12, 34]

arr2 = [-6, -6, 1, 2, 2, 3]

arr3 = [-6, -6, 1, 1, 1, 1, 6, 6, 6]

'''O(n) solution trivial. Attempt Divide and Conquer Soln'''
def magicIndex(arr):
    return magicIndexRecur(arr, 0)

'''mid + offset is the real index in the original array'''
def magicIndexRecur(arr, offset):
    print(arr)
    size = len(arr)
    if(size == 0):
        return -1
    mid = int(size/2)
    print("mid: " + str(mid))
    print("offset: " + str(offset))
    print("real index is " + str(mid+offset))
    print("arr[mid]: " + str(arr[mid]))
    if(arr[mid] == mid + offset):
        return mid + offset #<- Return the real index
    elif(arr[mid] > mid + offset):
        print("Get left")
        return magicIndexRecur(arr[:(mid)], offset) #<- dont subtract 1 because it goes from first element to last element that isnt included
    else:
        print("Get Right")
        return magicIndexRecur(arr[(mid+1):], mid+1)

print(arr1)
print(magicIndex(arr1))

#Duplicates dont work on this algo since duplicates dont work on binary search either
#Algorithm failed when elements are not distinct. Gutta search left and right now.

#RIGHT ALGO FOR THIS CASE LATER
####################################################################################################################



