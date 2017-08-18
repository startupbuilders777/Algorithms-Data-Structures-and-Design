'''
Given 2 arrays of integers, compute the pair of values, one value in each array,
with the smallest(non-negative) difference. Return the difference
'''

arr1 = [1,3,15,11,2]
arr2 = [23,127,235,19,8]

arr3 = [11,1,5,6,1,3,4,7,9,5,2]
arr4 = [6,3,7,3,8,9,12,4,67,5]

'''
The smallest diff is between adjacent numbers in the merged list of the 2 arrays
[1,2,3,8,11,15,19,23,127,235]
[A,A,A,B,A,A,B,B,B,B]

Algo performance O(nlogn)

The simpler algo which was not implemented here was moving up the 
ptr for the array with the smaller array value by 1. Therefore
getting closer to minimizing the difference.
'''

def smallestDif(arr1, arr2):
    a1 = arr1[:] # <- Copy of arr
    a2 = arr2[:] # <- Copy of arr2
    a1.sort()
    a2.sort()
    min = abs(a1[0] - a2[0])
    val1, val2, index1, index2 = 0, 0, 0, 0
    lastVal = "start"
    for i in range(0, len(a1) + len(a2)):
        print min
        if(index1 == len(a1) or index2 == len(a2)):
            print("Exhausted a list")
            return min
        elif(a1[index1] < a2[index2]):
            print("a1 < a2")
            print("last val: " + lastVal)
            val1 = a1[index1]
            index1 += 1
            if lastVal == "a2" and val1-val2 < min: #<- val2 is last value of arr2
                min = val1 - val2
            lastVal = "a1"
        elif(a1[index1] >= a2[index2]):
            print("a2 < a1")
            print("last val: " + lastVal)
            val2 = a2[index2]
            index2 += 1
            if lastVal == "a1" and val2-val1 < min: #<- val1 is last value of arr1
                min = val2 - val1
            lastVal = "a2"
        print(a1)
        print(a2)
    return min



#val = smallestDif(arr1, arr2)
#print(val)
val = smallestDif(arr3, arr4)
print(val)