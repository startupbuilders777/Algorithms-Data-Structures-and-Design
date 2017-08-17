'''
Given 2 arrays of integers, compute the pair of values, one value in each array,
with the smallest(non-negative) difference. Return the difference
'''

arr1 = [1,3,15,11,2]
arr2 = [23,127,235,19,8]

def smallestDif(arr1, arr2):
    sortedArr1 = arr1[:] # <- Copy of arr
    sortedArr2 = arr2[:] # <- Copy of arr2
    sortedArr1.sort()
    sortedArr2.sort()


    print(sortedArr1)
    print(sortedArr2)



val = smallestDif(arr1, arr2)

print(val)