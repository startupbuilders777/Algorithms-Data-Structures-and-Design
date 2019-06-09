


arr = [1,2,1,1,1,1,2,1,1,2,1]

def calculateBalance(arr1, arr2):
    sum1 = 0
    sum2 = 0
    for i in arr1:
        sum1 += i
    for i in arr2:
        sum2 += 1
    if(sum1 == sum2):
        return 0
    if(sum1 > sum2):
        return "Left"
    if(sum < sum2):
        return "Right"

print(calculateBalance(arr[0:4], arr[4:8]))

def findRealLoonies(arr):
