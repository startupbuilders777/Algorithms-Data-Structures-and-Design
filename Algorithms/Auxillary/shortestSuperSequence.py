'''

Shortest Supersequence:
You are given 2 arrays, one shorter than the other (with all distinct elements) and one longer. Find the shortest
subarray in the longer array hat contains all the elements in the shorter array. The items can appear in any order.

{1,5,9} | {7, 5, 9, 0, 2, 1, 3, 5, 7, 9, 1, 1, 5, 8, 8, 9, 7}
output -> [7, 10]

'''

'''
Algo:

Look for 1:
{5, 10, 11}

look for 9:
{
1: 5,10,11
9: 2, 9, 15, 

9: 2 gets discarded

1: 5, 10, 11
9: 9, 15
You should check first: 5, 9
Then check 

Search for valid subranges and keep looking

}

'''
def verifyShort(inner, arr2, indexStart, indexEnd):
    ''' 
        Check if the inner array is within the specified indices
    '''
    indexCheck = 0
    for i in range(indexStart, indexEnd):
        if(arr2[i] == inner[indexCheck]):
            indexCheck += 1
    if(indexCheck == len(inner) - 1):
        return True
    else:
        return False

'''
Unit test verifyShort
'''

print(verifyShort([1,2,3], [3,4,6,7,4,1,2,3,4,6,7,8], 0, 5) == False)
print(verifyShort([1,2,3], [3,4,6,7,4,1,2,3,4,6,7,8], 3, 7) == True)

def shortestSeq(arr1, arr2):
    '''Also verify the smallest ranges and move outward if the shorter ranges dont contain all the elements'''
    if(len(arr1) == 0):
        return 0
    if(len(arr1) == 1):
        '''Check if it is in the seq and if it is, return 1, otherwise 0'''
        for i in arr2:
            if(i == arr1[0]):
                return 1
            else:
                return -1
    if(len(arr1) >= 2):
        left = arr1[0]
        right = arr1[len(arr1)-1]

