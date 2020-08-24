'''
Hacker Rank; '
https://www.hackerrank.com/challenges/3d-surface-area/problem


Say you break a cube in half and it leaves ridges on the surface where the break occurred,


OK SO GIVEN A GRID which represents height at each cell for a cube. 
calculate the surface area of this broken cube: 

Input:
1 3 4
2 2 3
1 2 4

SA:
60

TO SOLVE, REMEMBER THAT A 1X1 cube has 6 surface area. 
So recurse on that! 
And just use the entire cube tower and process each i,j and see how much surface area it adds.
Code below.

'''

# Complete the surfaceArea function below.
def surfaceArea(A):
    W = len(A)
    H = len(A[0])

    SA = 0 

    for i in range(W):
        for j in range(H):
            # go through each piece. 
            pieceHeight = A[i][j]
            #print(piece)
            SA += 2 # top and bottom. 
            # check 4 edging pieces now. 


            if(i + 1 < W):
                SA += max(0, pieceHeight - A[i+1][j])
            else:
                # edging piece add all of the pieceHeight
                SA += pieceHeight
            
            if(i -1 >= 0):
                SA += max(0, pieceHeight - A[i-1][j])
            else:
                SA += pieceHeight
            
            if(j + 1 < H):
                SA += max(0, pieceHeight - A[i][j+1])
            else:
                SA += pieceHeight
            
            if( j - 1 >= 0 ):
                SA += max(0, pieceHeight - A[i][j-1])
            else:
                SA  += pieceHeight
    
    return SA
