"""

       0. 1. 2. 3. 4
  a = [4, 2, 1, 6, 3]
  S = {0, 1,  3, 4}
         ^

      [0,1] -> 1
      2 x 2 = 4


  a = [4, 2, 1, 6, 3]
  l = [4, 2, 1], r = [6, 3]
  

  0-1 => a[0:2] => [4, 2] => min([4, 2]) = 2
  0-3 => [4, 2, 1, 6] => 1
  1-3 => [2, 1, 6] => 1

  2 + 1 + 1 = 4 <- 

  [
    [4, 2, 1,1]
    [2, 1,1]
    [1,1]
    [6]
    [3]
  ]
   -> O(N^2  + S^2)
  S ~ N

  O(S^2*N)
  
  [
    []
    
  ]



"""
def soln(A, S):
  
  mins = [[] for i in range(len(A)) ]

  for idx, i in enumerate(A):


    for minArrIdx in range(idx+1):
      if len(mins[minArrIdx]) == 0:
        mins[minArrIdx].append(i)

      else:
        currentMinInArray = mins[minArrIdx][-1]
        mins[minArrIdx].append(  min(currentMinInArray, i) )

  print("mins arr", mins)

  sz = len(S)
  
  all_mins = []
  for i in range(sz):
    for j in range(i+1, sz):
      left = S[i]
      right = S[j]
      print("left, right-left", left, right-left)

      minVal = mins[left][right-left-1]
      print("LEFT, RIGHT, MIN VAL IS", left, right, minVal)

      all_mins.append(minVal)

  return sum(all_mins)


soln([4, 2, 1, 6, 3], [0, 1, 3, 4])