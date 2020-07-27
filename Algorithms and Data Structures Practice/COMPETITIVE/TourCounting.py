'''
Problem Statement
    	
You are given a directed graph g, and you must determine the number of 
distinct cycles in g that have length less than k. Since this number can be 
really big, return the answer modulo m. A cycle is a non-empty sequence of 
nodes (not necessarily distinct) in which there is an edge from each node to the 
next node, and an edge from the last node in the sequence to the first node. 

Two cycles are distinct if their sequences are not identical. 
See example 0 for further clarification.

g will be given as a String[] where the jth character of the ith 
element indicates whether there is an edge from node i to node j 
('Y' means there is one, and 'N' means there is not).

 
Definition
    	
Class:	TourCounting
Method:	countTours
Parameters:	String[], int, int
Returns:	int
Method signature:	int countTours(String[] g, int k, int m)
(be sure your method is public)
    
 
Notes
-	The answer modulo m means that you must return the remainder of dividing the result by m.
 
Constraints
-	g will have between 1 and 35 elements, inclusive.
-	Each element of g will have exactly N characters, where N is the number of elements in g.
-	Each character of each element of g will be 'Y' or 'N'.
-	The ith character of the ith element of g will be 'N'.
-	k will be between 1 and 1000000 (106), inclusive.
-	m will be between 1 and 1000000000 (109), inclusive.
 
Examples
0)	
    	
{"NYNY",
 "NNYN",
 "YNNN",
 "YNNN"}
6
100
Returns: 12
The possible cycles with length less than 6 are:

(0,3) ; (3,0) ; (0,1,2) ; (1,2,0) ; (2,0,1)

(0,3,0,3) ; (3,0,3,0) ; (0,1,2,0,3) ; (0,3,0,1,2)

(1,2,0,3,0) ; (2,0,3,0,1) ; (3,0,1,2,0)

Note that (0,3), (3,0) and (0,3,0,3) are all considered different.

1)	
    	
{"NYNNNNN",
 "NNYNNNN",
 "NNNYNNN",
 "NNNNYNN",
 "NNNNNYN",
 "NNNNNNY",
 "YNNNNNN"}
40
13
Returns: 9

All cycles have lengths that are multiples of 7. For each starting node and each 
multiple of 7 there exists one cycle. There are 5 non-zero multiples of 7 that 
are less than 40 (7,14,21,28,35) and 7 possible starting nodes. Therefore, the 
total number of cycles is 5x7=35. 35 modulo 13 is 9.
2)	
    	
{"NYNY",
 "NNNN",
 "YYNY",
 "NYNN"}
1000000
1000000000
Returns: 0
The graph does not have cycles.
3)	
    	
{"NY",
 "YN"}
1500
1
Returns: 0
Any number modulo 1 is zero.
4)	
    	
{"NYYNYYN",
 "YNYNYNY",
 "NYNYNYN",
 "YYYNYNY",
 "NNYYNNY",
 "NYYYNNY",
 "YYYYYYN"}
30
100
Returns: 72
5)	
    	
{"NYYY",
 "YNYY",
 "YYNY",
 "YNYN"}
1000
10000
Returns: 3564


This problem statement is the exclusive and proprietary property of TopCoder, Inc. 
Any unauthorized use or reproduction of this information without the prior written 
consent of TopCoder, Inc. is strictly prohibited. (c)2010, TopCoder, Inc. All rights reserved.


'''

'''

TourCounting

In this problem we are asked to find number of cycles in graph. 
I'll present a long way to solve this problem by DP to show how 
DP is accelerated by fast matrix exponentiation. All the cycles 
have a starting vertex. Let's handle only cycles that start from vertex 0, 
the case of all vertices can be solved by running the solution for each starting 
vertex separately. For the sake of simplicity we will count empty cycles too. 
Since there are exactly n such cycles, subtract n from the 
final answer to get rid of empty cycles.


The DP state domain is (i,v)->C where i is length of tour,
v is last vertex and C is number of ways to get from vertex 0 to 
vertex v in exactly i moves. The recurrent equation is the following: 
C(i+1,v) = sum_u(C(i,u) * A[u,v]) where A is adjacency matrix. 

The DP base is: C(0,*) = 0 except C(0,0) = 1. 

The answer is clearly sum_i(C(i,0)) for all i=0..k-1. 
This is DP solution with time complexity O(k*n). It is 
too much because we have to run this solution for 
each vertex separately.


Let's try to use binary matrix exponentiation to speed this DP up. 
The DP is layered, the matrix is just adjacency matrix, but answer 
depends not only on the last layer, but on all layers. 
Such a difficulty appears from time to time in problems 
with matrix exponentiation. The workaround is to add 
some variables to our vector. Here it is enough to add one 
variable which is answer of problem. Let R(i) = sum_j(C(j,0)) 
for all j=0..i-1. The problem answer is then R(k), 
so it no longer depends on intermediate layers results. 
Now we have to include it into DP recurrent equations: R(i+1) = R(i) + C(i,0).

To exploit vector-matrix operations we need to define vector of 
layer results and transition matrix. 

Vector is V(i) = [C(i,0), C(i,1), C(i,2), ..., C(n-1,0); R(i)]. 

Matrix TM is divided into four parts: upperleft (n x n) part is precisely the 
adjacency matrix A, upperright n-column is zero, bottomright element is 
equal to one, bottomleft n-row is filled with zeros except for 
first element which is equal to one. It is transition matrix because V(i+1) = V(i) * TM. 

Better check on the piece of paper that the result of vector-matrix multiplication 
precisely matches the recurrent equations of DP. The answer is last (n-th) element 
of vector V(k) = V(0) * TM^k. DP base vector V(0) is (1, 0, 0, ..., 0; 0). 

If power of matrix is calculated via fast exponentiation, the time complexity 
is O(n^3 * log(k)). Even if you run this DP solution for each vertex separately 
the solution will be fast enough to be accepted.

But there are redundant operations in such a solution. Notice that the core 
part of transition matrix is adjacency matrix. It remains the same for each of 
DP run. To eliminate this redundancy all the DP runs should be merged into one run. 

The slowest part of DP run is getting power of transition matrix. Let's merge all transition matrices.
The merged matrix is (2*n x 2*n) in size, upperleft (n x n) block is adjacency matrix, 
upperright block is filled with zeros, bottomright and bottomleft blocks are identity 
matrices. This matrix contains all previously used transition matrices as submatrices. 
Therefore the k-th power of this matrix also contains k-th powers of all used transition 
matrices TM. Now we can get answer of each DP by multiplying the vector corresponding to 
DP base and getting correct element of result. The time complexity for the whole problem is O(n^3 * log(k)).

struct Matrix {                                     //class of (2n x 2n) matrix
  int arr[MAXN*2][MAXN*2];
};
Matrix Multiply(const Matrix &a, const Matrix &b) { //get product of two matrices
  Matrix res;
  for (int i = 0; i<s; i++)
    for (int j = 0; j<s; j++) {
      int tres = 0;
      for (int u = 0; u<s; u++)
        add(tres, mult(a.arr[i][u], b.arr[u][j]));
      res.arr[i][j] = tres;
    }
  return res;
}
...
    Matrix matr, res;                               //matr is the constructed matrix
    n = g.size();                                   //res will be its power
    s = 2*n;                                        //s is size of matrix
    for (int i = 0; i<n; i++)
      for (int j = 0; j<n; j++) {                   //fill the matrix elements:
        matr.arr[i][j] = (g[i][j] == 'Y');          //upperleft = adjacency matrix
        matr.arr[i+n][j+n] = (i==j);                //bottomright = identity matrix
        matr.arr[i+n][j] = (i==j);                  //borromleft = identity matrix
      }
      
    for (int i = 0; i<s; i++)                       //set res matrix to identity
      for (int j = 0; j<s; j++)
        res.arr[i][j] = (i==j);
    for (int p = k; p>0; p>>=1) {                   //perform binary exponentiation
      if (p & 1) res = Multiply(res, matr);         //is current bit of power is set, multiply result by matr
      matr = Multiply(matr, matr);                  //matr represents (2^b)-th power of original matrix
    }
    
    int answer = ((-n)%m + m) % m;                  //subtract n empty cycles from problem answer
    for (int i = 0; i<n; i++)                       //sum results of individual DP runs
      add(answer, res.arr[n+i][i]);                 //get (n+i)-th element of V(0) * (matr ^ k)
    return answer;                                  //where V(0) has only i-th non-zero element set to 1

'''
