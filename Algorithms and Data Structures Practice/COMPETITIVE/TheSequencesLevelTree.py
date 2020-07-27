'''
Problem Statement
    	
When John and Brus were high school students, they liked to investigate integer sequences. 
Once, John wrote down a sequence containing distinct positive integers. Brus wanted to reorder 
the elements to get a "mountain sequence". A sequence a0, a1, ..., an-1 is called a mountain 
sequence if there exists an index j, where 0 < j < n-1, such that the sequence a0, a1, ..., aj 
is strictly increasing, and the sequence aj, aj+1, ..., an-1 is strictly decreasing. A sequence 
is strictly increasing if each element is strictly greater than the element before it, and a sequence 
is strictly decreasing if each element is strictly less than the element before it.

Brus also wanted the resulting sequence to satisfy one additional rule. The absolute difference 
between each pair of adjacent elements must be less than or equal to k.

You are given a int[] sequence containing John's original sequence. Return the number of possible valid 
mountain sequences Brus could construct modulo 1234567891. If no valid sequences can be constructed, return 0.

 
Definition
    	
Class:	TheSequencesLevelThree
Method:	find
Parameters:	int[], int
Returns:	int
Method signature:	int find(int[] sequence, int k)
(be sure your method is public)
 
Constraints
-	sequence will contain between 1 and 50 elements, inclusive.
-	Each element of sequence will be between 1 and 1,000,000,000, inclusive.
-	All elements in sequence will be distinct.
-	k will be between 1 and 1,000,000,000, inclusive.
 
Examples
0)	
    	
{1, 5, 10, 4}
10
Returns: 6
There are six ways for Brus to get the 
"mountain sequence" - {1, 4, 10, 5}, {1, 5, 10, 4}, {1, 10, 5, 4}, {4, 5, 10, 1}, {4, 10, 5, 1}, {5, 10, 4, 1}.
1)	
    	
{1, 5, 10, 4}
6
Returns: 4
Because of the additional rule where adjacent elements cannot differ by more than k=6, 
the following sequences are not valid: {1, 10, 5, 4} and {4, 5, 10, 1}.
2)	
    	
{4, 44, 7, 77}
1
Returns: 0
No possible ways.
3)	
    	
{96, 29, 21, 90, 46, 77, 31, 63, 79}
44
Returns: 126

'''

'''

Take sequence

Sort it, 
Pick a pivot point for the mountain. 
Isnt it the largest value Yup. 

then generate sequence for right side and left side. 

Ok process sorted array largest to smallest. 

when there is 3 elements -> only 1 possible solution, 

in other words, just choose elements indiscriminately and put on left and 
right side. 

can also just make 2 increasing subsequences, how many ways to do that!
    -> TRUE

Brute force un-optimized: 
SORT, get max element, 
then go through rest of elements in unsorted manner; 
process each element, put on either right or left side.
at the end, both subsequences in recursion have to have all elements included and work
and both have to have length > 1

Better -> preprocess -> create graph that tells u all indexes smaller than a certain element. 
Start at max val, create left and right subsequence, starting only with kids of max_val which is graph[max_val]
From there, put elements in left and right subseq, 
process sorted list from greatest to least, and use graph to help you count faster. 
when you add a new element to process in the DP, try all possible ways, 
'''
'''
SOLUTION

TheSequencesLevelThree

The key idea for this problem is to sort all elements and then construct possible 
sequences by adding elements one-by-one. Since we add elements in increasing order, 
each new element can be pushed to the left or to the right. So each partial solution 
has x left-most elements and y right-most elements already set, the remaining middle 
part of the sequence is not determined yet. When we add a new element either x or y is 
increased by one. Since there is additional constraint on the difference between neighbouring 
elements, we have to memorize the border elements. The values of these elements may be very large, 
so it is better to store their indices: L is the index of last pushed element on the left 
(x-th element from the left) and R is the index of last pushed element to the right (y-th element from the right). 
Look code example for the picture.

Ok, now we have defined state domain (k,x,y,L,R)->C where k is number of used elements, 
x is number of elements to the left, y is number of elements to the right, L is index of the left border element, 
R is index of the right border element, C is number of such partial solutions. 
Note that L and R are 1-indexed for the reason described further in the text. 
There are two conditional transitions: to (k+1,x+1,y,k+1,R) and to (k+1,x,y+1,L,k+1). 
We see that DP is layered by parameter k, so we can iterate through all DP states 
in order of increasing k. To get the problem answer we have to consider all 
states with one undefined element (i.e. k = N-1) and try to put maximal element to the middle. 

Also the statement says that there must be at least one element to the left and 
one to the right of the "top" number, so only states with x>=1 and y>=1 
are considered for the answer.

It is obvious that this state domain contains a lot of impossible states because 
any valid state has x + y = k. From this equation we express y = k — x explicitly. 
Now we can throw away all states with other y values. The new state domain is 
(k,x,L,R)->C. It is great that the DP is still layered by parameter k, so 
the order of state domain traversal remains simple. Note that if we exploited k = x + y 
instead and used state domain (x,y,L,R)->C we would have to iterate 
through all states in order of increasing sum x + y.

Ok, but do we really need the x and y parameters?! How does the problem answer 
depend on them? The only thing that depends on x and y is getting the 
problem answer. Not exact x and y values are required but only whether 
they are positive or zero. The states with x=2 and x=5 are thus equivalent, 
though x=0 and x=1 are not. The information about x and y parameters is almost 
unnecessary. To deal with x>=1 and y>=1 conditions we introduce "null" element in sequence. 
If parameter L is equal to zero then there is no element on the left side (as if x = 0). 
If L is positive then it is index of sequence element which is on the left border 
(and of course x>0). Right side is treated the same way. Now information about 
x parameter is not necessary at all. Let's merge equivalent states by deleting parameter 
x from the state domain. The state domain is now (k,L,R)->C.

But there is still room for improvement. Notice that for any valid state max(L,R) = k. 
That's because k-th element is the last added element: it must be the left border 
element or the right border element. So states with max(L,R) != k are all 
impossible (give zero results). We can exploit this equation. 

Eliminating parameter k from state domain is not convenient because 
then we would have to iterate through states (L,R) in order of increasing 
max(L,R). So we would replace parameters L,R to parameters d,m. These 
two parameters do not have much sense — they are used only to encode 
valid L,R pairs: m==false: L = d, R = k; m==true : L = k, R = d; 

The final state domain is (k,d,m), where k is number of set elements, 
d is index of element other than k, m means which border element is k. 
Since DP is layered, we can use "store two layers" space optimization. 

The final time complexity is O(N^2) and space complexity is O(N). 
Note that such a deep optimization is overkill 
because N<=50 in the problem statement. 

You could stop at O(N^4) =)


// 
// [a(1), a(2),  ..., a(x-1), arr[L], ?, ?, ..., ?, arr[R], b(y-1), ..., b(2), b(1)]
//  \____________known_____________/     unknown    \____________known____________/
//            x elements                                       y elements
//                          already set elements: k = x + y
 
int n;                                                //k  -  number of elements already set
int64 res[2][MAXN][2];                                //d: arr[d] is last element on one of borders
...                                                   //m=0 means arr[d] is last on the left, m=1  means  on the right
    n = arr.size();                                   //note that actual elements are enumerated from 1
    arr.push_back(-1);                                //index d=0 means there is no element on the border yet
    sort(arr.begin(), arr.end());
    res[0][0][0] = 1;                                 //DP base: no elements, no borders = 1 variant
    int64 answer = 0;
    if (n < 3) return 0;                              //(better to handle this case explicitly)
    for (int k = 0; k<n; k++) {                       //iterate through all states
      memset(res[(k+1)&1], 0, sizeof(res[0]));        //(do not forget to clear the next layer)
      for (int d = 0; d<=k; d++)                      //(d cannot be greater than k)
        for (int m = 0; m<2; m++) {
          int64 tres = res[k&1][d][m];
          if (tres == 0) continue;                    //discard null states
          int L = (m==0 ? d : k);                     //restore L,R parameters from d,m
          int R = (m==0 ? k : d);
          int nelem = arr[k+1];                       //we'll try to add this element
            
          if (L==0 || abs(arr[L]-nelem)<=maxd)        //trying to add element to the left border
            add(res[(k+1)&1][R][0], tres);                
            
          if (R==0 || abs(arr[R]-nelem)<=maxd)        //trying to add element to the right border
            add(res[(k+1)&1][L][1], tres);
            
          if (k == n-1)                               //trying to add highest(last) element to the middle
            if (L>0 && abs(arr[L]-nelem)<=maxd)       //ensure that there is a good element to the left
              if (R>0 && abs(arr[R]-nelem)<=maxd)     //ensure that there is a good element to the right
                add(answer, tres);                    //adding nelem to the middle produces solutions
        }
    }
    return int(answer);
  }

'''
