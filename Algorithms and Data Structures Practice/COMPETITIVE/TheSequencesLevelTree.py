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

DO THIS PROBLEM!!!
