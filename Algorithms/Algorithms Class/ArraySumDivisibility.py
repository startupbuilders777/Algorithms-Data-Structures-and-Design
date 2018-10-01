'''
  Consider the following problem: on input A[1..n], an array of integers with n >= 2
   elements, decide whether there exist two indices i, j with 1 <= i < j <= n such that
   the arithmetic mean of the elements A[i..j] is an integer. For example, your algorithm
   should return no on the input A = [4, 3, 6, 1] and yes on the input B = [1, 3, 4].
    a [7 marks] Give an efficient algorithm to solve this problem. Your answer should
        include (a) the pseudocode; (b) a brief explanation in English of how it works;
        (c) a brief justification of correctness.
    b [3 marks] Analyze your algorithm in the unit-cost model. Express the running
        time, using asymptotic notation, as a function of n, the number of inputs. You
        can assume that basic arithmetic operations on integers, such as integer division
        and remainder (mod), can be done in unit time. Show your work.

'''



'''

'''

def divisible_integers(arr):
    cumulative_sum = []
    sum = 0
    cumulative_sum.append(sum)
    for i in range(len(arr)):
        sum += arr[i]
        cumulative_sum.append(sum)
        
    
    # print cumulative_sum
    a = 0 

    input_arr_length = len(arr)
    
    while a < input_arr_length:
        b = a + 1
        while b < input_arr_length:
            range_sum = cumulative_sum[b+1] - cumulative_sum[a]
            divisor = b - a + 1
            # print range_sum
            if divisor != 1 and (range_sum % divisor) == 0:
                return True
            
            b += 1
        
        a += 1

    
    return False

'''

'''
A = [4,3,6,1] # False 
B = [1,3,4] # True
C = [4,3,1] # True
D = [5,2,3,4,12] # True
E = [2,5,7,9,10] # True [5,7,9]
F = [2,9,8,11,6] # False

print("divisible integers: ", divisible_integers([4,3,6,1]))
print("divisible integers: ", divisible_integers([1,3,4]))

print("A", divisible_integers(A))

print("B", divisible_integers(B))
print("C", divisible_integers(C))
print("D", divisible_integers(D))
print("E", divisible_integers(E))
print("F", divisible_integers(F))



'''
------------------
Explanation:

The algorithm utilizes a cumulative sum array.  
The first element in the cumulative sum array is 0.
Index k in the cumulative sum array is the sum of all elements up to but not including k 
in the input array. 
The cumulative sum array is longer than the input array by 1 because it includes 
0 as the pushed in as the first element.   

For instance => input array [1,3,4] yields a cumulative sum array which is [0, 1, 4, 8]

To find the sum between 2 indices i and j inclusive in the input array, 
the calculation cumulative_sum[j+1] - cumulative_sum[i] is performed. The number of elements 
between i and j is j - i + 1 which can be rearranged as (j+1) - i. 

To determine the solution to the questions, all possible ranges for the sums are checked. 
To check, the modulo between the sum of a range and its length is calculated and if its 0, then true is 
returned, otherwise, other sum ranges are checked. If all sum ranges are checked and none are divisible,
then false is returned. 

----------------------------
Proof of Correctness

Use loop variables


-----------------------------

Running Time


(b) 
Assume the length of the input is N. 
The running time of the algorithm is O(N^2)

The cumulative sum array is computed with O(N) operations by iterating through the input array and calculating the sums. 

The checking of all possible sums is done in 2 for loops, one nested in another, with 2 iterator variables. 
The outer iterator, a, iterates from 0 to N.
The inner iterator, b,  iterates from a + 1 to N, in each iteration of a
The run time of the two loops is Theta(N^2) 


The final runtime is THETA(N) + THETA(N^2) which simplifies to THETA(N^2)

'''


'''
3 BONUS:

[Bonus question, 5 marks extra credit only] Characterize the integers n >= 2 for which
   there is at least one permutation of 1, 2, . . . , n such that the algorithm in the previous
   problem answers no. To get marks you need to prove your answer for all n.
                         

integers n >= for which there is at least one permutation of 1, 2, ... n such that the algo in the previous problem answers no :
n = 3 has no permutations where the Q2 algorithm returns "no", and is thus not in the set of n-values that Q3 is asking you to describe.
[1, 2, 3]
[1, 3, 2]
[2, 1, 3]
[2, 3, 1]
[3, 1, 2]
[3, 2, 1]

n = 4


1, 4, 3, 2 => Returns Yes
1, 4, 3, 2 => Returns Yes

n = 4 => 

n = 5

do not classify for n where sum of all element is divisible by length so 
n = 5 => sum is (5 + 1)(5) / 2 = 15 % 5 == 0

1, 2, 5, 3, 4


n = 6 => (6+1)(6) / 2 = (21 % 6 => gucci


n = 7 =>  (7+1)*7 / 2 = 28% 7 => NOT in set 
1 + 2 + 3 + 4 + 5 + 6 + 7 = 28

n = 8 = > (8+1) * 8 / 2 = 36 % 8 = > not divisibleself SO IN THE SET.

ok so
if n is odd, then it has 


The assignment says "To get marks you need to prove your answer for all n".    That means you should prove
 
"If (blah) holds then n has the desired property" 
AND
"If (blah) doesn't hold then n doesn't have the desired property."


Through observation, for integers N >= 2, 
if N is even then there is atleast one permutation of 1,2,.., n such that the algorithm
answers "no". 

if N is odd then there is no permutations of 1,2,.., n such that the algorithm answers "no"

For the N is odd case: 


For N is even: 

we can construct the array that results in the answer being no. To prove 
this, we will perform induction on even values of N >= 2

Base Case:
When N == 2, the array is [2, 1] which returns No because 2+1 == 3 is not divisible by 2


Inductive Hypothesis: 
For Even N, there exists a permutation called X which is [a,b,c...x] 
that results in the No answer


Inductive Step: 
Prove for N+2 there exists a permutation

So the permutation that results in the No answer for array size N is X
To construct an array size N + 2, append [N+2, N+1] at the end of X 
resulting in the arrray [a,b,c..x, N+2, N+1]


We have to check all ranges that include N+2
Call this range that includes N+2, F, and F, which contains A, and N+2 ([..A, N+2]), 
where A is the other elements in the range, and a_len is the length of A. 
A is not divisible by a_len

A % a_len = r1, where r1 > 0 

A = a_len*divisor + r1



A + N+2 = (a_len+1)*divisor2 + r2
A + N+2 = a_len*divisor2 + divisor2 + r2
alen*divisor + r1 + N+2 = a_len* divisor2 + divisor2  + r2


A + N + 2 is not divisible by a_len + 1, because A is not  N+2 is even and a_len+1 is odd. 

And we have to check all ranges that include N+2 and N+1
These ranges have added onto them 2N+3 (N+2 + N+1)

range + 2N+3 % range_len + 2 !== 0






QED









'''
