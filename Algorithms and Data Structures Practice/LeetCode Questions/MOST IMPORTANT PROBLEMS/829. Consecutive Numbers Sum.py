'''
829. Consecutive Numbers Sum
Hard

348

471

Add to List

Share
Given a positive integer N, how many ways can we write it as a sum of consecutive positive integers?

Example 1:

Input: 5
Output: 2
Explanation: 5 = 5 = 2 + 3
Example 2:

Input: 9
Output: 3
Explanation: 9 = 9 = 4 + 5 = 2 + 3 + 4
Example 3:

Input: 15
Output: 4
Explanation: 15 = 15 = 8 + 7 = 4 + 5 + 6 = 1 + 2 + 3 + 4 + 5
Note: 1 <= N <= 10 ^ 9.

'''


'''
THIS IS THE MATH SOLUTION:

we can decompose a number in the largest consecutive sum to 
smaller consecutive sum and count all the ways. 

Sum up to a number -> n*(n+1)/2

Ok formula for this question:
n*(n+1)/2   -  (k-1)*(k+1-1)/2 

-> Simplify and factorize
(n+k)(n-k+1) = 2*Sum

So find all factors of 2*Sum and that is the answer. 
well you actually still have to check that n-k+1 >= 0
to mjake sure we respect the sum of "positive" integers

For instance -> 15 has following factors for twice its sum -> 30: [(10,3), (15, 2), (5,6), (1,30)]
so total ways can be 4 ways but we have to make sure k >= 0

if a and b are the factors then: [ a is bigger value, b is smaller value'
since a = n + k, and b = n  - k -1
]

k = (b-a-1)/(-2)
n =  (a+b-1)/2

so for 10,3

->k: 3-10-1/-2 => 4 
->n: 10+3-1/2 => 6

So it is 4 + 5 + 6 or  sum(to 6) - sum(to 3) = 15 


Should we also consider when both a and b are negative instead of positive because 
it still satsifies the (n+k)(n-k+1) = 2*Sum ?
'''

# How do find factors? 
'''
just check all numbers up to sqrt(2*N)

or check only prime factors -> and compose the other factors? 
with sieve of erthanosis
'''
        
import math
class Solution:
    
    def findFactors(self, N):
        
        upper = int(math.sqrt(N))
        
        factors = []
        i = 1
        while i <= upper:
            # print("SAW I", i)
            
            if N % i == 0:
                factors.append( (i, N/i) )
            i += 1
        return factors
        
        
    def consecutiveNumbersSum(self, N: int) -> int:        
        facts = self.findFactors(2*N)
        # now test each factor to make sure they are both positive: 
        
        def sumUpTo(a):
            return (a)*(a+1)/2
           
        count = 0
        for b, a in facts:
            k = (b-a-1)/(-2)
            n = (a + b - 1) / 2
            # print("N, K-1 IS", (n, k-1, sumUpTo(n) - sumUpTo(k-1)))
            
            if k - 1 < 0:
                continue
                
            # we also have to ensure that they are both whole numbers!
            if k % 1 != 0:
                continue
            
            count += 1
        
        return count


# An explained different solution:
'''
The thought process goes like this- Given a number N, we can possibly write it as a sum of 
2 numbers, 3 numbers, 4 numbers and so on. Let's assume the fist number in this series be x. Hence, we should have

x + (x+1) + (x+2)+...+ k terms = N
kx + k*(k-1)/2 = N implies
kx = N - k*(k-1)/2

So, we can calculate the RHS for every value of k and if it is a multiple of k then we can construct a sum of N using k terms starting from x.
Now, the question arises, till what value of k should we loop for? That's easy. In the worst case, RHS should be greater than 0. That is

N - k*(k-1)/2 > 0 which implies
k*(k-1) < 2N which can be approximated to
k*k < 2N ==> k < sqrt(2N)
Hence the overall complexity of the algorithm is O(sqrt(N))

PS: OJ expects the answer to be 1 greater than the number of possible ways 
because it considers N as being written as N itself. 
It's confusing since they ask for sum of consecutive 
integers which implies atleast 2 numbers. But 
to please OJ, we should start count from 1.

'''

'''
class Solution {
public:
    int consecutiveNumbersSum(int N) {
        int count = 1;
        for( int k = 2; k < sqrt( 2 * N ); k++ ) {
            if ( ( N - ( k * ( k - 1 )/2) ) % k == 0) count++;
        }
        return count;
    }
};
'''

'''
N can be expressed as k, k + 1, k + 2, ..., k + (i - 1), where k is a positive integer; therefore

N = k * i + (i - 1) * i / 2 => N - (i - 1) * i / 2 = k * i, which implies that as long as 
N - (i - 1) * i / 2 is k times of i, we get a solution corresponding 
to i; Hence iteration of all possible values of i, starting from 1, will cover all cases of the problem.
'''
def consecutiveNumbersSum(self, N: int) -> int:
    i, ans = 1, 0
    while N > i * (i - 1) // 2:
        if (N - i * (i - 1) // 2) % i == 0:
            ans += 1
        i += 1
    return ans

# Faster soln similar to yours:

class Solution:
    def consecutiveNumbersSum(self, N: int) -> int:
        pairs = []
        base = 2*N
        sr = int((base)**0.5)
        for i in range(1,sr+1):
            if base%i == 0:
                pairs.append((i,base//i))
        res = 0
        for pair in pairs:
            if (pair[1] - 1 - pair[0])%2 == 0:
                res += 1
        return res


# FASTEST: 

class Solution:
    def consecutiveNumbersSum(self, N: int) -> int:
        n = N
        res = 1
        i = 3
        while n % 2 == 0:
            n /= 2
        while i*i <= n:
            count = 0
            while n % i == 0:
                n /= i
                count +=1
            res*= count + 1
            i +=2
        return res if n == 1 else res * 2


'''
LEES COUNT ODD FACTORS SOLN:

    Basic Math
    N = (x+1) + (x+2) + ... + (x+k)
    N = kx + k(k+1)/2
    N * 2 = k(2x + k + 1),where x >= 0, k >= 1

    Either k or 2x + k + 1 is odd.

    It means that, for each odd factor of N,
    we can find a consecutive numbers solution.

    Now this problem is only about counting odd numbers!
    I believe you can find many solutions to do this.
    Like O(sqrt(N)) solution used in all other post.

    Here I shared my official solution.


    Some Math about Counting Factors
    If N = 3^a * 5^b * 7*c * 11*d ...., the number of factors that N has equals
    N_factors = (a + 1) * (b + 1) * (c + 1) * (d + 1) .....


    Explanantion:
    Discard all factors 2 from N.
    Check all odd number i if N is divided by i.
    Calculate the count of factors i that N has.
    Update res *= count.
    If N==1, we have found all primes and just return res.
    Otherwise, N will be equal to P and we should do res += count + 1 where count = 1.

    Complexity:
    To be more accurate, it's O(biggest prime factor).
    Because every time I find a odd factor, we do N /= i.
    This help reduce N faster.

    Assume P is the biggest prime factor of a odd number N .
    If N = 3^x * 5^y ...* P, Loop on i will stop at P^0.5
    If N = 3^x * 5^y ...* P^z, Loop on i will stop at P.
    In the best case, N = 3^x and our solution is O(log3N)
    In the worst case, N = P^2 and our solution is O(P) = O(N^0.5)

    Though in my solution, we didn't cache our process of finding odd factor.
    Moreover, if we prepare all prime between [3，10^4.5].
    it will be super faster because there are only 3400 primes in this range.
    This complexity will be O(P/logP) with P < sqrt(N)


    Java

        public int consecutiveNumbersSum(int N) {
            int res = 1, count;
            while (N % 2 == 0) N /= 2;
            for (int i = 3; i * i <= N; i += 2) {
                count = 0;
                while (N % i == 0) {
                    N /= i;
                    count++;
                }
                res *= count + 1;
            }
            return N == 1 ? res : res * 2;
        }
    Short C++/Java:

            int res = 1, count;
            while (N % 2 == 0) N /= 2;
            for (int i = 3; i * i <= N; res *= count + 1, i += 2)
                for (count = 0; N % i == 0; N /= i, count++);
            return N == 1 ? res : res * 2;
    Python

        def consecutiveNumbersSum(self, N):
            res = 1
            i = 3
            while N % 2 == 0:
                N /= 2
            while i * i <= N:
                count = 0
                while N % i == 0:
                    N /= i
                    count += 1
                res *= count + 1
                i += 2
            return res if N == 1 else res * 2

'''

'''
Another One:

With n consecutive integers, the first number we can form is 1 + 2 + ... + n. 
The next number is 2 + 3 + ... + n + n + 1, or 1 + 2 + ... + n + n, and then 1 + 2 + ... + n + n + n, and so on.

Therefore, a number N can be formed by n consecutive integers, if N - (1 + 2 + ... n) modulo n is zero. 

The code below just increases n and tests if N can be formed by n numbers.
Note that a sum of arithmetic progression from 1 to n can be calculated as n * (n + 1) / 2.

    int consecutiveNumbersSum(int N, int res = 0) {
        for (auto n = 2; n * (n + 1) / 2 <= N; ++n) res += (N - n * (n + 1) / 2) % n == 0 ? 1 : 0;
        return res + 1;
    }
'''



'''
SHORTEST SOLUTION:


public int consecutiveNumbersSum(int N) {
    int ans = 0;
    for(int i = 1, n = N - 1; n >= 0; n -= ++i)
        if ((n % i) == 0) ans++;
    return ans;
}


Explaination:
let N = k + (k+1) + (k+2) + (k+3) + ... + (k+i-1) = i*k + (1+2+3+... + i-1)
let sum(i) = (1+2+3+...+i-1), then we have
N = sum(i) + i*k ==>i*k = N - sum(i)
Which means: for each i, we can calculate N-sum(i). If N-sum(i) can be divided by i, there is an answer with length i.
Because, let k = (N - sum(i)) / i, we can add an integer k to each of (0,1, 2, 3, 4, ...., i-1) to become (k, k+1, k+2, k+3,.... k + i-1)
that is: sum(i) + i * k = N

The naive solution is:

public int consecutiveNumbersSum(int N) {
    int sum = 0, ans = 0;
    for(int i = 1; sum < N; i++) {
        sum += i;
        if (((N-sum) % i) == 0)
            ans++;
    }
    return ans;
}

Which is the same as the first solution.
Obviously, the runtime is O(n^0.5).
'''




