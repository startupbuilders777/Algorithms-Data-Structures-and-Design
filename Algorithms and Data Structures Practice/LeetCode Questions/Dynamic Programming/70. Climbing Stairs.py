'''
70. Climbing Stairs
Easy

3217

107

Add to List

Share
You are climbing a stair case. It takes n steps to reach to the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

Note: Given n will be a positive integer.

Example 1:

Input: 2
Output: 2
Explanation: There are two ways to climb to the top.
1. 1 step + 1 step
2. 2 steps
Example 2:

Input: 3
Output: 3
Explanation: There are three ways to climb to the top.
1. 1 step + 1 step + 1 step
2. 1 step + 2 steps
3. 2 steps + 1 step

'''


'''
BOTTOM UP DP
'''

class Solution(object):
    
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        '''
        
        BOTTOM UP. 
        
        We know how many when you 
        
        
        F[N] = F[N-1] + F[N-2]
        
        so need base cases
        F[0] = 0
        F[1] = 1
        F[2] = 2
        
        then can compute 
        F[3] = F[] 
        F[4]
        easy
        
        For space optimization only need 2 variables. 
        
        '''    
        
        if n == 0:
            return 0
        elif n == 1:
            return 1
        elif n == 2: 
            return 2
        
        prev = 1
        next = 2
        temp = 0
        
        i = 2
        
        while i < n:
            temp = next
            next = prev + next
            prev = temp
            i += 1
            
        return next


'''
FASTER PYTHON SOLUTIOONS

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 2:
            return n
        
        ways = [0] * (n+1)
        ways[1] = 1
        ways[2] = 2
        for i in xrange(3, n+1):
            for step in [1, 2]:
                ways[i] += ways[i-step]
        return ways[-1]

class Solution(object):
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        
        # bottom up
        if n ==     0 or n == 1:
            return 1
        num_ways = [0] * (n+1)
        num_ways[0] = 1
        num_ways[1] = 1
        for i in range(2, n+1):
            num_ways[i] = num_ways[i-1] + num_ways[i-2]
        
        return num_ways[n]
        
#         top down approach      
#         memo = [0] * (n + 1)
#         def solve(n, memo):
#             if n == 0 or n == 1:
#                 return 1
            
#             # check if answer already exists
#             if memo[n] > 0:
#                 return memo[n]
            
#             memo[n] = solve(n-1, memo) + solve(n-2, memo)
#             return memo[n]
        
#         return solve(n, memo)
        


'''