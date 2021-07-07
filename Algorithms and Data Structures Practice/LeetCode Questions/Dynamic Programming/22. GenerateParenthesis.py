#COMPLETED 

'''
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:

[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]

'''
# ITERATIVE SOLUTION:

class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        # create all !
        
        '''
        ok so 3 n//2 open, 
        and n//2 closed.
        
        once you run out of open, you have to put closed. 
        '''
        
        result = []
        stack = [(0, 0, "")]
        
        while stack:
            o, c, piece = stack.pop()
            if o == n and c == n:
                result.append(piece)  
            if c < o:
                stack.append((o, c+1, piece + ")"))
            if o < n:
                stack.append((o+1, c, piece + "("))
        return result


# MY SOLUTION
class Solution:
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        # "(" is an open paren
        # ")" is a closing paren
        
        # can we save partial openParen, closeParen solutions, because both 
        # will go toward 0
        
        # add dynamic programming to speed up
        
        cache = {}
        def paren(n, openParen):
            
            if(cache.get((n, openParen)) is not None ):
                return  cache.get((n, openParen))
        
            if(n == 0 and openParen == 0):
                return [""]
            
            # print an open paren:
            return1 = []
            return2 = []
            if(openParen > 0):
                result = paren(n, openParen - 1)
                return1 = ["(" + i for i in result]
                    
            # print a close paren if there are open parens that need to be closed. this is equal to n - openParen
            # once we close, we subtract close Paren by 1. however, this doesnt work.
            
            if(n - openParen > 0):
                result = paren(n-1, openParen)
                return2 = [")" + i for i in result]
            
            cache[(n, openParen)] = [*return1, *return2]
            return cache[(n, openParen)]
            
        return paren(n, n)    


'''
LEETCODE SOLUTION: BACKTRACKING:

'''

class Solution(object):
    def generateParenthesis(self, N):
        ans = []
        def backtrack(S = '', left = 0, right = 0):
            if len(S) == 2 * N:
                ans.append(S)
                return
            if left < N:
                backtrack(S+'(', left+1, right)
            if right < left:
                backtrack(S+')', left, right+1)

        backtrack()
        return ans


'''
ENUMERATIONS AND CLOSURE NUMBER SOLUTION:
Intuition

To enumerate something, generally we would like to express 
it as a sum of disjoint subsets that are easier to count.

Consider the closure number of a valid parentheses sequence S: 
the least index >= 0 so that S[0], S[1], ..., S[2*index+1] is valid. 
Clearly, every parentheses sequence has a unique closure number. We can try to enumerate them individually.

Algorithm

For each closure number c, we know the starting and ending brackets must be at index 0 and 2*c + 1. 
Then, the 2*c elements between must be a valid sequence, plus the rest of the elements must be a valid sequence.

'''

class Solution(object):
    def generateParenthesis(self, N):
        if N == 0: return ['']
        ans = []
        for c in range(N):
            for left in self.generateParenthesis(c):
                for right in self.generateParenthesis(N-1-c):
                    ans.append('({}){}'.format(left, right))
        return ans
