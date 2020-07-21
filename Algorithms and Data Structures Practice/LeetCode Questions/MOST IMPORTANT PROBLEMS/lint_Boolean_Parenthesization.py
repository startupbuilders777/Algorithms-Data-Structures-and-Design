# NOT DONE 

'''
725. Boolean Parenthesization

Given a boolean expression with following symbols.

Symbols
    'T' ---> true 
    'F' ---> false 
And following operators filled between symbols

Operators
    &   ---> boolean AND
    |   ---> boolean OR
    ^   ---> boolean XOR 
Count the number of ways we can parenthesize the expression so that the value of expression evaluates to true.

Let the input be in form of two arrays one contains the symbols (T and F) in order and other contains operators (&, | and ^}

Example
Example 1:

Input： ['T', 'F', 'T']，['^', '&']
Output：2
Explanation：
The given expression is "T ^ F & T", it evaluates true, in two ways "((T ^ F) & T)" and "(T ^ (F & T))"
Example 2:

Input：['T', 'F', 'F']，['^', '|']
Output：2
Explanation：
The given expression is "T ^ F | F", it evaluates true, in two ways "( (T ^ F) | F )" and "( T ^ (F | F) )".

'''

class Solution:
    """
    @param symb: the array of symbols
    @param oper: the array of operators
    @return: the number of ways
    """
    def countParenth(self, symb, oper):
        
        
        AND = '&' in oper
        OR = '|' in oper
        XOR = '^' in oper
        
        
        # m = {}
        
        def helper(i, j):
            # index i and j of symbols.
            if i == j:
                return 0, 0
                # If the symbol is true return true = 1, false = 0
                # else, return true = 0, false = 1 
                #if symb[i] == "T":
                #    return  1, 0
                #else:
                #    return 0, 1
    
            # do we need this case. 
            if j == i + 1:
                if symb[i] == "T":
                    return  1, 0
                else:
                    return 0, 1
                # put an operator in the middle. 
                
            
            totalT = 0
            totalF = 0
            
            k = i + 1
            
            while k != j:
                
                leftT, leftF = helper(i, k)
                rightT, rightF = helper(k, j)
                
                # place an &:
                if AND:
                    totalT += leftT*rightT
                    totalF += leftT*rightF + leftF*rightT + leftF*rightF
                
                
                #place a xor:
                if XOR:
                    totalT += leftT*rightF + leftF*rightT
                    totalF += leftT*rightT + leftF*rightF
                
                #place a X
                if OR:
                    totalT += leftT*rightF + leftF*rightT + leftT*rightT
                    totalF += leftF*rightF
                
                k += 1
                
            return totalT, totalF
            
        return helper(0, len(symb))[0]
        
    
solve = Solution()

print(solve.countParenth("TFT", "^&"))