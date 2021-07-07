'''
518. Coin Change 2
Medium

1483

54

Add to List

Share
You are given coins of different denominations and 
a total amount of money. Write a function to compute 
the number of combinations that make up that amount. 
You may assume that you have infinite number of each kind of coin.

 

Example 1:

Input: amount = 5, coins = [1, 2, 5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
Example 2:

Input: amount = 3, coins = [2]
Output: 0
Explanation: the amount of 3 cannot be made up just with coins of 2.
Example 3:

Input: amount = 10, coins = [10] 
Output: 1
 

Note:

You can assume that

0 <= amount <= 5000
1 <= coin <= 5000
the number of coins is less than 500
the answer is guaranteed to fit into signed 32-bit integer
'''

# MY SOLUTION

from collections import deque

class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        
        '''
        # recursively choose each element, keep going, etcs.
        # Do it bottom UP!!
        
        # You have to watch out. we need combinations of coins.
        # NOT PERMUTATIONS. 
        '''
        
        '''
        
        top down, you can choose 1 then 2, or choose 2 then 1, 
        which is same thing!!
        cant count twice, so we have to remove these paths. 
        remove by sorting the path then hashing. 
        then checking if its in the visited set. 
        
        Other DP way is this:
        select number of coins once!
        then cant add any more of that type of coin after, 
        move on to next type of coin.
        Only include paths that end at the total amount. 
        
        '''        
        
        # how much of each amount can we get?
        amounts = [0]*(amount + 1)
        amounts[0] = 1 # base case
        
        for i in coins:
            # We can only add coins once
            # then we have to go to next coin
            # this ENFORCES combinations invariant. 
            j = 1
            while j < (amount + 1):
                if(j - i >= 0):
                    amounts[j] += amounts[j - i]
        
                j += 1
        
        return amounts[-1]
    

# DP with a dic:

class Solution(object):
    def change(self, amount, coins):
        dic = {0: 1}
        for coin in coins:
            for j in range(amount + 1):
                dic[j] = dic.get(j, 0) +  dic.get(j - coin, 0)
        return dic.get(amount, 0)