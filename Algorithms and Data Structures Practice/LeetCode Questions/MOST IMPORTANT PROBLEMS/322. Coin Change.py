'''

322. Coin Change

You are given coins of different denominations 
and a total amount of money amount. Write a 
function to compute the fewest number of coins 
that you need to make up that amount. If that 
amount of money cannot be made up by any combination 
of the coins, return -1.

Example 1:

Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1
Example 2:

Input: coins = [2], amount = 3
Output: -1
Note:

You may assume that you have an infinite number 
of each kind of coin.


'''

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        
        
        
        '''
        bottom up. 
        
        Base Case:
        
        A[1] = 1
        A[2] = 1
        A[5] = 1
        
        
        
        A[X] -> how many ways you can make X. 
        A[X] = min(A[X-5] + 1, A[X-2] + 1, A[X-1] + 1)
        
        return A[amount]
        
        '''
        
        # Allocate 1 extra space!
        
        A = [None] * (amount + 1)
        
        
        # Set up the base cases!
        A[0] = 0 # Another Base Case!
        
        for c in coins:
            if c < amount:
                A[c] = 1
        
        for i in range(1, amount + 1):
            # try to use one of the coins
            # if none of the coins lead to 
            # an A[i - CoinValue] that has a value
            # keep A[i] as none. 
            
            minVal = float("inf")
            valid = False
            for c in coins:
                if(i-c >= 0 and A[i - c] is not None and A[i-c] < minVal):
                    valid = True
                    minVal = A[i - c]
                    
                    
            if(valid):
                A[i] = 1 +  minVal
        
        if(A[amount] is None):
            return -1
        else:
            return A[amount]
        

# Really cool solution by using largest coins first, and pruning DFS tree. REALLY FAST!!!


class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        coins.sort(reverse = True)
        min_coins = float('inf')

        def count_coins(start_coin, coin_count, remaining_amount):
        nonlocal min_coins

        if remaining_amount == 0:
            min_coins = min(min_coins, coin_count)
            return

        # Iterate from largest coins to smallest coins
        for i in range(start_coin, len(coins)):
            remaining_coin_allowance = min_coins - coin_count
            max_amount_possible = coins[i] * remaining_coin_allowance

            if coins[i] <= remaining_amount and remaining_amount < max_amount_possible:
            count_coins(i, coin_count + 1, remaining_amount - coins[i])

        count_coins(0, 0, amount)
        return min_coins if min_coins < float('inf') else -1



# other well written solution

class Solution(object):
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        rs = [amount+1] * (amount+1)
        rs[0] = 0
        for i in xrange(1, amount+1):
            for c in coins:
                if i >= c:
                    rs[i] = min(rs[i], rs[i-c] + 1)

        if rs[amount] == amount+1:
            return -1
        return rs[amount]