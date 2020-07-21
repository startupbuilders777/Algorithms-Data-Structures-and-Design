'''
309. Best Time to Buy and Sell Stock with Cooldown
Medium

2364

82

Add to List

Share
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)
Example:

Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]

'''

class Solution:
    # TOP DOWN ACCEPTED SOLUTION
    def maxProfit(self, prices: List[int]) -> int:
        
        @lru_cache(maxsize=None)
        def helper(i, bought, cooldown):
            
            if i == len(prices):
                return 0
            
            if bought == -1 and not cooldown:
                return helper(i+1, prices[i], False)
            
            if bought == -1 and cooldown:
                return helper(i+1, -1, False)
            
            if prices[i] < bought:
                return helper(i+1, prices[i], False)
            
            if prices[i] > bought:
                return max(helper(i+1, -1, True) + (prices[i] - bought), helper(i+1, bought, False) )
            
            if prices[i] == bought:
                return helper(i+1, bought, False)
            
        return helper(0, -1, False)
                
'''
STEFAN FINITE STATE MACHINE. 
'''

def maxProfit(self, prices):
    free = 0
    have = cool = float('-inf')
    for p in prices:
        free, have, cool = max(free, cool), max(have, free - p), have + p
    return max(free, cool)
'''
free is the maximum profit I can have while being free to buy.
have is the maximum profit I can have while having stock.
cool is the maximum profit I can have while cooling down.
'''

'''
First, in Python iteration, writings variable assignment in one 
line assures all values to be written concurrently. i.e.



Second, (now we understand all three values are updated at the same time), 
there are 3 states if you can ever be in as we iteration through the day.

case 1 (free): free to trade (free last round or cool-down last round)
case 2 (have): not free to trade because bought in the current iteration (last round must be free)
case 3 (cool): not free to trade because sold in the current iteration (last round must be having)
Third, if we understand the 3 case definition, we should be able to write down the update step without any issue.

        free = max(free, cool)
        have = max(have, free - p)   # if we were free last round and just bought, then our profit(in balance) need to adjust because buying cost money
        cool = have + p # to be in cool-down, we just sold in last round (realizing profit), then profit would increase by the current price
        
really, if you think of profit like balance, it would make more sense. have + p2 = (free -p1 + p2) = free + (p2 - p1). 
Notice that p1 and p2 are two different p because we can't be in have state and at the same time in cool state.


Last, now everything should be very clear, but how we initialize things and what to return are also important.
What to return is easier to answer because if we having stock, we can't be at maximum profit. Therefore, 
the only two states we can be when we are at a max profit would be either a cool-down state or in the free state.

What to initialize requires us to think about the definition for each state again. If we initialize 
have and cool to be 0, then we would be saying, by default, we have already bought the stock at price 0. 
That doesn't make sense. Think about this: we can only be in have if we were free and we can only 
be in cool if we were in have. Therefore, only free can be 0, the other two must be -inf.        

'''


'''
NOT STATE MACHINE SOLUTION:

I am sharing my step-by-step process of how the idea is built up.

We know we have to make a choice on day[i], so we can think 
from the perspective of what determines the option of choices we have on day[i].

On day[i], we can choose cooldown, buy, or sell:

Under what condition we can choose to cooldown on day[i]?
It is obvious, there is not requirement. We can choose to cooldown on anyday.
Under what condition we can choose to buy a stock on day[i]?
The answer is we need make sure that we do not own any stock at end of 
day[i-2] because there is one day cooldown requirement.

Under what condition we can choose to sell a stock on day[i]?

The answer is we must own a stock at the end of day[i-1].

Now we can see the factors that determine the options of choices we have on day[i] is the status of 
whether we owned a stock previously. So, let own[i] represents the maximum profit for the first 
i days if we own a stock at the end of day[i]. not_own[i] represents the maximum profit for the 
first i days if we do not own a stock at the end of day[i].

Luckily, knowing own[i] and not_own[i] are enough because
1: The maximum profit can be derived from the two status because we either own or not own a stock at end of one day.
2: We can derive the the current status on day[i] if we know the previous status.

Finally, we can write up the equations:
own[i] = max(own[i-1], not_own[i-2] - prices[i])
not_own[i] = max(not_own[i-1], own[i-1] + prices[i])

In fact, the equations are the same of the idea of buy and sell, but I 
think the process to come up with the idea could be better understood by my approach.

'''

class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        res = 0
        own_last = -prices[0]
        not_own_last2 = 0
        not_own_last = 0
        for i in range(1, len(prices)):
            own = max(own_last, not_own_last2 - prices[i])
            not_own = max(not_own_last, own_last + prices[i])

            not_own_last2 = not_own_last
            own_last = own
            not_own_last = not_own
            res = max(res, max(own, not_own))
        return res

