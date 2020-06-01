'''
121. Best Time to Buy and Sell Stock

Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction 
(i.e., buy one and sell one share of the stock),
design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
'''

class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        '''
        YOU CANT DO THIS INCORRECT!!!

        buyPrice = float("inf")
        sellPrice = 0
        
        for i in prices:
            if(i < buyPrice):
                buyPrice = i
            elif(i > sellPrice):
                sellPrice = i
            
        if(sellPrice > buyPrice):
            return sellPrice - buyPrice
        else: 
            return 0
            
        '''        
        
        '''
        ok run thru. 
        if you find a smallest price, start j going to right
        to find biggest price before you reach another smallest price.
        
        record that as current largest.
        then start again from the next smallest price -> run it right to get 
        its largest buy price.
        update current largest.
        return current largest at end of array.
        
        '''
        
        buy = float("inf")
        sell = 0
        # runningMaxProfit = 0 (can be derived from sell - buy)
        
        bestProfit = 0
        
        for i in prices:
            print("curr buy and sell are", (buy, sell))
            if(i < buy):
                # update bestProfit, 
                # then set a new running max profit
                if((sell-buy) > bestProfit):
                    bestProfit = (sell-buy)
                
                buy = i
                sell = 0
            elif(i > sell):
                sell = i
        
        if((sell-buy) > bestProfit):
            bestProfit = (sell-buy)
        
        return bestProfit
