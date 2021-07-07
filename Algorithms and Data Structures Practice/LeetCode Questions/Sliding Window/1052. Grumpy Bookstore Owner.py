'''
1052. Grumpy Bookstore Owner
Medium

149

16

Favorite

Share
Today, the bookstore owner has a store open for customers.length minutes.  Every minute, some number of customers (customers[i]) enter the store, and all those customers leave after the end of that minute.

On some minutes, the bookstore owner is grumpy.  If the bookstore owner is grumpy on the i-th minute, grumpy[i] = 1, otherwise grumpy[i] = 0.  When the bookstore owner is grumpy, the customers of that minute are not satisfied, otherwise they are satisfied.

The bookstore owner knows a secret technique to keep themselves not grumpy for X minutes straight, but can only use it once.

Return the maximum number of customers that can be satisfied throughout the day.

 

Example 1:

Input: customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], X = 3
Output: 16
Explanation: The bookstore owner keeps themselves not grumpy for the last 3 minutes. 
The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.
'''


class Solution(object):
    def maxSatisfied(self, customers, grumpy, X):
        """
        :type customers: List[int]
        :type grumpy: List[int]
        :type X: int
        :rtype: int
        """
        
        
        '''
        Match customers coming in with 
        not grumpy
        
        ok go througgh customers array. set elments to 0 if grumpy is 0
        
        Then we find the max interval from left to right pass on this 
        fixed customers array, with interval length X!
        
        
        as we move left to right, keep running sum, as well as 
        var for first element, reassign first element when you move to the right.
        
        
        '''
        
        sad_customers = []
        result = 0
        for idx, i in enumerate(grumpy):
            if i == 0:
                sad_customers.append(0)
                result += customers[idx]
            else:
                sad_customers.append(customers[idx])
                
        i = 0
        j = X-1
        
        runningMax = 0
        for k in range(X):
            runningMax += sad_customers[k]
        
        currMax = runningMax
        
        while j+1 < len(sad_customers):
            runningMax -= sad_customers[i]  
            i += 1
            j += 1
            
            runningMax += sad_customers[j]
            currMax = max(runningMax, currMax)
        
        # ok find max now:
        return result + currMax
        
                
                
                
                
                
                
                
                
                
                
                
                