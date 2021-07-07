'''
746. Min Cost Climbing Stairs
Easy

1848

421

Add to List

Share
On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).

Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.

Example 1:
Input: cost = [10, 15, 20]
Output: 15
Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
Example 2:
Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
Output: 6
Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
Note:
cost will have a length in the range [2, 1000].
Every cost[i] will be an integer in the range [0, 999].

'''

class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        
        '''
        Base Case:
        start at step 0 or step 1
        
        
        I[x] = min cost to reach step x
        return I[-1]
    
        cost of reaching step 2  is min of step 0 and 1 + cost of using step 2
        
        Bottom up
        I[3] = min(I[2], I[1]) + cost(I(X))
        
        Top down
        start from step 0, go to 1 or 2 cost is 0, pay cost of 0.
        then repeat until you reach end.
        take min of both paths. 
        
        return cost of using that step. 
        
        
        
        
        '''
        
        costs = [0] + cost
        print("cost", cost)
        print("costs", costs)
        m = {}
        
        def helper(idx):
            print(idx)
            
            nonlocal m
            if(m.get(idx)):
                return m[idx]
            
            if(idx >= len(costs)):
                return 0
            
            stepCost = min(helper(idx + 1), helper(idx + 2)) + costs[idx]
            m[idx] = stepCost
            return m[idx]
        
        return helper(0)
