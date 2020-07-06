'''
1029. Two City Scheduling
Easy

1295

166

Add to List

Share
There are 2N people a company is planning to interview. The cost of flying the i-th person to city A is costs[i][0], and the cost of flying the i-th person to city B is costs[i][1].

Return the minimum cost to fly every person to a city such that exactly N people arrive in each city.

 

Example 1:

Input: [[10,20],[30,200],[400,50],[30,20]]
Output: 110
Explanation: 
The first person goes to city A for a cost of 10.
The second person goes to city A for a cost of 30.
The third person goes to city B for a cost of 50.
The fourth person goes to city B for a cost of 20.

The total minimum cost is 10 + 30 + 50 + 20 = 110 to have half the people interviewing in each city.
 

Note:

1 <= costs.length <= 100
It is guaranteed that costs.length is even.
1 <= costs[i][0], costs[i][1] <= 1000
'''

class Solution:
    def twoCitySchedCost(self, costs: List[List[int]]) -> int:
        '''
        
        Choose the lowest N, in each array
        if a person appears in both lowests, 
        take the next most expensive one. 
        
        
        Assume everyone goes to city A. 
        
        then move people to city B
        move the people that cause the highest reduction in difference. 
        
        When trying to reduce space in algo, realize what you actually need from question
        and throw away all the miscellaneous bs information you are collecting.
        Such as for minimum cost, you should be able to do it with little space because 
        you dont care about the exact nodes you want etc. Its just a traversal. Reduce
        space usage!!!
        
        '''
        
        differences = sorted(map(lambda x: x[1] - x[0], costs)) 
        
        minCost = sum(map(lambda x: x[0], costs))

        for j in range(len(costs)//2):
            minCost += differences[j] 
        
        return minCost

        