'''
There are N children standing in a line. Each child is assigned a rating value.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
What is the minimum candies you must give?

Example 1:

Input: [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
Example 2:

Input: [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
             The third child gets 1 candy because it satisfies the above two conditions.
'''

# MY SOLUTION:

class Solution:
    def candy(self, ratings: List[int]) -> int:
        
        '''
        [1,3,2,2,1]
        answer ->
        1, 2, 1, 2, 1
        if the values are the same, you can reset it to 1!
        [1,6,10,8,7,3,2]
        1, 2, 3, 4  3  2  1  

        [1,3,4,5,2]
        1 2 3 4 1
        '''
        
        # idx -> candy
        
        m = {}
        minVal = 0 
        
        # first kid gets 1 candy
        # m[0] = 1
        
        # this will cause decreasing array to be called on first element. 
        prevRate = float("inf") # ratings[0]
        arr = []
        
        def appendDecreasingKids(arr, m, reverseIdx):
            k = reverseIdx
            rate = arr.pop()
            m[k] = m.get(k, 1) # it might have an assigned value already, dont lose it!
            
            while arr:
                dk = arr.pop() # pop from end 
                # check the left side, thats why we take max, 
                # because array could have a previous constraint
                # from a child on its left side, (in the left to right direction)
                m[k-1] = max(m.get(k-1, 1), m[k] + 1)
                rate = dk
                k -= 1
                
        for i in range(0, len(ratings)): 
            rate = ratings[i]
            if rate > prevRate:
                # ok if arr has elements, then reverse set values for them!
                if arr: 
                    appendDecreasingKids(arr, m, i-1)
                m[i] = m[i-1] + 1
                arr.append(rate) # we are appending it because the right neighbor might still be bigger!
            elif rate < prevRate:
                # ok so now we get a decreasing array, we need to find its end, and set the end element to 1
                arr.append(rate)
            else:
                # if the rates are equal, empty out your array and start a new one
                # because when elements are the same, then one of the elements 
                # can take the value 1
                if arr: 
                    appendDecreasingKids(arr, m, i-1)
                arr.append(rate)                
            prevRate = rate

        if arr: 
            appendDecreasingKids(arr, m, len(ratings) - 1)
            
        return sum(m.values())

# EASIER 2 PASS:

class Solution:
    # @param ratings, a list of integer
    # @return an integer
    # 5:46
    def candy(self, ratings):
        if not ratings:
            return 0

        n = len(ratings)
        candy = [1] * n
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candy[i] = candy[i - 1] + 1
            
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1] and candy[i] <= candy[i + 1]:
                candy[i] = candy[i + 1] + 1

        return sum(candy)

# REALLY HARD CONSTANT SPACE SOLN THAT USES SLOPES:

class Solution:
    def candy(self, ratings):
        if not ratings:
            return 0

        count = up = down = 1

        for i in range(1, len(ratings)):
            if ratings[i] >= ratings[i - 1]:
                if down > 1:
                    count -= min(down, up) - 1
                    up, down = 1, 1
                up = ratings[i] == ratings[i - 1] or up + 1
                count += up
            else:
                down += 1
                count += down

        if down > 1:
            count -= min(down, up) - 1

        return count
    
    


# EASIER AND BETTER SOLUTIONS?

# HERES A SIMPLE ONE, 2 PASS, RESPECT CONSTRAINTS FROM 
# BOTH SIDES

'''
 int candy(vector<int> &ratings)
 {
	 int size=ratings.size();
	 if(size<=1)
		 return size;
	 vector<int> num(size,1);
	 for (int i = 1; i < size; i++)
	 {
		 if(ratings[i]>ratings[i-1])
			 num[i]=num[i-1]+1;
	 }
	 for (int i= size-1; i>0 ; i--)
	 {
		 if(ratings[i-1]>ratings[i])
			 num[i-1]=max(num[i]+1,num[i-1]);
	 }
	 int result=0;
	 for (int i = 0; i < size; i++)
	 {
		 result+=num[i];
		// cout<<num[i]<<" ";
	 }
	 return result;
 }
'''
