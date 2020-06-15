# DONE

'''
Given n non-negative integers representing an elevation map where the width of 
each bar is 1, compute how much water it is able to trap after raining.


The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. 
In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

'''


class Solution(object):
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        
        '''
        
        You keep 2 pointers.
        
        
        increment first pointer to first height
        
        increment second pointer to second height.
        capture water. 
        
        move first pointer to second pointer.
        increase second pointer until you reach height bigger than first or end.
        if you reach height bigger, then execute capture. 
        then repeat. if you reach end, move first pointer up one, and try again 
        
        
        '''
        
        i = 0
        j = 0
        
        first = 0
        water = 0
        
        while i < len(height):
            
            if(height[i] == 0):
                i += 1
                continue
            first = height[i]
            
            j = i+1
            reached_end = True
            
            max_j = j  
            running_second  = 0
            
            while j < len(height):
                if(height[j] >= height[i]):
                    second = height[j]
                    # now we need to calc trapped rain water
                    blocks = sum([height[k] for k in range(i+1, j)])
                    trapped_water = (j - i - 1) * min(first, second) - blocks
                    water += trapped_water
                    i = j # Move first pointer up!
                    reached_end = False
                    break
                else:
                    if(height[j] > running_second ):
                        running_second = height[j]
                        max_j = j
                
                j += 1
                           
            if(reached_end):
                j = max_j
                second = running_second
                blocks = sum([height[k] for k in range(i+1, j)])
                trapped_water = (j - i - 1) * min(first, second) - blocks
                water += trapped_water
                i = j
                
                # i += 1
                        
            
            
        return water

'''

SOLUTIONS:

STACK SOLUTION!

'''
class Solution:
    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        #initialize the  two variables required
        stk = []
        waterCollected = 0

        #iterate over the whole array of bars
        for i in range(0,len(height)):

            #while there is something  in the stack and the current bar's height is more than 
            #the height of index of the stack's top value
            while stk and height[i] > height[stk[-1]]:

                #let's pop the top of the stack and call this index as top
                top = stk.pop()

                #check if the last pop results in underflow --> break 
                if not len(stk):
                    break

                #calculate the distance from current bar to the top of the stack's element index
                #remember this is different from the element we popped; due to the while loop
                # we are trying to touch all the elements in the stack which are smaller than current
                distance = i - stk[-1] - 1

                #now find out the water collected between the current and stack's last 
                #element index's height--> only the minimum of these two will help us 
                #determine the water collected. Again, we need not worry about the bars in 
                #between as they were already covered in the while loop. We're concerned with 
                #two bars here, the distance between them and the water trapped between them 
                #and above the height of top element's height. 
                waterBetweenBars = min(height[i], height[stk[-1]]) - height[top]

                #add each iterative waterBetweenBars collected to the result
                waterCollected += distance*waterBetweenBars 

            #if the height of the current bar is less than or equal to height[stk[-1]]
            #add that index to the stack
            stk.append(i)

        #return the default value or whatever was calculated
        return waterCollected

