'''
11. Container With Most Water
Medium

4781

514

Add to List

Share
Given n non-negative integers a1, a2, ..., an , 
where each represents a point at coordinate (i, ai). 
n vertical lines are drawn such that the two endpoints of line i 
is at (i, ai) and (i, 0). Find two lines, which together with x-axis 
forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

 



The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. 
In this case, the max area of water (blue section) the container can contain is 49.

 

Example:

Input: [1,8,6,2,5,4,8,3,7]
Output: 49

'''

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        '''
        two pointers. 
        
        
        Right ptr. go from left to right. 
        
        Only move to something on the right if 
        its bigger than what you had previously. 
        
        
        Left ptr. -> only go left if its bigger than what you had 
        previously. 
        
        These are the only candidates we have to test for max water. 
        
        When pointers switch sides, stop. 
        
        
        
        Also you have to move the pointer with the smaller height,
        because we wont get higher area by moving the other pointer, since 
        we are bottlenecked by the smaller height at all time since it is the min. 
        
        '''
        
        
        rightPtr = 0
        leftPtr = len(height) - 1
        currRight = height[rightPtr]
        currLeft = height[leftPtr]
        
        maxArea = min(currRight, currLeft) * (leftPtr - rightPtr)
        currArea = 0
        while True:
            
            if currRight == min(currRight, currLeft):
                # MOVE RIGHT PTR
                while True:
                    rightPtr += 1
                    if rightPtr >= leftPtr:
                        break
                        
                    if height[rightPtr] > currRight:
                        currRight = height[rightPtr]
                        currArea = min(currRight, currLeft) * (leftPtr - rightPtr)
                        maxArea = max(maxArea, currArea)
                            
                        break
                        
            else:
                # MOVE LEFT PTR
                while True:
                    leftPtr -= 1
                    if rightPtr >= leftPtr:
                        break
                    
                    if height[leftPtr] > currLeft:
                        currLeft = height[leftPtr]
                        currArea = min(currRight, currLeft) * (leftPtr - rightPtr)
                        maxArea = max(maxArea, currArea)
                        break
                    
                    
                   
        
            if rightPtr >= leftPtr:
                break
                
        return maxArea
                
                