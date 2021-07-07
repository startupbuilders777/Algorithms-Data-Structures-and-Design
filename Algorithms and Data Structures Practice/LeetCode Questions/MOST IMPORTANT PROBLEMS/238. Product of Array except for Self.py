'''
238. Product of Array Except Self
Medium

3555

306

Add to List

Share
Given an array nums of n integers where n > 1,  
return an array output such that output[i] is equal 
to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? 
(The output array does not count as extra space for 
the purpose of space complexity analysis.)
'''

class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # without division
        # O(n)
        
        # Constant space, Output array does not count as extra space. 
        
        
        '''
        collect all elements in set.
        remove element in set and multiply?
        
        thats O(n^2)
        
        vali = product / elei
        
        vali * elei = product
        
        
        
        Get total product. 
        
        Get partial product
        divide? 
        
        product on right side
        product on left side. 
        Multiply!!

        Repeat for each.
        
        Let both sides run. 
        
        1 pointer left side
        1 pointer right side. 
        Input:                          [1,2,3,4]
        Right pointer product array ->  [1, 2, 6, 24]
        left pointer product array ->   [ 24, 24 ,12 ,4]
        Output:                         [24,12,8,6]
        you take the left element from right product 
        and right element from left product array!
        
        to reduce space, output array initally contains left product array. 
        and keep running right pointer array in one variable.
        we only need two, and we keep updating right pointer
        
        to get product of an element.
        multiply element from ripht and left. 
        
        '''
        
        
        out = [1] * len(nums)
        # create left product array
        left_product = 1
        for i in range(len(nums) - 1, -1, -1):
            left_product *= nums[i]
            out[i] = left_product
        
        print(out)
        
        right_running_prod = 1
        
        
        for i in range(len(nums)):
            
            left_running_prod = 1
            if(i + 1 < len(nums)):
                left_running_prod = out[i+1]

            out[i] = right_running_prod * left_running_prod
            right_running_prod *= nums[i]
        
        # pop last element which was 1
        # out.pop()
        
        return out
        
            
            
            
        
        