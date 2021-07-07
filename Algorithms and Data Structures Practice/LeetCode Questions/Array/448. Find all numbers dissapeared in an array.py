'''
Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), 
some elements appear twice and others appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? 
You may assume the returned list does not count as extra space.

Example:

Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]

'''

# WITHOUT EXTRA SPACE SOLUTIONS:

#Solution without using Extra Space

#Can we avoid the set and somehow mark the input array which tells us what numbers are seen and what are not? We have additional information that the numbers are positive and numbers lie between 1 and N.
#Approach 1: Iterate the array and mark the position implied by every element as negative. Then in the second iteration, we simply need to report the positive numbers.
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number
        for i in xrange(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = - abs(nums[index])

        return [i + 1 for i in range(len(nums)) if nums[i] > 0]

#Approach 2: Iterate the array and add N to the existing number at the position implied by every element. This means that positions implied by the numbers present in the array will be strictly more than N (smallest number is 1 and 1+N > N). Therefore. in the second iteration, we simply need to report the numbers less than equal to N to return the missing numbers..
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        N = len(nums)
        for i in range(len(nums)):
            nums[(nums[i]%N)-1] += N
        return [i+1 for i in range(len(nums)) if nums[i]<=N]


#HARMANS SOLUTION: ALSO O(1) SPACE
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        # you could go through the array, and if an element is not in the correct index, swap it into
        # the correct index in the array. you
        # can transform the array into a counter, where you are counting the number of elements 
        # at each index. 
        # if you swap element 3 into index 2, and then you see element 2, then swap that into index 1
        # but index 1 already is correct, throw away the other side. keep the counter at index 1 0. 
        
        
        # then go through array again and check places where counter is 0, return that.
        

        
        i = 0
        while i < len(nums):
            val = nums[i]  
            # print("PROCESSING FOLLOWING ELEMENT", i)

            if(val != i+1):
                # not correct, fix it.
                # idx = i
                curr_pos = i
                nums[curr_pos] = 0
                while nums[val-1] != val:
                    if(val == 0):
                        break
                    # print("CURR ITEARTION, ARRAY IS", nums)
                    correct_position =  val - 1
                    # print("curr pos, curr val, correct pos", (curr_pos, val, correct_position))
                    
                    temp = nums[correct_position]
                    # print("curr tempp is", temp)
                    if(temp == val): # the val there is already correct!
                        # print("SAW THE val again. there are two of this val:: ", temp)
                        # nums[idx] = 0
                        nums[curr_pos] = 0
                        break #base case!
                    else:
                        # nums[curr_pos] = 0
                        nums[correct_position] = val
                        val = temp
                        curr_pos = correct_position
                  
                    
                        
            i += 1
        result = []
        print("nums", nums)
        for idx, i in enumerate(nums):
            if(i == 0):
                result.append(idx+1)
        
        return result




