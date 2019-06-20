class Solution:
    def majorityElement(self, nums):
        # BOYER MOORE VOTING ALGORITHM!
        
        
        # CANT WE USE QUICK SELECT? yes we can! w/ median of medians. 
        
        counter = 1
        # we look at majority, add 1 if we see the same letter again, subtract 1 if we dont
        # keep going. if the counter becomes 0, 
        # look at the next character, check to see if that is a max. 
        
        curr_major = nums[0]
        for i in range(1, len(nums)):
            
            if(nums[i] == curr_major):
                counter += 1
            else:
                if(counter > 0):
                    counter -= 1
                else:
                    counter = 1
                    curr_major = nums[i]
        
        return curr_major
        
