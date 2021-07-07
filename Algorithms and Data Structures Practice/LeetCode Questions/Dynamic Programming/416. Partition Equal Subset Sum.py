
'''
416. Partition Equal Subset Sum

Given a non-empty array containing only positive integers, 
find if the array can be partitioned into two subsets such 
that the sum of elements in both subsets is equal.

Note:

Each of the array element will not exceed 100.
The array size will not exceed 200.
 

Example 1:

Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
 

Example 2:

Input: [1, 2, 3, 5]

Output: false

Explanation: The array cannot be partitioned into equal sum subsets.
'''

class Solution(object):
    
    def canPartitionHarmanMethod(self, nums):
        '''
        The two equal subsets have to a sum that is total_sum / 2
        
        
        
        So we have to find if any set is equal to total_sum/2 right!
        
        
        # Then we can do include/dont include strategy recurison
        
        -> memoize 
        # Bottom up it looks like this:
        
        we memoize all the possible values we can make. 
        by adding or removing an element. 
        
        arr[1..n] contains all possible values we can make with this smaller set righ?
        
        Isnt this just testing every subset?
        
        Create all subsets with double for loops trick 
        but save only the sums we get!
        
        Throw away sums that go over. keep the ones that stay under. then move on to next element
        
        thats how we move forward from arr[1...n] -> arr[1..n+1]
        
        Base Case:
        
        A[1..0] = 0
        A[1..1] = 0, A[1]
        
        A[1..2] = 0, A[1], 0 + A[2], A[1] + A[2]
        
        -> we are just adding the element to every element. 
        -> then checking. is this even DP?
        
        A[1..3] = 0, A[1],  0 + A[2], A[1] + A[2], 0 + A[3], A[1] + A[3],  0 + A[2] + A[3], A[1] + A[2] + A[3], 
        
        
        Here is another kids analysis:
        
        This problem is essentially let us to find 
        whether there are several numbers in a set which are 
        able to sum to a specific value (in this problem, the value is sum/2).

        Actually, this is a 0/1 knapsack problem, for each number, we can pick it or not. 
        Let us assume dp[i][j] means whether the specific sum j can be gotten from the first i numbers. 
        If we can pick such a series of numbers from 0-i whose sum is j, dp[i][j] is true, otherwise it is false.

        Base case: dp[0][0] is true; (zero number consists of sum 0 is true)

        Transition function: For each number, if we don't pick it, dp[i][j] = dp[i-1][j], 
        which means if the first i-1 elements has made it to j, dp[i][j] would also make it to j 
        (we can just ignore nums[i]). If we pick nums[i]. dp[i][j] = dp[i-1][j-nums[i]], which 
        represents that j is composed of the current value nums[i] and the remaining composed of 
        other previous numbers. Thus, the transition function is dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i]]
        
        ok yeah its just a double for loop that enumerates all subsets:
        
        '''
        
        
        s = 0
        for i in nums:
            s += i
        # if sum is odd, it is impossible
        if(s % 2 == 1):
            return False
        
        half_sum = s/2
        
    
        
        
        # ALSO TRY TO SAVE SPACE. YOU ONLY NEED LAST 2 ROWS IF YOU THINK ABOUT IT!
        '''
        To save space, think about how next level relates to previous levels, and if the
        relationship only requires like 1 prev level, thats a place to save space!
        '''
        
        prev = [0]
        next_row = []
        i = 0
        N = len(nums)
            
        while i < N:
            curr = nums[i]
            for ele in prev:
                new_partial_sum = ele + curr
                if(new_partial_sum == half_sum):
                    return True
                if(new_partial_sum < half_sum):
                    next_row.append(new_partial_sum)
                
                
            prev = prev + next_row
            next_row = []
            i += 1
        
        return False
        
        
    def canFindSum(self, nums, target, ind, n, d):
        if target in d: return d[target] 
        if target == 0: d[target] = True
        else:
            d[target] = False
            if target > 0:
                for i in xrange(ind, n):
                    if self.canFindSum(nums, target - nums[i], i+1, n, d):
                        d[target] = True
                        break
        return d[target]
    
    def canPartition(self, nums):
        s = sum(nums)
        if s % 2 != 0: return False
        return self.canFindSum(nums, s/2, 0, len(nums), {})
        
        