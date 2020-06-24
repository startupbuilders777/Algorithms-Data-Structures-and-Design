'''
128. Longest Consecutive Sequence
Hard

3186

173

Add to List

Share
Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(n) complexity.

Example:

Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.

'''

from collections import defaultdict

class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        # Every node will only have 1 child!
        
        m = {}
        
        # create directed graphs. then walk along them, see how far you can get aka the height!
        
        # we need start keys, aka, smallest elements
        # start with all keys, and remove the ones we see. 
        # there should not be any cycles, -> 
        roots = set(nums)
        exists = set()
        
        for i in nums:            
            if i not in exists: 
                exists.add(i)
            else:
                # already processed, saw same node
                continue
                
            if i-1 in exists:
                m[i-1] = i
                roots.remove(i)
            if i+1 in exists:
                m[i] = i + 1
                roots.remove(i+1)
        
        longest = 0        
        for i in roots:
            l = 0
            node = i
            while True:
                l += 1
                node = m.get(node)
                if node == None:
                    break
                    
            longest = max(l, longest)
        return longest

# BETTER SOLUTION:

'''
This optimized algorithm contains only two changes from 
the brute force approach: the numbers are stored in a HashSet (or Set, in Python) 
to allow O(1)O(1) lookups, and we only attempt to build sequences from numbers that are 
not already part of a longer sequence. This is accomplished by first ensuring that the 
number that would immediately precede the current number in a sequence is not present, 
as that number would necessarily be part of a longer sequence.
'''

class Solution:
    def longestConsecutive(self, nums):
        longest_streak = 0
        num_set = set(nums)

        for num in num_set:
            if num - 1 not in num_set:
                current_num = num
                current_streak = 1

                while current_num + 1 in num_set:
                    current_num += 1
                    current_streak += 1

                longest_streak = max(longest_streak, current_streak)

        return longest_streak

# BEST SOLUTION
# WALK THE STREAK:

def longestConsecutive(self, nums):
    nums = set(nums)
    best = 0
    for x in nums:
        if x - 1 not in nums:
            y = x + 1
            while y in nums:
                y += 1
            best = max(best, y - x)
    return best

