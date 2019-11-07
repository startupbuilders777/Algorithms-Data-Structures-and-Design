'''


347. Top K Frequent Elements
Medium

1956

126

Favorite

Share
Given a non-empty array of integers, return the k 
most frequent elements.

Example 1:

Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
Example 2:

Input: nums = [1], k = 1
Output: [1]
Note:

You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than
O(n log n), where n is the array's size.


'''

    
from heapq import *
import collections 
import itertools

class Solution(object):

    
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # Count all the elements. then do  quick select 
        # on the count for k most frequent elements.
        # Then extract 
        
        # using bisect prolly write!
        
        
        # Use a min heap. Insert into min heap if 
        #amount of elements is less than K or 
        #top element is less than an element you 
        # are currently looking at it. 
        
        
        
        # count elements
        
        
        num_of_items_to_return = k
        m = collections.defaultdict(int)
        
        for i in nums:
            m[i] += 1
        
        
        pq = [] # heapq
        counter = itertools.count()
        
        # entry_finder = {} Used for deleting other elements in heapq!
        
        for k, v in m.items():
           
            if len(pq) < num_of_items_to_return:
                count = next(counter)
                i = [v, count, k] #[priority, count, task]
                heappush(pq, i)
            else:
                top =  pq[0][0] # get priority
                print("TOP IS", top)

                if v > top:
                    _ = heappop(pq)
                    
                    
                    count = next(counter)
                    i = [v, count, k] #[priority, count, task]
                    
                    heappush(pq, i)
                    
        return map(lambda x: x[-1], pq)
                
            