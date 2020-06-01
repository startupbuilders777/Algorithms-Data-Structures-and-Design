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
import itertools


from collections import Counter

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        
        '''
        Count them up. 
        
        then initialize minheap of size K. 
        
        insert first k.
        Then only insert if its bigger than the current min frequent element. 
        
        O(n) + Nlog(K) 
        
        return elements in heap. 
        '''
        
        counted_items = list(Counter(nums).items())        
        pq = []
        
        count = itertools.count()
        
        for i in range(k):
            key, value = counted_items[i]
            heappush(pq, [value, next(count), key])
    
        j = k        
        min_freq, _, _ = pq[0]
        
        while j < len(counted_items):
            k, v = counted_items[j]
            
            if(v > min_freq):
                _ = heappop(pq)
                heappush(pq, [v, next(count), k])
                min_freq, _, _ = pq[0] # get new min freq
            j += 1
        
        return list(map(lambda x: x[-1], pq))


### Other cool ways to do it:


### BUCKET SORT:

# Otherwise I wouldn't have posted it in a thread that's about O(n). 
# But thanks for making me look again, I now replaced bucket[len(nums) - freq] 
# with bucket[-freq], no idea why I hadn't used that right away.


def topKFrequent(self, nums, k):
    bucket = [[] for _ in nums]
    for num, freq in collections.Counter(nums).items():
        bucket[-freq].append(num)
    return list(itertools.chain(*bucket))[:k]

# Other bucket sort. 
class Solution:
    def topKFrequent(self, nums, k):
        bucket, res = [[] for _ in range(len(nums) + 1)], []
        for a, b in collections.Counter(nums).items():
            bucket[b].append(a)
        for l in bucket[::-1]:
            if len(l): res += l
            if len(res) >= k: return res[ : k]


## USE QUICK SELECT
class Solution:
    def partition(self, nums, start, end):
        def swap(i, j):
            temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
        
        temp = random.randint(start, end)
        swap(temp, end)
        pivot = end
        
        it_w = start
        it_r = start
        while it_r < end:
            if nums[it_r] > nums[pivot]:
                swap(it_r, it_w)
                it_w += 1
            it_r += 1
        swap(it_w, pivot)
        return it_w
        
    def quickSelect(self, nums, start, end, k):
        if start == end: return nums[start]
        pivot = self.partition(nums, start, end)
        if pivot == k: return nums[pivot]
        elif pivot < k: return self.quickSelect(nums, pivot + 1, end, k)
        else: return self.quickSelect(nums, start, pivot - 1, k)
    
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = dict()
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        freq = list(count.values())
        threshold = self.quickSelect(freq, 0, len(freq)-1, k)
        
        result1 = []
        result2 = []
        for num in count:
            if count[num] > threshold: result1.append(num)
            elif count[num] == threshold: result2.append(num)
        return (result1 + result2)[0:k]


#### USE QUICK SELECT:
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        count = Counter(nums)
        unique = list(count.keys())
        
        def partition(left, right, pivot_index) -> int:
            pivot_frequency = count[unique[pivot_index]]
            # 1. move pivot to end
            unique[pivot_index], unique[right] = unique[right], unique[pivot_index]  
            
            # 2. move all less frequent elements to the left
            store_index = left
            for i in range(left, right):
                if count[unique[i]] < pivot_frequency:
                    unique[store_index], unique[i] = unique[i], unique[store_index]
                    store_index += 1

            # 3. move pivot to its final place
            unique[right], unique[store_index] = unique[store_index], unique[right]  
            
            return store_index
        
        def quickselect(left, right, k_smallest) -> None:
            """
            Sort a list within left..right till kth less frequent element
            takes its place. 
            """
            # base case: the list contains only one element
            if left == right: 
                return
            
            # select a random pivot_index
            pivot_index = random.randint(left, right)     
                            
            # find the pivot position in a sorted list   
            pivot_index = partition(left, right, pivot_index)
            
            # if the pivot is in its final sorted position
            if k_smallest == pivot_index:
                 return 
            # go left
            elif k_smallest < pivot_index:
                quickselect(left, pivot_index - 1, k_smallest)
            # go right
            else:
                quickselect(pivot_index + 1, right, k_smallest)
         
        n = len(unique) 
        # kth top frequent element is (n - k)th less frequent.
        # Do a partial sort: from less frequent to the most frequent, till
        # (n - k)th less frequent element takes its place (n - k) in a sorted array. 
        # All element on the left are less frequent.
        # All the elements on the right are more frequent.  
        quickselect(0, n - 1, n - k)
        # Return top k frequent elements
        return unique[n - k:]

