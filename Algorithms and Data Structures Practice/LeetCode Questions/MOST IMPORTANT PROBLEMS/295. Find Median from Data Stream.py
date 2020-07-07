'''
295. Find Median from Data Stream
Hard

2569

47

Add to List

Share
Median is the middle value in an ordered integer list. 
If the size of the list is even, there is no middle value. 
So the median is the mean of the two middle value.

For example,
[2,3,4], the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to 
the data structure.
double findMedian() - Return the median of all elements so far.
 

Example:

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
 

Follow up:

If all integer numbers from the stream are between 0 and 100, how would 
you optimize it?

If 99% of all integer numbers from the stream are between 0 and 100, 
how would you optimize it?
'''


import heapq

class MedianFinder:

    def __init__(self):
        """
        Use 1 MAXHEAP for bottom half of numbers.
        Use 1 MINHEAP for top half of numbers. 
        """
        self.low = []  # maxheap
        self.high = [] # minheap
        

    def addNum(self, num: int) -> None:            
        if len(self.low) == len(self.high):
            # insert into low!
            # same size, insert into low.
            # compare against high, if its smaller, add to low, 
            # if bigger, pop high, insert into high, put popped element o fhigh into low
            insert = num
            if len(self.high) > 0 and self.high[0] < num:
                insert = heapq.heappop(self.high)
                heapq.heappush(self.high, num)
                
            heapq.heappush(self.low, -insert)             
        else:
            insert = num
            if  -self.low[0] > num:
                insert = -(heapq.heappop(self.low))
                heapq.heappush(self.low, -num)
                
            heapq.heappush(self.high, insert)
        
    def findMedian(self) -> float:
        if len(self.low) == len(self.high):
            # print("self.low", self.low)
            # convert maxh val from neg -> pos    
            med = (-self.low[0] + self.high[0])/2
            return med
        else:
            return -self.low[0]

'''

all numbers between 0 and 100
just have an array of size 101, 

keep a pointer on the middle element.

also keep track of even and odd counts
when its before pointer, increment pointer by 1, 
when its after pointer. 

increment each index by 1 when you see it. 
keep count of total elements. 

'''


