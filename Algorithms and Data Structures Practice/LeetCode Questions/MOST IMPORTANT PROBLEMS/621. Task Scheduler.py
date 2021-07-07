'''
621. Task Scheduler
Medium

Given a char array representing tasks CPU need to do. 
It contains capital letters A to Z where different letters 
represent different tasks. Tasks could be done without original 
order. Each task could be done in one interval. For each interval, 
CPU could finish one task or just be idle.

However, there is a non-negative cooling interval n that means between 
two same tasks, there must be at least n intervals that CPU are doing 
different tasks or just be idle.

You need to return the least number of intervals the CPU will 
take to finish all the given tasks.


Example:

Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: A -> B -> idle -> A -> B -> idle -> A -> B.
 

Note:

The number of tasks is in the range [1, 10000].
The integer n is in the range [0, 100].

'''


from collections import Counter

class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        
        '''
        Ok just get the most frequently occuring item
        multiply by n = 2: 
        
        these are number of intervals between them.
        
        take max of this and 
        
        just the total count of items in the array!
        '''
        
        # Constant space is achieved because there is 
        # only 26 letters in the alphabet!
        '''
        SOLUTION:
        
        just find the most frequently occuring item.
        count how many are at the same max frequency (call this same count)!
        
        Then its:
        Taska,b,c <----> Taska,b,c <---->....Taska,b,c
        say a,b,c are the most freq occuring.
        
        Number of intervals is n-1, 
        then we add in the tasks between the intervals,
        the last most frequently occuring task will be our bottleneck
        since we have to wait for its cooldown (last one is c in this case.)
        
        answer: 
        (maxCount - 1)*intervalLength + maxCount + (sameCount-1)
          
        '''
        
        c = Counter(tasks)
        most_frequent = None
        count = 0
        sameCount = 0 
        for k,v in c.items():
            if v > count:
                most_frequent = k
                count = v
                sameCount = 1
            elif v == count:
                sameCount += 1
    
        
        print("Most freq", (most_frequent, count))
        
        return max(len(tasks), count + n*(count-1) + sameCount - 1 )
